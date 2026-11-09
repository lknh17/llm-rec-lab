"""训练与评估：token 流 → 定长块 → next-token 交叉熵 → AdamW；评估同时报每 token loss 和 bits-per-byte。

为什么要 bits-per-byte（bpb）：每 token loss 的分母是 token 数。词表越大，一个 token 装的字节越多，
每 token loss 天然越高；字节级分词器一个 token 只有 1 个字节，每 token loss 天然很低。
bpb = 总 NLL / ln2 / 原文字节数，分母换成"同一批文本的字节数"，不同分词器之间才可比。

L02 起：data_frac / epochs（只用训练流的前缀、重复几遍）、按算力预算定步数、constant 学习率、target_offset（教学故障）；
L03 起：MoE 负载均衡损失、训练稳定性统计；L04 起：梯度累积、fp16 与 GradScaler、doc mask（在模型里）。
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F


def to_stream(tok, docs: list[str], batch: int = 2000) -> np.ndarray:
    """文档 → [EOS, doc1..., EOS, doc2..., EOS, ...] 的一维 token 流（开头放 EOS，第一篇的第一个 token 也有预测）。"""
    parts = [np.array([tok.eos_id], dtype=np.int32)]
    for i in range(0, len(docs), batch):
        for ids in tok.encode_batch(docs[i:i + batch]):
            parts.append(np.asarray(ids, dtype=np.int32))
            parts.append(np.array([tok.eos_id], dtype=np.int32))
    return np.concatenate(parts)


# --8<-- [start:schedule]
def lr_at(step: int, total: int, cfg: dict) -> float:
    """线性 warmup，然后余弦降到峰值的 10%（cosine）或保持峰值不变（constant）。"""
    warm = max(1, int(total * cfg["warmup_frac"]))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    if cfg.get("lr_schedule", "cosine") == "constant":
        return cfg["lr"]
    p = (step - warm) / max(1, total - warm)
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))
# --8<-- [end:schedule]


def half_dtype(device, cfg: dict):
    """16 位计算精度：bf16（指数位和 fp32 一样 8 位，不会下溢）、fp16（指数只有 5 位，小于 6e-8 的数变成 0）或 None（fp32）。"""
    p = cfg["precision"]
    if p == "bf16" or (p == "auto" and device.type == "cuda"):
        return torch.bfloat16
    if p == "fp16":
        return torch.float16
    return None


def autocast(device, cfg: dict):
    dt = half_dtype(device, cfg)
    if dt is not None:
        return torch.autocast(device_type=device.type, dtype=dt)
    return torch.autocast(device_type=device.type, enabled=False)


def make_optimizer(model, cfg: dict):
    """权重衰减只给矩阵权重；norm 的缩放和 embedding 不衰减（embedding 被输出层共享，衰减它会压低 logits 尺度）。"""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 and "wte" not in n and "wpe" not in n else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8,
                             fused=torch.cuda.is_available())


# --8<-- [start:batch]
def make_batch(stream: np.ndarray, starts, T: int, offset: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """从 token 流里切 [B, T + 1] 的块：输入 x = 前 T 个，目标 y = 右移 offset 位的 T 个。
    offset = 1 是 next-token；offset = 0（教学故障）时 y 就是 x 本身，模型只要"抄"当前位置。"""
    blk = np.stack([stream[s:s + T + 1] for s in starts]).astype(np.int64)    # [B, T + 1]
    return blk[:, :T], blk[:, offset:offset + T]                               # [B, T]、[B, T]
# --8<-- [end:batch]


# --8<-- [start:evaluate]
@torch.no_grad()
def evaluate(model, stream: np.ndarray, n_bytes: int, eos_id: int, cfg: dict, device, max_windows: int | None = None,
             T: int | None = None) -> dict:
    """不重叠的定长窗口过一遍。分隔符 EOS 作为预测目标时不计（它不对应原文字节）。
    返回 loss（每 token 平均 NLL，nats）、bpb（总 NLL / ln2 / 原文字节数）。T 不写时用训练的 seq_len。"""
    model.eval()
    T = T or cfg["seq_len"]
    B = max(1, cfg["batch_size"] * cfg["seq_len"] // T)
    off = cfg.get("target_offset", 1)
    n_win = max(1, (len(stream) - 1 + T - 1) // T)
    if max_windows:
        n_win = min(n_win, max_windows)
    total_nll, total_tok = 0.0, 0
    for w0 in range(0, n_win, B):
        xs, ys = [], []
        for w in range(w0, min(w0 + B, n_win)):
            seg = stream[w * T: w * T + T + 1]
            x = np.full(T, eos_id, dtype=np.int64)
            y = np.full(T, -100, dtype=np.int64)
            n = len(seg) - 1
            x[:n], y[:n] = seg[:-1], seg[off:off + n]
            y[y == eos_id] = -100
            xs.append(x)
            ys.append(y)
        x = torch.from_numpy(np.stack(xs)).to(device)                      # [b, T]
        y = torch.from_numpy(np.stack(ys)).to(device)                      # [b, T]
        with autocast(device, cfg):
            logits = model(x)                                              # [b, T, V]
        nll = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction="sum")
        total_nll += float(nll)
        total_tok += int((y != -100).sum())
    model.train()
    if total_tok == 0:  # 空子集（例如去重后验证集里没有泄漏文档）
        return {"loss": None, "bpb": None, "tokens": 0, "bytes": 0}
    frac = n_win / max(1, (len(stream) - 1 + T - 1) // T)                  # 只评估了一部分窗口时按比例折算字节
    loss = total_nll / max(total_tok, 1)
    return {"loss": round(loss, 5), "bpb": round(total_nll / math.log(2) / max(n_bytes * frac, 1), 5),
            "tokens": total_tok, "bytes": int(n_bytes * frac)}
# --8<-- [end:evaluate]


@torch.no_grad()
def evaluate_docs(model, docs_ids: list[list[int]], n_bytes: int, eos_id: int, cfg: dict, device) -> dict:
    """每篇文档单独评估：[EOS, 文档]，从位置 0 开始、前面没有别的文档（打标推理时的真实形态）。超过 seq_len 的截断。"""
    model.eval()
    T = cfg["seq_len"]
    total_nll, total_tok = 0.0, 0
    for ids in docs_ids:
        seq = [eos_id] + list(ids[:T])
        x = torch.tensor([seq[:-1]], device=device)                        # [1, n]
        y = torch.tensor([seq[1:]], device=device)
        with autocast(device, cfg):
            logits = model(x)
        total_nll += float(F.cross_entropy(logits.float()[0], y[0], reduction="sum"))
        total_tok += y.numel()
    model.train()
    return {"loss": round(total_nll / max(total_tok, 1), 5), "bpb": round(total_nll / math.log(2) / max(n_bytes, 1), 5),
            "tokens": total_tok, "docs": len(docs_ids)}


# --8<-- [start:budget]
def plan_steps(n_blocks: int, cfg: dict, budget_steps: int | None = None) -> dict:
    """训练多少步、用哪些块：只用前 data_frac 的块（训练流的前缀），过 epochs 遍，每遍重新打乱；
    给了算力预算时步数 = 预算 ÷ 每步 FLOPs（不超过数据能提供的步数）。"""
    B = cfg["batch_size"]
    n_used = max(1, int(n_blocks * cfg.get("data_frac", 1.0)))
    epochs = int(cfg.get("epochs", 1))
    order = np.concatenate([np.random.default_rng(cfg["seed"] + 1000 * e).permutation(n_used) for e in range(epochs)])
    total = len(order) // B
    capped = False
    if budget_steps:
        capped = budget_steps > total
        total = min(total, budget_steps)
    if cfg["max_steps"]:
        total = min(total, cfg["max_steps"])
    if total < 1:
        raise ValueError(f"训练集只有 {n_used} 个块，不够一步（batch_size={B}）")
    return {"order": order, "total": total, "n_used": n_used, "epochs": epochs, "budget_capped": capped}
# --8<-- [end:budget]


# --8<-- [start:train]
def train(ctx, cfg: dict, model, train_stream: np.ndarray, train_bytes: int, valid: dict, eos_id: int, device,
          budget_steps: int | None = None) -> dict:
    """token 流切成 [seq_len + 1] 的块，打乱块顺序，每步取 batch_size 块（梯度累积时拆成 grad_accum 份）。"""
    T, B = cfg["seq_len"], cfg["batch_size"]
    accum = int(cfg.get("grad_accum", 1))
    if B % accum:
        raise ValueError(f"batch_size={B} 不能被 grad_accum={accum} 整除")
    mb = B // accum
    n_blocks = (len(train_stream) - 1) // T
    plan = plan_steps(n_blocks, cfg, budget_steps)
    total, starts = plan["total"], plan["order"] * T
    off = cfg.get("target_offset", 1)
    bytes_per_token = train_bytes / max(len(train_stream), 1)
    opt = make_optimizer(model, cfg)
    use_scaler = half_dtype(device, cfg) == torch.float16 and cfg.get("grad_scaler", False)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler) if use_scaler else None
    aux_coef = cfg.get("moe_aux_coef", 0.01)
    eval_at = sorted({max(1, round(total * (i + 1) / cfg["eval_points"])) for i in range(cfg["eval_points"])} | {total})
    log_every = max(1, total // 200)
    probe_at = {0, total // 10, total // 2}                                 # 在这几步统计 logits 梯度的下溢比例
    warm_end = max(1, int(total * cfg["warmup_frac"]))
    ctx.log(f"训练：{total} 步 × {B} × {T} = {total * B * T:,} token（训练流共 {len(train_stream):,} token，用前 "
            f"{plan['n_used'] * T:,} 个 × {plan['epochs']} 遍；每 token {bytes_per_token:.2f} 字节）"
            f"{f'；梯度累积 {accum} × {mb}' if accum > 1 else ''}；在第 {eval_at} 步评估")
    x0, y0 = make_batch(train_stream, starts[:B], T, off)
    batch_check = {"y_eq_x_frac": round(float((x0 == y0).mean()), 4), "x_head": x0[0, :8].tolist(),
                   "y_head": y0[0, :8].tolist()}
    model.train()
    t0 = time.time()
    t_last, tok_since = t0, 0
    curve, tps_log, losses, gnorms, dlogit_zero, moe_tail = [], [], [], [], [], []
    gmax = torch.zeros((), device=device)
    nan_step, loss_warm, skipped, step = None, None, 0, -1
    loss = torch.zeros(())
    for step in range(total):
        lr = lr_at(step, total, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = starts[step * B:(step + 1) * B]
        loss_sum = 0.0
        for k in range(accum):
            x, y = make_batch(train_stream, idx[k * mb:(k + 1) * mb], T, off)
            x = torch.from_numpy(x).to(device, non_blocking=True)                   # [mb, T]
            y = torch.from_numpy(y).to(device, non_blocking=True)                   # [mb, T]
            with autocast(device, cfg):
                logits = model(x)                                                   # [mb, T, V]
            if step in probe_at and k == 0 and logits.requires_grad:
                logits.register_hook(lambda g: dlogit_zero.append(float((g == 0).float().mean())))
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
            aux = model.aux_loss() if hasattr(model, "aux_loss") else None
            total_loss = (loss + aux_coef * aux if aux is not None else loss) / accum
            (scaler.scale(total_loss) if scaler else total_loss).backward()
            loss_sum = loss_sum + loss.detach() / accum
        loss = loss_sum
        if scaler:
            scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"] if cfg["grad_clip"] > 0 else 1e9)
        gmax = torch.maximum(gmax, torch.nan_to_num(gn.detach().float(), nan=0.0, posinf=1e9))
        if scaler:
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        tok_since += B * T
        logged = (step + 1) % log_every == 0 or step + 1 == total or step + 1 == warm_end
        if logged:
            lv = float(loss)
            if not math.isfinite(lv):
                nan_step = step
                ctx.warn("nan_loss", f"第 {step} 步 loss = {lv}，停止训练")
                break
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            tps = tok_since / max(now - t_last, 1e-9)
            tps_log.append(tps)
            losses.append(lv)
            gnorms.append(float(gn))
            if step + 1 == warm_end:
                loss_warm = lv
            extra = {}
            ms = model.moe_stats() if hasattr(model, "moe_stats") else None
            if ms:
                extra["moe_max_load"] = round(ms["max_over_mean"], 4)
                if step >= 0.95 * total:
                    moe_tail.append(ms)
            if scaler:
                extra["scaler_scale"] = scaler.get_scale()
            ctx.metric("train_loss", lv, step=step + 1, lr=lr, grad_norm=float(gn), tokens_per_s=round(tps),
                       bytes_per_s=round(tps * bytes_per_token), **extra)
            ctx.metric("grad_norm", round(float(gn), 4), step=step + 1)
            for k, v in extra.items():
                ctx.metric(k, v, step=step + 1)
            ctx.check()
        if step + 1 in eval_at:
            ev = evaluate(model, valid["stream"], valid["bytes"], eos_id, cfg, device)
            ctx.metric("valid_loss", ev["loss"], step=step + 1, split="valid")
            ctx.metric("valid_bpb", ev["bpb"], step=step + 1, split="valid")
            curve.append({"step": step + 1, **ev, "train_tokens": (step + 1) * B * T})   # ev 里的 tokens 是验证集 token 数
            ctx.log(f"step {step + 1}/{total} train_loss {float(loss):.4f} valid loss {ev['loss']:.4f} bpb {ev['bpb']:.4f}")
        if logged or step + 1 in eval_at:  # 评估耗时不算进吞吐
            t_last, tok_since = time.time(), 0
    seconds = time.time() - t0
    done = step + 1 if nan_step is None else nan_step
    tps_med = float(np.median(tps_log)) if tps_log else done * B * T / max(seconds, 1e-9)
    tail = losses[-max(1, len(losses) // 20):]
    spikes = sum(1 for i in range(10, len(losses)) if losses[i] > 1.2 * float(np.median(losses[i - 10:i])))
    stats = {"loss_at_warmup_end": round(loss_warm, 5) if loss_warm is not None else None,
             "train_loss_tail": round(float(np.mean(tail)), 5) if tail else None, "loss_spikes": spikes,
             "grad_norm_max": round(float(gmax), 4), "grad_norm_median": round(float(np.median(gnorms)), 5) if gnorms else None,
             "nan_step": nan_step, "steps_done": done, "micro_batch": mb, "grad_accum": accum,
             "epochs": plan["epochs"], "unique_tokens": plan["n_used"] * T, "budget_capped": plan["budget_capped"],
             "batch_check": batch_check}
    precision = {"dtype": str(half_dtype(device, cfg) or "float32").replace("torch.", ""),
                 "dlogits_zero_frac": round(float(np.mean(dlogit_zero)), 5) if dlogit_zero else None,
                 "grad_scaler": bool(scaler), "scaler_scale_final": scaler.get_scale() if scaler else None,
                 "skipped_steps": skipped if scaler else None}
    moe = None
    if moe_tail:
        load = np.mean([np.mean(m["load"], axis=0) for m in moe_tail], axis=0)        # [E]，最后 5% 步、各层平均
        E = len(load)
        moe = {"load_frac": [round(float(v), 4) for v in load],
               "load_max_over_mean": round(float(np.mean([m["max_over_mean"] for m in moe_tail])), 4),
               "dead_experts": int(np.mean([sum(v < 0.1 / E for v in np.mean(m["load"], axis=0)) for m in moe_tail]) + 0.5),
               "router_entropy_norm": round(float(np.mean([m["entropy"] for m in moe_tail])), 4)}
    return {"steps": total, "seconds": round(seconds, 1), "train_tokens": done * B * T,
            "train_bytes_seen": round(done * B * T * bytes_per_token), "bytes_per_token": round(bytes_per_token, 4),
            "tokens_per_s": round(tps_med), "bytes_per_s": round(tps_med * bytes_per_token),
            "final_train_loss": round(float(loss), 5), "valid_curve": curve, "stats": stats, "precision": precision,
            "moe": moe}
# --8<-- [end:train]
