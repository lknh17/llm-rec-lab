"""训练与评估：token 流 → 定长块 → next-token 交叉熵 → AdamW；评估同时报每 token loss 和 bits-per-byte。

为什么要 bits-per-byte（bpb）：每 token loss 的分母是 token 数。词表越大，一个 token 装的字节越多，
每 token loss 天然越高；字节级分词器一个 token 只有 1 个字节，每 token loss 天然很低。
bpb = 总 NLL / ln2 / 原文字节数，分母换成"同一批文本的字节数"，不同分词器之间才可比。
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


def lr_at(step: int, total: int, cfg: dict) -> float:
    warm = max(1, int(total * cfg["warmup_frac"]))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def autocast(device, cfg: dict):
    use_bf16 = cfg["precision"] == "bf16" or (cfg["precision"] == "auto" and device.type == "cuda")
    if use_bf16:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def make_optimizer(model, cfg: dict):
    """权重衰减只给矩阵权重；norm 的缩放和 embedding 不衰减（embedding 被输出层共享，衰减它会压低 logits 尺度）。"""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 and "wte" not in n else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8,
                             fused=torch.cuda.is_available())


# --8<-- [start:evaluate]
@torch.no_grad()
def evaluate(model, stream: np.ndarray, n_bytes: int, eos_id: int, cfg: dict, device, max_windows: int | None = None) -> dict:
    """不重叠的定长窗口过一遍。分隔符 EOS 作为预测目标时不计（它不对应原文字节）。
    返回 loss（每 token 平均 NLL，nats）、bpb（总 NLL / ln2 / 原文字节数）。"""
    model.eval()
    T, B = cfg["seq_len"], max(1, cfg["batch_size"])
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
            x[:len(seg) - 1], y[:len(seg) - 1] = seg[:-1], seg[1:]
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


# --8<-- [start:train]
def train(ctx, cfg: dict, model, train_stream: np.ndarray, train_bytes: int, valid: dict, eos_id: int, device) -> dict:
    """一遍训练集：token 流切成 [seq_len + 1] 的块，打乱块顺序，每步取 batch_size 块。"""
    T, B = cfg["seq_len"], cfg["batch_size"]
    n_blocks = (len(train_stream) - 1) // T
    total = n_blocks // B
    if cfg["max_steps"]:
        total = min(total, cfg["max_steps"])
    if total < 1:
        raise ValueError(f"训练集只有 {n_blocks} 个块，不够一步（batch_size={B}）")
    order = np.random.default_rng(cfg["seed"]).permutation(n_blocks)
    starts = order * T
    bytes_per_token = train_bytes / max(len(train_stream), 1)
    opt = make_optimizer(model, cfg)
    eval_at = sorted({max(1, round(total * (i + 1) / cfg["eval_points"])) for i in range(cfg["eval_points"])} | {total})
    log_every = max(1, total // 200)
    ctx.log(f"训练：{total} 步 × {B} × {T} = {total * B * T:,} token（训练集共 {len(train_stream):,} token，"
            f"每 token {bytes_per_token:.2f} 字节）；在第 {eval_at} 步评估")
    model.train()
    t0 = time.time()
    t_last, tok_since = t0, 0
    curve, tps_log = [], []
    for step in range(total):
        idx = starts[step * B:(step + 1) * B]
        blk = np.stack([train_stream[s:s + T + 1] for s in idx]).astype(np.int64)   # [B, T + 1]
        blk = torch.from_numpy(blk).to(device, non_blocking=True)
        x, y = blk[:, :-1], blk[:, 1:]                                              # [B, T]
        lr = lr_at(step, total, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        with autocast(device, cfg):
            logits = model(x)                                                       # [B, T, V]
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        if not torch.isfinite(loss):
            ctx.warn("nan_loss", f"第 {step} 步 loss = {float(loss)}，停止训练")
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"] if cfg["grad_clip"] > 0 else 1e9)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tok_since += B * T
        if (step + 1) % log_every == 0 or step + 1 == total:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            tps = tok_since / max(now - t_last, 1e-9)
            tps_log.append(tps)
            ctx.metric("train_loss", float(loss), step=step + 1, lr=lr, grad_norm=float(gn), tokens_per_s=round(tps),
                       bytes_per_s=round(tps * bytes_per_token))
            ctx.check()
        if step + 1 in eval_at:
            ev = evaluate(model, valid["stream"], valid["bytes"], eos_id, cfg, device)
            ctx.metric("valid_loss", ev["loss"], step=step + 1, split="valid")
            ctx.metric("valid_bpb", ev["bpb"], step=step + 1, split="valid")
            curve.append({"step": step + 1, "tokens": (step + 1) * B * T, **ev})
            ctx.log(f"step {step + 1}/{total} train_loss {float(loss):.4f} valid loss {ev['loss']:.4f} bpb {ev['bpb']:.4f}")
        if (step + 1) % log_every == 0 or step + 1 in eval_at:  # 评估耗时不算进吞吐
            t_last, tok_since = time.time(), 0
    seconds = time.time() - t0
    tps_med = float(np.median(tps_log)) if tps_log else total * B * T / max(seconds, 1e-9)
    return {"steps": total, "seconds": round(seconds, 1), "train_tokens": total * B * T,
            "train_bytes_seen": round(total * B * T * bytes_per_token), "bytes_per_token": round(bytes_per_token, 4),
            "tokens_per_s": round(tps_med), "bytes_per_s": round(tps_med * bytes_per_token),
            "final_train_loss": round(float(loss), 5), "valid_curve": curve}
# --8<-- [end:train]
