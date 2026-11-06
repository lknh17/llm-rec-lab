"""参数更新：手写的 SFT 训练循环。没有 Trainer，数据整理、loss、优化器调用都在这一页里。"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .data import IGNORE, collate, sample_trace_payload, token_table


# --8<-- [start:loss]
def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """位置 t 的 logits 预测位置 t+1 的 token，所以 logits 去掉最后一位、labels 去掉第一位再对齐。

    归一化：被监督 token 的 loss 求和 / 被监督 token 总数（整个 batch 一起平均）。
    长回答的样本在 batch 里权重更大；另一种做法是先每条样本内平均再跨样本平均。
    分母用 clamp(min=1)：没有任何监督 token 时 loss = 0 而不是 nan——训练不会报错，但也什么都没学。
    """
    shift_logits = logits[:, :-1, :]                     # [B, T-1, V]
    shift_labels = labels[:, 1:]                         # [B, T-1]
    n_sup = (shift_labels != IGNORE).sum()
    loss_sum = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)).float(), shift_labels.reshape(-1),
                               ignore_index=IGNORE, reduction="sum")
    return loss_sum / n_sup.clamp(min=1), n_sup


def forward_supervised(model, b: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """只在需要算 loss 的位置计算 lm_head。

    词表 V=151936：完整 logits [8, 320, V] 的 fp32 张量约 1.5 GB，而被监督的只有末尾约 40 个位置。
    start = batch 内最早的被监督位置 - 1（该位置的 logit 预测第一个被监督 token），之前的位置不算 logits。
    all_tokens 监督时 start=0，退化为完整计算。返回 (logits[B, T-start, V], labels[B, T-start], start)。
    """
    labels = b["labels"]                                         # [B, T]
    T = labels.shape[1]
    sup_pos = (labels != IGNORE).any(dim=0).nonzero()
    start = max(int(sup_pos[0]) - 1, 0) if len(sup_pos) else T - 1
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], logits_to_keep=T - start)
    return out.logits, labels[:, start:], start
# --8<-- [end:loss]


@torch.no_grad()
def eval_loss(model, examples: list[dict], pad_id: int, device, bs: int = 8) -> float:
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(examples), bs):
        b = collate(examples[s: s + bs], pad_id, device)
        logits, labels, _ = forward_supervised(model, b)
        loss, n_sup = causal_lm_loss(logits, labels)
        tot += loss.item() * n_sup.item()
        n += n_sup.item()
    model.train()
    return tot / max(n, 1)


def lora_b_norm(model) -> float:
    return float(sum(p.detach().float().norm().item() ** 2 for n, p in model.named_parameters() if "lora_B" in n) ** 0.5)


def base_weight_probe(model) -> float:
    """取一个基础权重（第 0 层 q_proj）的和作为指纹：LoRA 训练前后必须完全不变。"""
    for n, p in model.named_parameters():
        if "layers.0.self_attn.q_proj.base_layer.weight" in n:
            return float(p.detach().float().sum().item())
    return float("nan")


# --8<-- [start:train_loop]
def train(ctx, cfg, model, tok, examples: list[dict], dev_examples: list[dict], device, on_epoch_end=None) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]          # 只有 LoRA 的 A、B
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    bs = cfg["batch_size"]
    steps_per_epoch = math.ceil(len(examples) / bs)
    total = steps_per_epoch * cfg["epochs"]
    if cfg["max_steps"]:
        total = min(total, cfg["max_steps"])
    warmup = max(1, int(0.05 * total))
    # 学习率：前 5% 步线性升温，之后线性降到 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    gen = torch.Generator().manual_seed(cfg["seed"])
    ctx.info("train", {"examples": len(examples), "steps_per_epoch": steps_per_epoch, "planned_steps": total,
                       "warmup_steps": warmup})
    b_norm0, base0 = lora_b_norm(model), base_weight_probe(model)
    step, seen_tokens, seen_sup, t_start = 0, 0, 0, time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(examples), generator=gen).tolist()
        for s in range(0, len(order), bs):
            if step >= total:
                break
            batch_ex = [examples[i] for i in order[s: s + bs]]
            b = collate(batch_ex, tok.pad_token_id, device)                       # 每个 [B, T]
            logits, labels, start = forward_supervised(model, b)                  # [B, T-start, V]
            loss, n_sup = causal_lm_loss(logits, labels)
            if step == 0:
                trace_first_batch(ctx, tok, batch_ex, b, logits, labels, start, loss, cfg)
            loss.backward()                                                        # 梯度只流到 LoRA 参数
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)                # 梯度裁剪，防止个别 batch 冲坏
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            loss_val, n_sup_val = loss.item(), int(n_sup.item())
            n_tok = int(b["attention_mask"].sum().item())
            seen_tokens += n_tok
            seen_sup += n_sup_val
            ctx.metric("train_loss", loss_val, step=step, epoch=epoch + 1, supervised_tokens=n_sup_val,
                       tokens=n_tok, lr=sched.get_last_lr()[0], grad_norm=round(float(grad_norm), 4),
                       tokens_per_s=round(seen_tokens / max(time.time() - t_start, 1e-6)))
            del logits, loss
            if device.type == "mps" and step % 10 == 0:
                # 每步序列长度不同，MPS 缓存分配器会为每种形状留一份显存；定期释放，避免占用一路涨到几十 GB
                torch.mps.empty_cache()
            if step % 10 == 0 or step == 1:
                ctx.log(f"step {step}/{total} loss={loss_val:.4f} 监督token={n_sup_val} "
                        f"lr={sched.get_last_lr()[0]:.2e} grad_norm={float(grad_norm):.3f}")
            ctx.check()
        dl = eval_loss(model, dev_examples, tok.pad_token_id, device)
        ctx.metric("dev_loss", dl, step=step, split="dev", epoch=epoch + 1)
        ctx.log(f"epoch {epoch + 1}: dev_loss={dl:.4f}")
        if on_epoch_end:
            on_epoch_end(epoch + 1, step)
        if step >= total:
            break
    result = {"steps": step, "train_tokens": seen_tokens, "supervised_tokens": seen_sup,
              "seconds": round(time.time() - t_start, 1),
              "lora_B_norm_before": b_norm0, "lora_B_norm_after": lora_b_norm(model),
              "base_weight_probe_before": base0, "base_weight_probe_after": base_weight_probe(model)}
    return result
# --8<-- [end:train_loop]


def trace_first_batch(ctx, tok, batch_ex, b, logits, labels, start, loss, cfg) -> None:
    """logits 只覆盖位置 [start, T)：位置 p 的 logit 预测 token p+1。这里换算回整条序列的下标。"""
    from labkit.runtime import tensor_info

    ex = batch_ex[0]
    if "raw" in ex:
        ctx.trace("sample", sample_trace_payload(ex, cfg["prompt_mode"],
                                                 "训练第 1 个 batch 的第 1 条样本（打乱后的真实顺序）；下方 token、张量、loss 都是这一条"))
    with torch.no_grad():
        lg = logits[0, :-1].float()                    # 预测位置 start+1 .. T-1
        lab = labels[0, 1:]
        per_tok = F.cross_entropy(lg, lab.clamp(min=0), reduction="none")
        per_tok = torch.where(lab == IGNORE, torch.zeros_like(per_tok), per_tok).tolist()
        pred_next = lg.argmax(-1).tolist()
    # 对齐到 token_table 的约定：per_token_loss[i-1] 是 token i 的 loss
    full_loss = [0.0] * start + per_tok
    table = token_table(tok, ex, full_loss)
    for row in table:
        j = row["i"] - 1 - start
        if row["supervised"] and 0 <= j < len(pred_next):
            row["model_pred"] = tok.decode([pred_next[j]])
    sup_text = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE])
    ctx.trace("batch", {
        "input_ids": tensor_info(b["input_ids"], 6), "attention_mask": tensor_info(b["attention_mask"], 6),
        "labels": tensor_info(b["labels"], 6),
        "logits": {"shape": list(logits.shape), "dtype": str(logits.dtype), "start_position": start,
                   "full_shape_if_not_sliced": [*b["input_ids"].shape, logits.shape[-1]]},
        "per_sample_len": [len(e["input_ids"]) for e in batch_ex],
        "per_sample_supervised": [e["n_supervised"] for e in batch_ex],
        "pad_tokens": int((b["attention_mask"] == 0).sum().item()),
        "batch_loss": round(loss.item(), 5),
    })
    ctx.trace("train_tokens", {"note": "训练第 1 个 batch 的第 1 条样本，逐 token 展开（真实 tokenizer 输出）",
                               "n_prompt": ex["n_prompt"], "n_total": len(ex["input_ids"]),
                               "n_supervised": ex["n_supervised"], "truncated": ex["truncated"],
                               "supervised_text": sup_text, "tokens": table})
