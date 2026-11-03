"""参数更新：手写训练循环（不用 Trainer），每一步做了什么都能直接看到。

一步 = 取 batch -> forward 得 logit -> BCE loss -> backward 得梯度 -> AdamW 更新参数。
第一步会被完整追踪：样本、张量形状、logit、梯度、更新前后的参数和 logit。
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .evaluate import auc, logloss, predict
from .model import bce_loss

TENSOR_FIELDS = ["user_id", "movie_id", "gender", "age", "occupation", "zip1", "year_bucket", "genres",
                 "user_cnt_bucket", "item_cnt_bucket", "user_like_rate", "item_like_rate", "label"]


def to_tensors(data, device: torch.device) -> dict[str, torch.Tensor]:
    """全部样本一次性放到设备上（100 万行 x 13 个字段约 130 MB），按下标取 batch，省掉 DataLoader 开销。"""
    return {k: torch.as_tensor(data.arrays[k], device=device) for k in TENSOR_FIELDS}


def make_optimizer(model, cfg) -> torch.optim.Optimizer:
    # --8<-- [start:optimizer]
    # 可训练参数 = 全部 embedding 表 + MLP。AdamW 对每个参数维护两份状态（一阶、二阶动量），
    # 所以优化器状态占 2 倍参数量的显存
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # --8<-- [end:optimizer]


def trace_first_update(ctx, data, model, opt, batch, batch_idx_np, cfg) -> float:
    """对真实的第一个 batch 做一次完整追踪，然后才执行这一步更新。"""
    i = int(batch_idx_np[0])  # 本 batch 的第一条样本就是被追踪的样本
    raw = data.raw_row(i)
    encoded = {f: batch[f][0].tolist() for f in TENSOR_FIELDS if f != "label"}
    from labkit.runtime import tensor_info

    shapes = {f: tensor_info(batch[f], 6) for f in TENSOR_FIELDS}
    model.train()
    x = model.features(batch)                          # [B, in_dim]
    logits = model.mlp(x).squeeze(-1)                  # [B]
    loss = bce_loss(logits, batch["label"])
    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["label"], reduction="none")
    field_vectors = {f: [round(v, 4) for v in model.emb[f](batch[f][:1])[0, :4].tolist()] for f in model.sparse}
    opt.zero_grad(set_to_none=True)
    loss.backward()
    user_row = int(batch["user_id"][0])
    emb_u = model.emb["user_id"] if "user_id" in model.emb else None
    before = emb_u.weight[user_row, :4].detach().clone().tolist() if emb_u is not None else None
    grad_row = emb_u.weight.grad[user_row] if emb_u is not None else None
    rows_with_grad = int((emb_u.weight.grad.abs().sum(-1) > 0).sum()) if emb_u is not None else None
    grad_norms = {n: round(p.grad.norm().item(), 6) for n, p in model.named_parameters() if p.grad is not None}
    opt.step()
    with torch.no_grad():
        logits_after = model(batch)
        loss_after = bce_loss(logits_after, batch["label"]).item()
    after = emb_u.weight[user_row, :4].detach().tolist() if emb_u is not None else None
    ctx.trace("sample", {"raw": raw, "encoded": encoded,
                         "note": "被追踪样本 = 训练第 1 个 batch 的第 1 条（真实打乱结果，不是挑出来的）"})
    ctx.trace("batch", {"batch_size": int(batch["label"].shape[0]), "tensors": shapes,
                        "mlp_input": tensor_info(x, 8), "in_dim": model.in_dim,
                        "positives": int(batch["label"].sum().item()),
                        "supervised_samples": int(batch["label"].shape[0])})
    ctx.trace("forward", {
        "field_embedding_head": field_vectors,
        "logit": round(logits[0].item(), 5), "prob": round(torch.sigmoid(logits[0]).item(), 5),
        "label": int(batch["label"][0].item()), "sample_loss": round(per_sample[0].item(), 5),
        "batch_loss_mean": round(loss.item(), 5),
        "formula": "loss_i = -[y·log σ(z) + (1-y)·log(1-σ(z))]，batch loss = 1/B Σ loss_i",
    })
    ctx.trace("update", {
        "lr": cfg["lr"], "optimizer": "AdamW",
        "user_embedding_row": user_row,
        "row_before_head4": [round(v, 6) for v in before] if before else None,
        "row_grad_head4": [round(v, 6) for v in grad_row[:4].tolist()] if grad_row is not None else None,
        "row_after_head4": [round(v, 6) for v in after] if after else None,
        "user_rows_with_grad": rows_with_grad, "user_rows_total": emb_u.weight.shape[0] if emb_u is not None else None,
        "grad_norm_by_param": grad_norms,
        "logit_before": round(logits[0].item(), 5), "logit_after": round(logits_after[0].item(), 5),
        "batch_loss_before": round(loss.item(), 5), "batch_loss_after": round(loss_after, 5),
        "note": "Adam 第一步的更新量约等于 lr x sign(grad)，与梯度大小基本无关",
    })
    return loss.item()


# --8<-- [start:train_loop]
def train(ctx, cfg, data, model, tensors, device, resume: dict | None = None) -> dict:
    opt = make_optimizer(model, cfg)
    train_idx = data.index("train")
    valid_idx = data.index("valid")
    gen = torch.Generator().manual_seed(cfg["seed"])
    bs = cfg["batch_size"]
    steps_per_epoch = (len(train_idx) + bs - 1) // bs
    start_epoch, step = 0, 0
    if resume:  # 续训：恢复模型、优化器、轮次、步数和随机数状态，而不只是权重
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        start_epoch, step = resume["epoch"], resume["step"]
        gen.set_state(resume["rng"])
        ctx.log(f"续训：从 epoch={start_epoch} step={step} 恢复（含优化器状态与随机数状态）")
    max_steps = cfg["max_steps"] or 10**12
    ctx.info("train", {"train_samples": int(len(train_idx)), "steps_per_epoch": steps_per_epoch,
                       "planned_steps": min(steps_per_epoch * cfg["epochs"], max_steps),
                       "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    window, t_last, best = [], time.time(), None
    for epoch in range(start_epoch, cfg["epochs"]):
        perm = train_idx[torch.randperm(len(train_idx), generator=gen).numpy()]  # 每轮重新打乱
        for s in range(0, len(perm), bs):
            if step >= max_steps:
                break
            idx_np = perm[s: s + bs]
            idx = torch.as_tensor(idx_np, device=device)
            batch = {k: v[idx] for k, v in tensors.items()}       # 每个字段 [B] 或 [B, 6]
            if step == 0 and not resume:
                loss_val = trace_first_update(ctx, data, model, opt, batch, idx_np, cfg)
            else:
                logits = model(batch)                             # [B]
                loss = bce_loss(logits, batch["label"])           # 标量
                opt.zero_grad(set_to_none=True)
                loss.backward()                                   # 只有本 batch 出现的 embedding 行有梯度
                opt.step()
                loss_val = loss.item()
            step += 1
            if loss_val == loss_val:  # 不是 nan
                window.append(loss_val)
            if step % 50 == 0:
                dt = time.time() - t_last
                ctx.metric("train_loss", float(np.mean(window)), step=step, epoch=epoch,
                           samples_per_s=round(50 * bs / max(dt, 1e-9)))
                window, t_last = [], time.time()
                ctx.check()  # 取消和预算检查点
        # 每轮结束在验证集上评估：迭代选择只看验证集
        prob = predict(model, tensors, valid_idx)
        y = data.arrays["label"][valid_idx]
        v_auc, v_ll = auc(y, prob), logloss(y, prob)
        ctx.metric("valid_auc", v_auc, step=step, split="valid", epoch=epoch + 1)
        ctx.metric("valid_logloss", v_ll, step=step, split="valid", epoch=epoch + 1)
        ctx.log(f"epoch {epoch + 1}: step={step} valid AUC={v_auc:.4f} LogLoss={v_ll:.4f}")
        save_checkpoint(ctx, model, opt, epoch + 1, step, gen)
        best = {"epoch": epoch + 1, "valid_auc": v_auc} if not best or v_auc > best["valid_auc"] else best
        if step >= max_steps:
            ctx.log(f"达到 max_steps={cfg['max_steps']}，停止训练")
            break
    return {"steps": step, "best": best}
# --8<-- [end:train_loop]


def save_checkpoint(ctx, model, opt, epoch: int, step: int, gen: torch.Generator) -> None:
    # --8<-- [start:checkpoint]
    # 续训需要的不只是权重：优化器动量、轮次、步数、打乱顺序用的随机数状态缺一不可
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "step": step,
             "rng": gen.get_state()}
    torch.save(state, ctx.artifact("checkpoint_last.pt"))
    # --8<-- [end:checkpoint]
    ctx.register_artifact("checkpoint_last.pt", "checkpoint", f"epoch={epoch} step={step}，可续训")
