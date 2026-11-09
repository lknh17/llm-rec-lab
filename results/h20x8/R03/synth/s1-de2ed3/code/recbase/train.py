"""手写训练循环：取 batch → forward 得 logit → BCE → backward → AdamW 更新。第一步完整追踪；每个 epoch 末评估一次。"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from labkit.runtime import tensor_info

from .model import RecModel, bce_loss

LOG_EVERY = 50


def to_tensors(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """全部样本一次放上设备（100 万行 × 十几个字段约 100 MB），按下标取 batch，省掉 DataLoader。"""
    out = {}
    for k, v in arrays.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.float() if v.dtype.kind == "f" else t.long()
        out[k] = out[k].to(device)
    return out


# --8<-- [start:optimizer]
def make_optimizer(model: RecModel, cfg: dict) -> torch.optim.Optimizer:
    """AdamW。embedding 表不做 weight decay：一个 batch 只碰到表里很少几行，decay 却每步作用在所有行上，
    长尾 ID 的向量会被一直往 0 拉。"""
    emb = [p for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()]
    ids = {id(p) for p in emb}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{"params": emb, "weight_decay": 0.0},
                              {"params": rest, "weight_decay": cfg["weight_decay"]}], lr=cfg["lr"])
# --8<-- [end:optimizer]


@torch.no_grad()
def predict(model: RecModel, X: dict[str, torch.Tensor], idx: np.ndarray, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    out = []
    dev = next(iter(X.values())).device
    for s in range(0, len(idx), batch_size):
        b = torch.as_tensor(idx[s: s + batch_size], device=dev)
        out.append(torch.sigmoid(model({k: v[b] for k, v in X.items()})).float().cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def trace_first_update(ctx, model: RecModel, opt, batch: dict[str, torch.Tensor], cfg: dict) -> float:
    """真实的第一个 batch：前向、loss、反向后 embedding 表有几行拿到了梯度、被追踪样本那一行更新前后的值。"""
    model.train()
    v = model.field_vectors(batch)                                          # [B, F, d]
    x = model.concat(v, batch)                                              # [B, in_dim]
    logits = model(batch)                                                   # [B]
    loss = bce_loss(logits, batch["label"])
    per = nn.functional.binary_cross_entropy_with_logits(logits, batch["label"], reduction="none")
    heads = {f: [round(t, 4) for t in model.emb[f](batch[f][:1])[0, :4].tolist()] for f in model.sparse}
    ctx.trace("batch", {"note": "训练第 1 个 batch：每个字段的张量形状（B = batch 大小）",
                        "batch_size": int(batch["label"].shape[0]),
                        "tensors": {k: tensor_info(t, 6) for k, t in batch.items()},
                        "field_vectors": f"[B, {model.n_fields}, {cfg['embedding_dim']}]",
                        "mlp_input": tensor_info(x, 8), "in_dim": model.in_dim,
                        "positives": int(batch["label"].sum().item())})
    ctx.trace("forward", {"field_embedding_head4": heads, "logit": round(logits[0].item(), 5),
                          "prob": round(torch.sigmoid(logits[0]).item(), 5), "label": int(batch["label"][0].item()),
                          "sample_loss": round(per[0].item(), 5), "batch_loss_mean": round(loss.item(), 5),
                          "formula": "loss_i = −[y·log σ(z) + (1−y)·log(1−σ(z))]，batch loss = 1/B Σ loss_i"})
    opt.zero_grad(set_to_none=True)
    loss.backward()
    upd = {"lr": cfg["lr"], "optimizer": "AdamW"}
    if "user_id" in model.emb:
        w = model.emb["user_id"].weight
        row = int(batch["user_id"][0])
        upd.update({"user_embedding_row": row, "row_before_head4": [round(t, 6) for t in w[row, :4].tolist()],
                    "row_grad_head4": [round(t, 6) for t in w.grad[row, :4].tolist()],
                    "user_rows_with_grad": int((w.grad.abs().sum(-1) > 0).sum()), "user_rows_total": int(w.shape[0]),
                    "oov_row_in_batch": int((batch["user_id"] == 0).sum())})
    opt.step()
    with torch.no_grad():
        after = model(batch)
        if "user_id" in model.emb:
            upd["row_after_head4"] = [round(t, 6) for t in model.emb["user_id"].weight[upd["user_embedding_row"], :4].tolist()]
        upd.update({"logit_before": round(logits[0].item(), 5), "logit_after": round(after[0].item(), 5),
                    "batch_loss_before": round(loss.item(), 5),
                    "batch_loss_after": round(bce_loss(after, batch["label"]).item(), 5),
                    "note": "只有本 batch 出现过的用户行有梯度；Adam 第一步每个元素的更新量约等于 lr × sign(grad)"})
    ctx.trace("update", upd)
    return loss.item()


# --8<-- [start:train_loop]
def train(ctx, cfg: dict, model: RecModel, X: dict[str, torch.Tensor], train_idx: np.ndarray,
          on_epoch: Callable[[int], dict]) -> dict:
    """训 cfg["epochs"] 遍；每遍重新打乱。每遍结束调用 on_epoch(第几遍) 评估，返回值记进 epochs 列表。"""
    opt = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(cfg["seed"])
    bs, dev = cfg["batch_size"], next(iter(X.values())).device
    steps_per_epoch = (len(train_idx) + bs - 1) // bs
    ctx.info("train", {"train_samples": int(len(train_idx)), "steps_per_epoch": steps_per_epoch,
                       "planned_steps": steps_per_epoch * cfg["epochs"]})
    step, epochs, t0 = 0, [], time.time()
    train_seconds = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        te = time.time()
        perm = train_idx[torch.randperm(len(train_idx), generator=gen).numpy()]
        window, ep_sum, ep_n, t_last = [], 0.0, 0, time.time()
        for s in range(0, len(perm), bs):
            idx = torch.as_tensor(perm[s: s + bs], device=dev)
            batch = {k: v[idx] for k, v in X.items()}                       # 每个字段 [B] 或 [B, 6]
            if step == 0:
                loss_val = trace_first_update(ctx, model, opt, batch, cfg)
            else:
                loss = bce_loss(model(batch), batch["label"])
                opt.zero_grad(set_to_none=True)
                loss.backward()                                             # 只有本 batch 出现的 embedding 行有梯度
                opt.step()
                loss_val = loss.item()
            step += 1
            window.append(loss_val)
            ep_sum += loss_val * len(idx)
            ep_n += len(idx)
            if step % LOG_EVERY == 0:
                dt = time.time() - t_last
                ctx.metric("train_loss", float(np.mean(window)), step=step, epoch=epoch,
                           samples_per_s=round(len(window) * bs / max(dt, 1e-9)))
                window, t_last = [], time.time()
                ctx.check()
        train_seconds += time.time() - te
        ev = {"epoch": epoch, "step": step, "train_logloss": round(ep_sum / max(ep_n, 1), 5), **on_epoch(epoch)}
        epochs.append(ev)
        ctx.log(f"epoch {epoch}: train logloss {ev['train_logloss']}  " +
                "  ".join(f"{k} {v}" for k, v in ev.items() if k not in ("epoch", "step", "train_logloss")))
    return {"epochs": cfg["epochs"], "steps": step, "seconds": round(time.time() - t0, 2),
            "train_seconds": round(train_seconds, 2),
            "samples_per_s": round(len(train_idx) * cfg["epochs"] / max(train_seconds, 1e-9)), "per_epoch": epochs}
# --8<-- [end:train_loop]
