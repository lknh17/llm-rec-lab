"""训练与评估：全量 softmax 交叉熵；评估对全部物品排序算 Recall@10 / NDCG@10，另报"目标 + 100 个随机负例"的采样指标作对照。"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import SeqData, eval_batch, to_device, train_batch

LOG_EVERY = 50


def autocast(device, cfg):
    on = cfg["precision"] == "bf16" or (cfg["precision"] == "auto" and device.type == "cuda")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=on and device.type in ("cuda", "cpu"))


def lr_at(step: int, total: int, cfg: dict) -> float:
    warm = max(1, int(total * cfg["warmup_frac"]))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def loss_on(model, b: dict, cfg: dict, rng: np.random.Generator) -> tuple[torch.Tensor, int]:
    """每个有下一个物品的位置都监督（四个模型相同）：全量 softmax 交叉熵。"""
    y = b["y"]
    logits = model(b)                                                       # [B, L, V]
    return F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=0), int((y != 0).sum())


# --8<-- [start:evaluate]
@torch.no_grad()
def scores(model, d: SeqData, users, cuts, cfg: dict, device, L: int) -> tuple[np.ndarray, np.ndarray, list]:
    """每个评估样本：目标在全部物品里的名次（0 = 第 1 名）和最后一个位置的 logits 前 k 个。返回 (rank, targets, kept)。"""
    model.eval()
    eb = max(1, min(512, 65536 // L))
    ranks, tgts, kept = [], [], []
    for s in range(0, len(users), eb):
        b = eval_batch(d, users[s:s + eb], cuts[s:s + eb], L, cfg["truncate"])
        bt = to_device(b, device)
        with autocast(device, cfg):
            logits = model(bt)[:, -1].float()                                # [B, V] 最后一个位置
        logits[:, 0] = float("-inf")
        tgt = bt["target"]
        t_score = logits.gather(1, tgt[:, None])
        ranks.append((logits > t_score).sum(1).cpu().numpy())
        tgts.append(b["target"]); kept += b["kept"]
    model.train()
    return np.concatenate(ranks), np.concatenate(tgts), kept


def recall_ndcg(rank: np.ndarray, k: int = 10) -> dict:
    hit = rank < k
    return {"recall10": round(float(hit.mean()), 5) if len(rank) else None,
            "ndcg10": round(float((hit / np.log2(rank + 2)).mean()), 5) if len(rank) else None, "n": int(len(rank))}


@torch.no_grad()
def sampled_rank(model, d: SeqData, users, cuts, cfg: dict, device, L: int, n_neg: int, seed: int = 12345) -> np.ndarray:
    """目标 + n_neg 个随机负例（从用户没交互过的物品里均匀抽）中的名次。负例和模型无关，每个场景相同。"""
    rng = np.random.default_rng(seed)
    model.eval()
    eb = max(1, min(512, 65536 // L))
    out = []
    for s in range(0, len(users), eb):
        uu, cc = users[s:s + eb], cuts[s:s + eb]
        b = eval_batch(d, uu, cc, L, cfg["truncate"])
        negs = []
        for u in uu:
            seen = set(d.seq(int(u))[0].tolist())
            cand = rng.integers(1, d.n_items, size=n_neg * 3)
            cand = [c for c in cand if c not in seen][:n_neg]
            while len(cand) < n_neg:
                c = int(rng.integers(1, d.n_items))
                if c not in seen:
                    cand.append(c)
            negs.append(cand)
        bt = to_device(b, device)
        with autocast(device, cfg):
            logits = model(bt)[:, -1].float()
        neg = torch.tensor(np.array(negs), device=device)
        t_score = logits.gather(1, bt["target"][:, None])
        out.append((logits.gather(1, neg) > t_score).sum(1).cpu().numpy())
    model.train()
    return np.concatenate(out)
# --8<-- [end:evaluate]


# --8<-- [start:train]
def train(ctx, cfg: dict, model, d: SeqData, device, L: int, B: int) -> dict:
    rng = np.random.default_rng(cfg["seed"] + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    vu, vc = d.evals["valid"]
    sub = np.random.default_rng(7).permutation(len(vu))[:max(1, int(len(vu) * cfg["valid_frac"]))]
    best = {"recall10": -1.0, "step": 0, "state": None}
    total = cfg["steps"]
    t0 = time.time()
    tok_sum, win, t_last, tok_last, tail = 0, [], time.time(), 0, []
    for step in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, cfg)
        b = to_device(train_batch(d, rng, B, L, cfg["truncate"]), device)
        with autocast(device, cfg):
            loss, ntok = loss_on(model, b, cfg, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        win.append(loss.item())
        tail = (tail + [win[-1]])[-LOG_EVERY:]
        tok_sum += int((b["x"] != 0).sum())
        if (step + 1) % LOG_EVERY == 0 or step + 1 == total:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t_last
            ctx.metric("train_loss", float(np.mean(win)), step=step + 1, lr=lr_at(step, total, cfg),
                       tokens_per_s=round((tok_sum - tok_last) / max(dt, 1e-9)))
            win, t_last, tok_last = [], time.time(), tok_sum
            ctx.check()
        if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == total:
            r, _, _ = scores(model, d, vu[sub], vc[sub], cfg, device, L)
            rec = recall_ndcg(r)["recall10"]
            ctx.metric("valid_recall10", rec, step=step + 1, split="valid")
            ctx.log(f"step {step + 1}/{total}  loss {loss.item():.4f}  验证 Recall@10 {rec:.4f}")
            if rec > best["recall10"]:
                best = {"recall10": rec, "step": step + 1, "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
    secs = time.time() - t0
    model.load_state_dict(best["state"])
    return {"steps": total, "seconds": round(secs, 2), "best_step": best["step"], "best_valid_recall10": best["recall10"],
            "samples_per_s": round(total * B / secs, 1), "tokens_per_s": round(tok_sum / secs, 1),
            "train_tokens": int(tok_sum), "final_loss": round(float(np.mean(tail)), 5), "batch_size": B}
# --8<-- [end:train]
