"""TIGER 的训练与评估：随机终点采样训练窗口；beam 解码出语义 ID → 物品 → Top-10。"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import TensorTrie, TigerEncDec, Tokenizer, beam_search


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def hist_batch(d, users, ends, max_hist: int) -> np.ndarray:
    """每个用户 seq[:end] 的最后 max_hist 个物品，左补齐 → [B, max_hist]。"""
    out = np.zeros((len(users), max_hist), np.int64)
    for i, (u, e) in enumerate(zip(users, ends)):
        h = d.seq(int(u))[0][max(0, e - max_hist):e]
        if len(h):
            out[i, max_hist - len(h):] = h
    return out


class Decoder:
    """把 beam 的 token 元组翻译成物品：有去重位时一对一；没有去重位时一个桶里按热度排开。"""

    def __init__(self, tok: Tokenizer, pop: np.ndarray):
        self.tok = tok
        self.map: dict[tuple, list[int]] = {}
        for item, row in enumerate(tok.item_tokens[1:].tolist(), start=1):
            self.map.setdefault(tuple(row), []).append(item)
        for k in self.map:
            self.map[k].sort(key=lambda it: -pop[it - 1])
        self.level_of = tok.level_of

    def topk(self, seqs: torch.Tensor, scores: torch.Tensor, k: int = 10) -> tuple[list[list[int]], dict]:
        """seqs [B, W, D] → 每个用户最多 k 个不重复物品；同时统计非法元组、错层 token、不重复物品数。"""
        seqs, scores = seqs.cpu(), scores.cpu()
        B, W, D = seqs.shape
        lv = self.level_of[seqs]                                            # [B, W, D]
        wrong = (lv != torch.arange(D)[None, None]).any(-1)
        out, invalid, uniq = [], 0, 0
        for b in range(B):
            items, seen_beams = [], set()
            for w in torch.argsort(scores[b], descending=True).tolist():
                if not math.isfinite(float(scores[b, w])):
                    invalid += 1
                    continue
                bucket = self.map.get(tuple(seqs[b, w].tolist()))
                if bucket is None:
                    invalid += 1
                    continue
                seen_beams.add(bucket[0])
                for it in bucket:
                    if it not in items and len(items) < k:
                        items.append(it)
            uniq += len(seen_beams)
            out.append(items)
        return out, {"invalid_frac": invalid / (B * W), "wrong_level_frac": float(wrong.float().mean()), "unique_per_user": uniq / B}


def rank_metrics(lists: list[list[int]], targets: np.ndarray) -> dict:
    hit = np.array([t in l[:10] for l, t in zip(lists, targets)])
    pos = np.array([l.index(t) if t in l[:10] else 99 for l, t in zip(lists, targets)])
    out = {"recall10": round(float(hit.mean()), 5), "ndcg10": round(float((hit / np.log2(pos + 2)).mean()), 5), "n": len(lists),
           "cands_per_user": round(float(np.mean([len(l) for l in lists])), 3)}
    out["recall_at"] = {f"k{k}": round(float(np.mean([t in l[:k] for l, t in zip(lists, targets)])), 5) for k in (1, 5, 10)}
    return out


# --8<-- [start:evaluate]
def evaluate(model, d, tok, trie, dec: Decoder, users, cuts, cfg, device, W: int, constrained: bool, init: str):
    model.eval()
    item_tok = tok.item_tokens.to(device)
    lists, stats = [], {"invalid_frac": 0.0, "wrong_level_frac": 0.0, "unique_per_user": 0.0}
    times = []
    for s in range(0, len(users), cfg["infer_batch"]):
        u, c = users[s:s + cfg["infer_batch"]], cuts[s:s + cfg["infer_batch"]]
        h = torch.from_numpy(hist_batch(d, u, c, cfg["max_hist"])).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast(device):
            seqs, sc = beam_search(model, item_tok[h], trie if constrained else None, W, init)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000 / len(u))
        l, st = dec.topk(seqs, sc)
        lists += l
        for k in stats:
            stats[k] += st[k] * len(u)
    targets = np.array([d.seq(int(u))[0][int(c)] for u, c in zip(users, cuts)])
    out = rank_metrics(lists, targets)
    out.update({k: round(v / len(users), 5) for k, v in stats.items()})
    out["ms_per_user"] = round(float(np.median(times[1:] or times)), 4)        # 第一批含 warmup，不计
    model.train()
    return out, lists, targets
# --8<-- [end:evaluate]


# --8<-- [start:train]
def train(ctx, cfg, model: TigerEncDec, d, tok, trie, dec, device) -> dict:
    rng = np.random.default_rng(cfg["seed"] + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    item_tok = tok.item_tokens.to(device)
    vu, vc = d.evals["valid"]
    sub = np.random.default_rng(7).permutation(len(vu))[:1000]
    total, warm = cfg["max_steps"], max(1, cfg["max_steps"] // 50)
    best = {"recall10": -1.0, "step": 0, "state": None}
    t0, win = time.time(), []
    for step in range(total):
        lr = cfg["lr"] * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
        for g in opt.param_groups:
            g["lr"] = lr
        users = d.train_users[rng.integers(0, len(d.train_users), size=cfg["batch_size"])]
        ends = np.array([int(rng.integers(1, d.n_train[u])) for u in users])     # 目标 = seq[end]，历史 = seq[:end]
        h = torch.from_numpy(hist_batch(d, users, ends, cfg["max_hist"])).to(device)
        tgt = torch.from_numpy(np.array([d.seq(int(u))[0][e] for u, e in zip(users, ends)], np.int64)).to(device)
        with autocast(device):
            logits = model(item_tok[h], item_tok[tgt])                      # [B, D, V]
            loss = F.cross_entropy(logits.float().flatten(0, 1), item_tok[tgt].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        win.append(loss.item())
        if (step + 1) % 100 == 0:
            ctx.metric("train_loss", float(np.mean(win)), step=step + 1, lr=lr)
            win = []
            ctx.check()
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == total:
            r, _, _ = evaluate(model, d, tok, trie, dec, vu[sub], vc[sub], cfg, device, 10, True, "one_root")
            ctx.metric("valid_recall10", r["recall10"], step=step + 1, split="valid")
            ctx.log(f"step {step + 1}/{total}  loss {loss.item():.4f}  验证 Recall@10（beam 10）{r['recall10']:.4f}")
            if r["recall10"] > best["recall10"]:
                best = {"recall10": r["recall10"], "step": step + 1,
                        "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
    model.load_state_dict(best["state"])
    return {"seconds": round(time.time() - t0, 2), "best_step": best["step"], "best_valid_recall10": best["recall10"],
            "steps": total, "params": sum(p.numel() for p in model.parameters())}
# --8<-- [end:train]
