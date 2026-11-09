"""k-means 与残差量化（RQ-KMeans）、去重位、语义 ID 的统计指标。全部在 torch 上做，距离分块计算（千万级物品时不会爆显存）。"""

from __future__ import annotations

import math

import numpy as np
import torch

CHUNK = 262_144


def _nearest(x: torch.Tensor, c: torch.Tensor, chunk: int = CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """每个点最近的中心和距离平方。‖x‖² − 2xcᵀ + ‖c‖²，按 chunk 行分块。"""
    cn = (c * c).sum(1)
    idx, dmin = [], []
    for s in range(0, len(x), chunk):
        xb = x[s:s + chunk]
        d = (xb * xb).sum(1, keepdim=True) - 2 * xb @ c.T + cn[None]
        m = d.min(1)
        idx.append(m.indices); dmin.append(m.values.clamp(min=0))
    return torch.cat(idx), torch.cat(dmin)


# --8<-- [start:kmeans]
def kmeans(x: torch.Tensor, k: int, iters: int, seed: int, on_iter=None) -> tuple[torch.Tensor, torch.Tensor]:
    """x [N, D] → (中心 [K, D], 分配 [N])。k-means++ 初始化（在至多 20 万个样本上）+ Lloyd 迭代；空簇用最远点重置。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(x)
    samp = x[torch.randperm(n, generator=g)[: min(n, 200_000)].to(x.device)]
    c = samp[torch.randint(len(samp), (1,), generator=g).to(x.device)]
    d2 = ((samp - c) ** 2).sum(1)
    for _ in range(1, k):                                                   # k-means++：按到最近中心的距离平方采样
        nxt = samp[torch.multinomial((d2 + 1e-12).cpu(), 1, generator=g).to(x.device)]
        c = torch.cat([c, nxt])
        d2 = torch.minimum(d2, ((samp - nxt) ** 2).sum(1))
    for it in range(iters):                                                 # Lloyd：分配 → 更新中心
        a, dist = _nearest(x, c)
        cnt = torch.bincount(a, minlength=k)
        new = torch.zeros_like(c).index_add_(0, a, x) / cnt.clamp(min=1)[:, None]
        empty = cnt == 0
        if empty.any():                                                     # 空簇：用离中心最远的点重置（码利用率保持 100%）
            new[empty] = x[dist.topk(int(empty.sum())).indices]
        c = new
        if on_iter:
            on_iter(it, float(dist.mean()))
    a, _ = _nearest(x, c)
    return c, a
# --8<-- [end:kmeans]


# --8<-- [start:rq]
def rq_kmeans(x: torch.Tensor, levels: int, k: int, iters: int, seed: int, residual: bool = True,
              fit_sample: int | None = None, on_iter=None) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
    """逐层：对"上一层没解释掉的部分"做 k-means，码 c_l，残差 r_l = r_{l−1} − C_l[c_l]。
    residual=False 是教学故障：每层都对原始 x 聚类（只换了种子），层与层之间重复描述同一件事。
    返回 (codes [N, L], 码本列表, 每层之后的相对重建误差 ‖x − x̂‖ / ‖x‖)。"""
    g = torch.Generator(device="cpu").manual_seed(seed + 99)
    n = len(x)
    fit_idx = torch.randperm(n, generator=g)[: min(n, fit_sample or n)].to(x.device)
    r = x.clone()
    recon = torch.zeros_like(x)
    norm = x.norm(dim=1).clamp(min=1e-8)
    codes, books, errs = [], [], []
    for lv in range(levels):
        target = r if residual else x
        c, _ = kmeans(target[fit_idx], k, iters, seed + lv,
                      on_iter=(lambda it, e, lv=lv: on_iter(lv * iters + it, e)) if on_iter else None)
        a, _ = _nearest(target, c)                                          # 全部物品分块分配
        recon = recon + c[a]
        r = x - recon
        codes.append(a); books.append(c)
        errs.append(float((r.norm(dim=1) / norm).mean()))
    return torch.stack(codes, 1), books, errs
# --8<-- [end:rq]


# --8<-- [start:collide]
def pack(codes: torch.Tensor, k: int) -> torch.Tensor:
    """码元组 → 一个 int64（K^L < 2^63 时）。"""
    key = torch.zeros(len(codes), dtype=torch.long, device=codes.device)
    for j in range(codes.shape[1]):
        key = key * k + codes[:, j].long()
    return key


def add_dedup(codes: torch.Tensor, k: int) -> torch.Tensor:
    """码完全相同的物品依次编号 0, 1, 2…（排序 + 分组，向量化），追加成最后一位：[N, L] → [N, L+1]。"""
    key = pack(codes, k)
    order = torch.argsort(key, stable=True)
    ks = key[order]
    first = torch.ones(len(ks), dtype=torch.bool, device=ks.device)
    first[1:] = ks[1:] != ks[:-1]
    grp_start = torch.cummax(torch.where(first, torch.arange(len(ks), device=ks.device), 0), 0).values
    dd = torch.empty_like(key)
    dd[order] = torch.arange(len(ks), device=ks.device) - grp_start
    return torch.cat([codes, dd[:, None]], 1)
# --8<-- [end:collide]


# --8<-- [start:metrics]
def collision_stats(codes: torch.Tensor, k: int) -> dict:
    """碰撞率 = 和别的物品码完全相同的物品比例；均匀基准 = 1 − (1 − K^−L)^(N−1)（N 个物品均匀撒进 K^L 个格子）。"""
    n, L = codes.shape
    _, inv, cnt = torch.unique(pack(codes, k), return_inverse=True, return_counts=True)
    coll = float((cnt[inv] > 1).float().mean())
    cap = k ** L
    return {"collision_rate": round(coll, 5), "combos": int(len(cnt)), "capacity": int(cap), "max_bucket": int(cnt.max()),
            "uniform": round(1 - math.exp((n - 1) * math.log1p(-1.0 / cap)), 5) if cap > 1 else 1.0}


def level_stats(codes: torch.Tensor, k: int) -> dict:
    """每层：用到的码字比例、困惑度 exp(熵)、困惑度 / K。利用率 100% 不代表均衡：困惑度才看分布有多均匀。"""
    out = {}
    for j in range(codes.shape[1]):
        cnt = torch.bincount(codes[:, j], minlength=k).double()
        p = cnt / cnt.sum()
        ppl = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
        out[f"L{j + 1}"] = {"used_frac": round(float((cnt > 0).double().mean()), 4), "ppl": round(ppl, 2),
                            "ppl_frac": round(ppl / k, 4)}
    return out


def nmi(a: torch.Tensor, b: torch.Tensor) -> float:
    """两层码的归一化互信息：1 = 第二层只是第一层换了编号，0 = 互相独立。"""
    a, b = a.cpu().numpy(), b.cpu().numpy()
    n = len(a)
    ka, kb = a.max() + 1, b.max() + 1
    joint = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb) / n
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / (pa[:, None] * pb[None])[nz])).sum()
    ha, hb = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(), -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return round(float(mi / max(np.sqrt(ha * hb), 1e-12)), 4)


def prefix_sim(x: torch.Tensor, codes: torch.Tensor, k: int, pairs: int = 20000, seed: int = 0) -> dict:
    """共享前 d 位码的物品对，在原向量空间里的平均余弦；random 是随机物品对。"""
    g = np.random.default_rng(seed)
    xn = torch.nn.functional.normalize(x.float(), dim=1)
    out = {}
    n = len(x)
    i, j = g.integers(0, n, pairs), g.integers(0, n, pairs)
    out["random"] = round(float((xn[i] * xn[j]).sum(1).mean()), 4)
    for name, d in (("d1", 1), ("d2", 2), ("full", codes.shape[1])):
        key = pack(codes[:, :d], k).cpu().numpy()
        order = np.argsort(key, kind="stable")
        ks = key[order]
        same = np.flatnonzero(ks[1:] == ks[:-1])                            # 排序后相邻且同前缀的物品对
        if len(same) == 0:
            out[name] = None
            continue
        pick = g.choice(same, size=min(pairs, len(same)), replace=False)
        a, b = order[pick], order[pick + 1]
        out[name] = round(float((xn[a] * xn[b]).sum(1).mean()), 4)
    return out
# --8<-- [end:metrics]
