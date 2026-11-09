"""合成长序列（R02）：会话 + 簇偏好的背景行为，加上周期性的"回访串"作为长程信号。

一个用户的序列：
  背景：会话长度 ~ Geom(均值 synth_session_mean)；会话内簇以 synth_stay 的概率保持，否则从长期偏好 π_u 重抽；
        新会话的首个簇直接从 π_u 抽。簇内物品按 Zipf(1) 抽（簇内排名全局固定）。
  时间：会话内间隔 ~ Exp(120 s)，会话之间 ~ LogNormal(ln 86400, 1)。
  长程信号：每个用户有一个"回访簇" f_u（从长期偏好几乎为 0 的簇里挑，背景行为里基本见不到它）。
        从序列末尾往前，每隔约 P_u 写一段回访 [τ_u, s_1..s_b]：τ_u 是 8 个"回访入口"之一，s_j 是簇 f_u 里按 Zipf 抽的物品。
        长程题 = 回访入口之后的第一个物品：只看最近的行为不知道 f_u 是哪个簇，必须在窗口里找到更早的回访会话，
        认出它们属于哪个簇（汇总一组同簇物品的方向，不需要精确定位某一个位置）。
物品编号：0 = padding；1..8 = 回访入口 τ；9.. = 背景物品（簇 k 的第 r 名是 9 + k·m + r）。
"""

from __future__ import annotations

import numpy as np

N_TAU = 8
FIRST_BG = 1 + N_TAU
# 评估目标的三种类型：长程（回访串）、近期（会话中间）、新会话（会话起点）
TYPES = ("long", "recent", "newsess")
TYPE_SHARE = (0.4, 0.4, 0.2)


def scaled(cfg: dict) -> dict:
    """len_scale 等比缩放长度、周期（冒烟档用）。"""
    s = cfg["len_scale"]
    return {"len_min": max(32, int(cfg["synth_len_min"] * s)), "len_max": max(64, int(cfg["synth_len_max"] * s)),
            "p_min": max(8, int(cfg["synth_period_min"] * s)), "p_max": max(16, int(cfg["synth_period_max"] * s))}


def n_items(cfg: dict) -> int:
    """词表大小（含 padding 0）。"""
    return FIRST_BG + cfg["synth_clusters"] * cfg["synth_items_per_cluster"]


# --8<-- [start:generate]
def gen_user(rng: np.random.Generator, cfg: dict, sc: dict, zipf_p: np.ndarray) -> dict:
    K, m, b = cfg["synth_clusters"], cfg["synth_items_per_cluster"], cfg["synth_burst"]
    N = int(np.exp(rng.uniform(np.log(sc["len_min"]), np.log(sc["len_max"] + 1))))
    pi = rng.dirichlet(np.full(K, cfg["synth_alpha"]))                     # 长期偏好：集中在少数几个簇
    # 会话：几何分布的会话长度，标出每个会话的起点
    sess = rng.geometric(1.0 / cfg["synth_session_mean"], size=N)
    starts = np.cumsum(np.r_[0, sess])[:-1]
    is_start = np.zeros(N, bool)
    is_start[starts[starts < N]] = True
    # 簇：会话起点或以 1 − stay 的概率换簇；每段从 π_u 抽一个簇
    change = is_start | (rng.random(N) > cfg["synth_stay"])
    change[0] = True
    seg = np.cumsum(change) - 1
    cl = rng.choice(K, size=seg[-1] + 1, p=pi)[seg]
    rank = rng.choice(m, size=N, p=zipf_p)
    items = (FIRST_BG + cl * m + rank).astype(np.int32)
    burst = np.zeros(N, np.int8)                                           # 0 = 背景；1 = τ；j + 1 = 回访串第 j 个
    # 长程信号：周期 P ~ LogUniform，至少放得下两段
    p_hi = max(sc["p_min"] + 1, min(sc["p_max"], N - 2 * (b + 1)))
    P = int(np.exp(rng.uniform(np.log(sc["p_min"]), np.log(p_hi))))
    tau = int(rng.integers(1, N_TAU + 1))
    rare = np.flatnonzero(pi < np.sort(pi)[K // 2])                         # 长期偏好排在后一半的簇
    f = int(rng.choice(rare)) if len(rare) else int(np.argmin(pi))
    pos, seg_starts = N - (b + 1), []
    while pos >= 0:
        seq = np.r_[tau, FIRST_BG + f * m + rng.choice(m, size=b, p=zipf_p)].astype(np.int32)
        noisy = rng.random(b) < cfg["synth_noise"]                       # 以概率 ε 把回访里的物品换成随机物品
        seq[1:][noisy] = rng.integers(FIRST_BG, FIRST_BG + K * m, size=int(noisy.sum()))
        items[pos:pos + b + 1] = seq
        burst[pos:pos + b + 1] = np.arange(1, b + 2)
        is_start[pos] = True                                               # 每段回访都是一个新会话
        is_start[pos + 1:pos + b + 1] = False
        seg_starts.append(pos)
        pos -= max(b + 2, int(round(P * (1 + rng.uniform(-0.05, 0.05)))))
    gaps = np.where(is_start, rng.lognormal(np.log(86400), 1.0, size=N), rng.exponential(120.0, size=N))
    ts = (1_000_000_000 + np.cumsum(1 + gaps)).astype(np.int64)                # 间隔至少 1 秒
    return {"items": items, "ts": ts, "is_start": is_start, "burst": burst, "N": N, "P": P, "tau": tau,
            "bursts": sorted(seg_starts), "revisit_cluster": f}


def generate(cfg: dict, n_users: int, seed: int) -> dict:
    """n_users 个用户拼成扁平数组 + 偏移量。"""
    rng = np.random.default_rng(seed)
    sc = scaled(cfg)
    m = cfg["synth_items_per_cluster"]
    zipf_p = 1.0 / np.arange(1, m + 1)
    zipf_p /= zipf_p.sum()
    users = [gen_user(rng, cfg, sc, zipf_p) for _ in range(n_users)]
    off = np.r_[0, np.cumsum([u["N"] for u in users])].astype(np.int64)
    cat = {k: np.concatenate([u[k] for u in users]) for k in ("items", "ts", "is_start", "burst")}
    return {**cat, "off": off, "N": off[1:] - off[:-1], "P": np.array([u["P"] for u in users]),
            "tau": np.array([u["tau"] for u in users])}
# --8<-- [end:generate]


def cut_eval(g: dict, seed: int) -> dict:
    """每个评估用户切一个目标：long（最后一段回访的第一个物品，紧跟在回访入口 τ 之后）、recent（会话中间）、
    newsess（会话起点、隔 ≥ 1 小时）。返回切点 cut（目标下标，输入是 [0, cut)）、类型和长程距离 dist
    （到上一段回访的第一个物品的距离：窗口要至少这么长才看得到上一次回访）。"""
    rng = np.random.default_rng(seed)
    n = len(g["N"])
    kinds = rng.choice(len(TYPES), size=n, p=TYPE_SHARE)
    cut, dist = np.zeros(n, np.int64), np.full(n, -1, np.int64)
    for u in range(n):
        o, N = g["off"][u], g["N"][u]
        items, burst, st, ts = g["items"][o:o + N], g["burst"][o:o + N], g["is_start"][o:o + N], g["ts"][o:o + N]
        if kinds[u] == 0:
            taus = np.flatnonzero(burst == 1)
            c = taus[-1] + 1                                                   # 最后一段回访的第一个物品
            dist[u] = c - (taus[-2] + 1) if len(taus) >= 2 else -1
        elif kinds[u] == 1:
            cand = np.flatnonzero((~st) & (burst == 0) & (np.arange(N) > N // 2))
            c = int(rng.choice(cand)) if len(cand) else N - 1
        else:
            gap = np.r_[0, np.diff(ts)]
            cand = np.flatnonzero(st & (burst == 0) & (gap >= 3600) & (np.arange(N) > N // 2))
            c = int(rng.choice(cand)) if len(cand) else N - 1
        cut[u] = c
    return {"cut": cut, "kind": kinds, "dist": dist}


def long_cover(dist: np.ndarray, kinds: np.ndarray, L: int) -> float:
    """长程题里，目标上一次出现还在长度 L 的窗口内的比例（历史里从没出现过的算没覆盖）。"""
    d = dist[kinds == 0]
    return round(float(((d > 0) & (d <= L)).mean()), 4) if len(d) else 0.0


def pad_waste(lengths: np.ndarray, L: int) -> dict:
    """补齐到 L 的浪费：token 浪费 = 1 − ΣLᵢ / (B·L)；注意力 (i, j) 对浪费 = 1 − ΣLᵢ² / (B·L²)。"""
    lens = np.minimum(lengths, L).astype(np.float64)
    return {"pad_token_waste": round(float(1 - lens.sum() / (len(lens) * L)), 4),
            "pad_attn_waste": round(float(1 - (lens ** 2).sum() / (len(lens) * L * L)), 4)}
