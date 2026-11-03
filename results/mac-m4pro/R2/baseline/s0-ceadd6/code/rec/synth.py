"""可控规模的合成推荐数据：已知真实点击概率，能算出"理论最优 AUC"作为天花板。

生成过程（全部由 seed 决定，可复现）：
  用户 u：潜向量 p_u ~ N(0, I_k)，活跃度 a_u ~ LogNormal → 交互次数按活跃度分配
  物品 i：潜向量 q_i ~ N(0, I_k)，流行度 ~ Zipf(s)，偏置 b_i = 0.5 · z(log 流行度)
  一次交互：按流行度抽物品，时间戳在 [0, T) 内均匀 + 用户自己的起始偏移
           点击概率 p = sigmoid(alpha · <p_u, q_i>/sqrt(k) + b_i + c_u)，标签 ~ Bernoulli(p)
  属性：用户的"性别/年龄段/职业/地区"由 p_u 前 4 维量化得到（有信息量，但不完整）；
        物品的"题材"取 q_i 最大的 1-3 维，"年代"由物品编号决定（无信息量，用作对照）。

因为知道真实概率 p，AUC(label, p) 就是任何模型在这份数据上能达到的上限（贝叶斯最优排序）。
模型 AUC 和它的差距 = 模型还没学到的部分；两者都接近 0.5 说明信号本身太弱（调 alpha）。

两种导出：
- frame()：一行一次交互的表（R1/R2 的单条样本形式，接 data.build）；
- sequences()：每个用户一条按时间排序的行为序列，长度可到 10K（给长序列/生成式推荐实验用），
  分块写 .npy，不在内存里建整张表。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SYNTH_VERSION = 1
K = 8                 # 潜向量维度
ZIPF_S = 1.1          # 物品流行度的 Zipf 指数：越大越集中在头部
T_SPAN = 3 * 365 * 86400  # 3 年的秒数


def fingerprint(n_users: int, n_items: int, n_rows: int, alpha: float, seed: int) -> str:
    s = json.dumps({"v": SYNTH_VERSION, "u": n_users, "i": n_items, "r": n_rows, "a": alpha, "s": seed}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _world(n_users: int, n_items: int, seed: int):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n_users, K)).astype(np.float32)          # [U, k]
    Q = rng.standard_normal((n_items, K)).astype(np.float32)          # [I, k]
    pop = 1.0 / np.arange(1, n_items + 1) ** ZIPF_S
    pop = pop[rng.permutation(n_items)]                               # 流行度和编号无关
    pop /= pop.sum()
    logpop = np.log(pop)
    b_item = 0.5 * (logpop - logpop.mean()) / logpop.std()            # 热门物品略更容易被点
    c_user = 0.3 * rng.standard_normal(n_users).astype(np.float32)    # 用户自身的点击倾向
    activity = rng.lognormal(0.0, 1.0, n_users)
    activity /= activity.sum()
    start = rng.uniform(0, 0.5 * T_SPAN, n_users)                     # 用户进入平台的时间
    return rng, P, Q, pop, b_item.astype(np.float32), c_user, activity, start


def _prob(P, Q, b_item, c_user, u, i, alpha: float) -> np.ndarray:
    logit = alpha * np.einsum("nk,nk->n", P[u], Q[i]) / np.sqrt(K) + b_item[i] + c_user[u]
    return 1.0 / (1.0 + np.exp(-logit))


def frame(n_users: int, n_items: int, n_rows: int, alpha: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """返回 (ratings, users, items, extra)。ratings 的 rating 列是 0/1 点击，oracle_p 是真实概率。"""
    rng, P, Q, pop, b_item, c_user, activity, start = _world(n_users, n_items, seed)
    counts = rng.multinomial(n_rows - n_users, activity) + 1          # 每个用户至少 1 次
    u = np.repeat(np.arange(n_users), counts)
    i = rng.choice(n_items, size=n_rows, p=pop)
    ts = (start[u] + rng.uniform(0, 1, n_rows) * (T_SPAN - start[u])).astype(np.int64)
    p = _prob(P, Q, b_item, c_user, u, i, alpha)
    y = (rng.uniform(0, 1, n_rows) < p).astype(np.int8)
    ratings = pd.DataFrame({"user_id": u + 1, "movie_id": i + 1, "rating": y.astype(np.float32),
                            "timestamp": ts, "oracle_p": p.astype(np.float32)})
    # 同一用户同一物品可能被抽到多次：保留第一次，和 MovieLens"每对只评一次"一致
    ratings = ratings.sort_values(["timestamp"], kind="stable").drop_duplicates(["user_id", "movie_id"]).reset_index(drop=True)

    def q(x: np.ndarray, n: int) -> np.ndarray:  # 按分位数量化成 n 档
        return np.searchsorted(np.quantile(x, np.linspace(0, 1, n + 1)[1:-1]), x)

    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "gender": np.where(P[:, 0] > 0, "M", "F"),
        "age": q(P[:, 1], 7),
        "occupation": q(P[:, 2], 21),
        "zip": q(P[:, 3], 10).astype(str),
    })
    top = np.argsort(-Q, axis=1)                                      # 题材：潜向量最大的 1-3 维（共 K 种）
    n_g = 1 + (np.arange(n_items) % 3)
    genres = ["|".join(f"g{int(g)}" for g in top[j, : n_g[j]]) for j in range(n_items)]
    items = pd.DataFrame({"movie_id": np.arange(1, n_items + 1), "title": [f"item {j + 1}" for j in range(n_items)],
                          "genres": genres, "year": 1950 + (np.arange(n_items) % 70)})
    extra = {"oracle_note": "oracle_p 是生成时的真实点击概率；AUC(label, oracle_p) 是理论上限",
             "alpha": alpha, "k": K, "zipf_s": ZIPF_S}
    return ratings, users, items, extra


def sequences(out_dir: Path, n_users: int, n_items: int, seq_len: int, alpha: float, seed: int,
              chunk_users: int = 4096) -> dict:
    """每个用户一条长度为 seq_len 的行为序列（按时间排序）。写 items.npy / labels.npy / oracle.npy，形状 [U, L]。

    10K 长度 × 1 万用户 = 1 亿个位置：int32 物品 400 MB + int8 标签 100 MB + float16 概率 200 MB。
    分块生成，峰值内存约 chunk_users × L × 16 字节。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng, P, Q, pop, b_item, c_user, _, _ = _world(n_users, n_items, seed)
    items = np.lib.format.open_memmap(out_dir / "items.npy", "w+", np.int32, (n_users, seq_len))
    labels = np.lib.format.open_memmap(out_dir / "labels.npy", "w+", np.int8, (n_users, seq_len))
    oracle = np.lib.format.open_memmap(out_dir / "oracle.npy", "w+", np.float16, (n_users, seq_len))
    for s in range(0, n_users, chunk_users):
        e = min(n_users, s + chunk_users)
        u = np.repeat(np.arange(s, e), seq_len)
        i = rng.choice(n_items, size=len(u), p=pop)
        p = _prob(P, Q, b_item, c_user, u, i, alpha)
        y = rng.uniform(0, 1, len(u)) < p
        items[s:e] = (i + 1).reshape(e - s, seq_len)
        labels[s:e] = y.reshape(e - s, seq_len)
        oracle[s:e] = p.reshape(e - s, seq_len)
    meta = {"n_users": n_users, "n_items": n_items, "seq_len": seq_len, "alpha": alpha, "seed": seed,
            "version": SYNTH_VERSION, "files": {"items.npy": "int32 [U, L]，0 为 padding，物品编号从 1 开始",
                                                "labels.npy": "int8 [U, L]", "oracle.npy": "float16 [U, L] 真实点击概率"}}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    for a in (items, labels, oracle):
        a.flush()
    return meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="导出长行为序列：python -m rec.synth out_dir --users 10000 --seq-len 10000")
    ap.add_argument("out_dir")
    ap.add_argument("--users", type=int, default=10000)
    ap.add_argument("--items", type=int, default=100000)
    ap.add_argument("--seq-len", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(sequences(Path(a.out_dir), a.users, a.items, a.seq_len, a.alpha, a.seed), ensure_ascii=False, indent=2))
