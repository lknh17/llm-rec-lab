"""序列数据：ML-1M（leave-one-out）和合成长序列共用一套存储、截断、训练窗口和评估样本。

存储：所有用户的物品和时间拼成扁平数组，off[u] 是第 u 个用户的起点。
训练：只用训练用户的前 n_train[u] 个行为；每条训练样本取一个随机终点 e，窗口 = truncate(seq[:e], L + 1)，
      x = 窗口[:-1]、y = 窗口[1:]，左补齐到 L。
评估：样本 = (用户, 切点 cut)，输入 = truncate(seq[:cut], L)，目标 = seq[cut]。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class SeqData:
    items: np.ndarray            # [总行为数] int32，0 = padding
    ts: np.ndarray               # [总行为数] int64 秒
    off: np.ndarray              # [用户数 + 1]
    n_items: int                 # 词表大小（含 0）
    train_users: np.ndarray      # 训练用户编号
    n_train: np.ndarray          # 每个用户可用于训练的前缀长度（不在训练集里的为 0）
    evals: dict = field(default_factory=dict)   # {"valid": (users, cuts), "test": (users, cuts)}
    meta: dict = field(default_factory=dict)

    def seq(self, u: int) -> tuple[np.ndarray, np.ndarray]:
        o0, o1 = self.off[u], self.off[u + 1]
        return self.items[o0:o1], self.ts[o0:o1]

    def length(self, u: int) -> int:
        return int(self.off[u + 1] - self.off[u])


# --8<-- [start:truncate]
def truncate(n: int, L: int, mode: str) -> tuple[int, int]:
    """长度 n 的历史只保留 L 个：返回保留的区间 [a, b)。
      recent    最近的 L 个：seq[-L:]
      earliest  最早的 L 个：seq[:L]（教学故障）"""
    if n <= L:
        return 0, n
    return (n - L, n) if mode == "recent" else (0, L)
# --8<-- [end:truncate]


def _left_pad(rows: list[np.ndarray], L: int, dtype) -> np.ndarray:
    out = np.zeros((len(rows), L), dtype=dtype)
    for i, r in enumerate(rows):
        if len(r):
            out[i, L - len(r):] = r
    return out


# --8<-- [start:windows]
def train_batch(d: SeqData, rng: np.random.Generator, B: int, L: int, mode: str) -> dict:
    """B 条训练窗口（左补齐到 L）：x [B, L]，y [B, L]（下一个物品，pad 处为 0），ts [B, L]，ts_next [B, L]。"""
    users = d.train_users[rng.integers(0, len(d.train_users), size=B)]
    xs, ys, t0, t1 = [], [], [], []
    for u in users:
        items, ts = d.seq(u)
        e = int(rng.integers(2, d.n_train[u] + 1))                         # 窗口终点（不含）：用 seq[:e]
        a, b = truncate(e, L + 1, mode)
        w, tw = items[a:b], ts[a:b]
        xs.append(w[:-1]); ys.append(w[1:]); t0.append(tw[:-1]); t1.append(tw[1:])
    return {"x": _left_pad(xs, L, np.int64), "y": _left_pad(ys, L, np.int64),
            "ts": _left_pad(t0, L, np.int64), "ts_next": _left_pad(t1, L, np.int64)}
# --8<-- [end:windows]


def eval_batch(d: SeqData, users: np.ndarray, cuts: np.ndarray, L: int, mode: str, pad_to: int | None = None) -> dict:
    """评估样本：输入是切点之前的历史截断到 L，目标 = seq[cut]；最后一个位置的请求时刻 = 目标的时间。"""
    xs, t0, t1, tgt, kept = [], [], [], [], []
    for u, c in zip(users, cuts):
        items, ts = d.seq(int(u))
        a, b = truncate(int(c), L, mode)
        xs.append(items[a:b]); t0.append(ts[a:b]); t1.append(np.r_[ts[a + 1:b], ts[c]])
        tgt.append(items[c]); kept.append((a, b))
    width = pad_to or max(1, max(len(x) for x in xs))
    return {"x": _left_pad(xs, width, np.int64), "ts": _left_pad(t0, width, np.int64),
            "ts_next": _left_pad(t1, width, np.int64), "target": np.array(tgt, np.int64), "kept": kept}


def to_device(b: dict, device) -> dict:
    return {k: torch.from_numpy(v).to(device) for k, v in b.items() if isinstance(v, np.ndarray)}


# ------------------------------------------------------------------ ML-1M
def load_ml1m(files: list[str] | None = None) -> tuple[SeqData, dict]:
    """ML-1M 评分 → 每个用户按时间排的序列（同一秒的评分按电影编号排，顺序本身没有意义）。
    leave-one-out：训练用 seq[:-2]；验证 = (输入 seq[:-2]，目标 seq[-2])；测试 = (输入 seq[:-1]，目标 seq[-1])。"""
    from recbase import data as RD

    raw = RD.load_raw(files)
    r = raw["ratings"]
    order = np.lexsort((np.arange(len(r["user"])), r["user"]))            # 按用户分组，组内保持时间顺序
    user, movie, ts = r["user"][order], r["movie"][order], r["ts"][order]
    movies_used = np.unique(movie)
    item = (np.searchsorted(movies_used, movie) + 1).astype(np.int32)      # 电影 → 1..V−1
    uniq_u, starts = np.unique(user, return_index=True)
    off = np.r_[starts, len(user)].astype(np.int64)
    N = off[1:] - off[:-1]
    n_users = len(uniq_u)
    d = SeqData(items=item, ts=ts.astype(np.int64), off=off, n_items=len(movies_used) + 1,
                train_users=np.arange(n_users), n_train=(N - 2).astype(np.int64))
    d.evals = {"valid": (np.arange(n_users), N - 2), "test": (np.arange(n_users), N - 1)}
    mrow = {m: i for i, m in enumerate(raw["movies"]["movie"].tolist())}
    rows = np.array([mrow.get(int(m), 0) for m in movies_used])
    d.meta = {"titles": [raw["movies"]["title"][i] for i in rows], "genres": raw["movies"]["genres"][rows],
              "genre_names": RD.GENRES}
    info = {"users": int(n_users), "n_items": int(d.n_items), "rows": int(len(item)),
            "len_median": float(np.median(N)), "len_p90": float(np.percentile(N, 90)), "len_max": int(N.max())}
    return d, info


# ------------------------------------------------------------------ 合成
def from_synth(g_train: dict, g_valid: dict, g_test: dict, cuts_valid: dict, cuts_test: dict, n_items: int) -> SeqData:
    """三组互不重叠的合成用户拼成一个 SeqData：训练用户全长参与训练；验证 / 测试用户只按切点评估。"""
    parts = [g_train, g_valid, g_test]
    items = np.concatenate([g["items"] for g in parts])
    ts = np.concatenate([g["ts"] for g in parts])
    N = np.concatenate([g["N"] for g in parts])
    off = np.r_[0, np.cumsum(N)].astype(np.int64)
    n0, n1 = len(g_train["N"]), len(g_valid["N"])
    n_train = np.zeros(len(N), np.int64)
    n_train[:n0] = g_train["N"]
    d = SeqData(items=items.astype(np.int32), ts=ts, off=off, n_items=n_items, train_users=np.arange(n0), n_train=n_train)
    d.evals = {"valid": (np.arange(n0, n0 + n1), cuts_valid["cut"]),
               "test": (np.arange(n0 + n1, n0 + n1 + len(g_test["N"])), cuts_test["cut"])}
    d.meta = {"valid_kind": cuts_valid["kind"], "valid_dist": cuts_valid["dist"],
              "test_kind": cuts_test["kind"], "test_dist": cuts_test["dist"],
              "burst": np.concatenate([g["burst"] for g in parts]), "is_start": np.concatenate([g["is_start"] for g in parts]),
              "P": np.concatenate([g["P"] for g in parts])}
    return d
