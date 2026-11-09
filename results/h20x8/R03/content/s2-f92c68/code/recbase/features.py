"""历史统计特征：某用户（或电影）在样本时刻之前被评分了多少次、其中多少次是喜欢。四种口径只差"统计到哪一刻为止"。"""

from __future__ import annotations

import numpy as np

MODES = ("point_in_time", "daily", "include_self", "full")
CNT_BUCKETS = 14


def serve_mode(mode: str) -> str:
    """线上能拿到的口径：含本行标签或含未来的两种，上线时只能退回时点正确的口径。"""
    return {"include_self": "point_in_time", "full": "point_in_time"}.get(mode, mode)


def _grouped(key: np.ndarray, t: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每行：同一 key、t 严格更小的行数 cnt 和正例数 pos；以及同一 (key, t) 的行数 same（含本行）。
    做法：按 (key, t) 排序 → 相同 (key, t) 并成一组 → 组内计数的前缀和减去本 key 第一组之前的前缀和。"""
    n = len(key)
    if n == 0:
        z = np.zeros(0, np.int64)
        return z, z, z
    o = np.lexsort((t, key))                                             # 按 (key, t) 排序
    k, tt, y = key[o], t[o], label[o].astype(np.int64)
    new_key = np.r_[True, k[1:] != k[:-1]]
    new_grp = new_key | np.r_[True, tt[1:] != tt[:-1]]                   # (key, t) 组的开头
    gid = np.cumsum(new_grp) - 1                                         # [n] 每行所在组
    g_n = np.bincount(gid)                                               # [G] 组内行数
    g_pos = np.bincount(gid, weights=y).astype(np.int64)
    g_first = np.maximum.accumulate(np.where(new_key[new_grp], np.arange(len(g_n)), 0))  # 本 key 的第一组
    c_n, c_p = np.cumsum(g_n) - g_n, np.cumsum(g_pos) - g_pos            # 本组之前（全局）的累计
    cnt, pos, same = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
    cnt[o] = (c_n - c_n[g_first])[gid]
    pos[o] = (c_p - c_p[g_first])[gid]
    same[o] = g_n[gid]
    return cnt, pos, same


# --8<-- [start:stats]
def stats(key: np.ndarray, ts: np.ndarray, label: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """每行的 (此前次数 cnt, 此前正例数 pos)。
      point_in_time  时间严格早于本行（线上请求时刻能拿到的；同一秒的行互相看不见）
      daily          严格早于本行所在的那一天（T+1 天级任务产出的特征）
      include_self   point_in_time + 本行自己的标签（特征里含答案）
      full           该 key 的全部行，含本行和未来（离线一次性聚合再拼回样本）"""
    if mode in ("point_in_time", "include_self"):
        cnt, pos, _ = _grouped(key, ts, label)
        if mode == "include_self":
            cnt, pos = cnt + 1, pos + label.astype(np.int64)
    elif mode == "daily":
        cnt, pos, _ = _grouped(key, ts // 86400, label)
    elif mode == "full":
        tot_n = np.bincount(key)
        tot_p = np.bincount(key, weights=label).astype(np.int64)
        cnt, pos = tot_n[key], tot_p[key]
    else:
        raise ValueError(mode)
    return cnt, pos
# --8<-- [end:stats]


def rows_upto_now(key: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """同一 key、时间 ≤ 本行时间的行数（含本行）。统计次数超过它，说明特征里数进了未来的行。"""
    cnt, _, same = _grouped(key, ts, np.zeros(len(key), np.int64))
    return cnt + same


def smooth_rate(cnt: np.ndarray, pos: np.ndarray, prior: float, alpha: float) -> np.ndarray:
    """历史很少时向先验收缩：(pos + α × prior) / (cnt + α)。"""
    return ((pos + alpha * prior) / (cnt + alpha)).astype(np.float32)


def cnt_bucket(cnt: np.ndarray) -> np.ndarray:
    """0 → 0；1 → 1；2-3 → 2；4-7 → 3 …… log2 分桶，上限 CNT_BUCKETS − 1。"""
    b = np.zeros(len(cnt), np.int64)
    nz = cnt > 0
    b[nz] = np.floor(np.log2(cnt[nz])).astype(np.int64) + 1
    return np.minimum(b, CNT_BUCKETS - 1)


def log_cnt(cnt: np.ndarray) -> np.ndarray:
    return (np.log1p(cnt) / 8).astype(np.float32)
