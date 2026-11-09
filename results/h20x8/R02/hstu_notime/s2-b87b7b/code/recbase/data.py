"""MovieLens-1M：读表、打标签、切分、编码、泄漏检查。只用 numpy + pyarrow。

切分约定（三种模式共用一个固定的"未来留出集"）：
  先取全局时间的 80%、90% 分位点 t80、t90；test = 时间 ≥ t90 的行，所有模式完全相同（相当于 L01 的干净留出集），
  模式只决定 t90 之前的行（pool）怎么分成 train / valid。
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from labkit import assets

DATASET = "ml-1m"
GENRES = ("Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western")
MAX_GENRES = 6
SPLIT_NAMES = ("train", "valid", "test")
AGES = (1, 18, 25, 35, 45, 50, 56)


def _read(path: str, types: dict | None = None) -> pa.Table:
    return pacsv.read_csv(path, parse_options=pacsv.ParseOptions(newlines_in_values=True),
                          convert_options=pacsv.ConvertOptions(column_types=types or {}))


# --8<-- [start:load_raw]
def load_raw(files: list[str] | None = None) -> dict:
    """三张表 → numpy。ratings.csv 的 timestamp 被引号包住且带换行（"978300760\\n"），先按字符串读再去空白转整数。
    评分按 (时间, 用户, 电影) 排序：后面所有"此前 / 之后"的判断都依赖这个顺序。"""
    files = files or assets.resolve_dataset_files(DATASET)
    by_name = {f.rsplit("/", 1)[-1]: f for f in files}
    r = _read(by_name["ratings.csv"], {"timestamp": pa.string()})
    ts = pc.cast(pc.utf8_trim_whitespace(r["timestamp"]), pa.int64()).to_numpy()
    user = r["userId"].to_numpy().astype(np.int64)
    movie = r["movieId"].to_numpy().astype(np.int64)
    rating = r["rating"].to_numpy().astype(np.int8)
    order = np.lexsort((movie, user, ts))
    u = _read(by_name["users.csv"], {"zip_code": pa.string(), "gender": pa.string()})
    m = _read(by_name["movies.csv"], {"title": pa.string(), "genres": pa.string()})
    titles = m["title"].to_pylist()
    years = np.array([int(x.group(1)) if (x := re.search(r"\((\d{4})\)\s*$", t or "")) else 0 for t in titles])
    genre_ids = np.zeros((len(titles), MAX_GENRES), dtype=np.int64)        # [N_movie, 6]，0 是 padding
    gidx = {g: i + 1 for i, g in enumerate(GENRES)}
    for i, gs in enumerate(m["genres"].to_pylist()):
        ids = [gidx[g] for g in (gs or "").split("|") if g in gidx][:MAX_GENRES]
        genre_ids[i, :len(ids)] = ids
    zips = u["zip_code"].to_pylist()
    return {
        "ratings": {"user": user[order], "movie": movie[order], "rating": rating[order], "ts": ts[order]},
        "users": {"user": u["userId"].to_numpy().astype(np.int64),
                  "gender": np.array([1 if g == "F" else 2 for g in u["gender"].to_pylist()]),
                  "age": np.array([AGES.index(a) + 1 if a in AGES else 0 for a in u["age"].to_pylist()]),
                  "occupation": u["occupation"].to_numpy().astype(np.int64) + 1,
                  "zip1": np.array([int(z[0]) + 1 if z and z[0].isdigit() else 0 for z in zips])},
        "movies": {"movie": m["movieId"].to_numpy().astype(np.int64), "title": titles, "year": years,
                   "genres": genre_ids, "genres_text": m["genres"].to_pylist()},
    }
# --8<-- [end:load_raw]


def subsample_users(raw: dict, max_users: int) -> dict:
    """冒烟档：只保留前 max_users 个用户（按用户编号）。"""
    if not max_users:
        return raw
    keep = np.unique(raw["ratings"]["user"])[:max_users]
    mask = np.isin(raw["ratings"]["user"], keep)
    return {**raw, "ratings": {k: v[mask] for k, v in raw["ratings"].items()}}


# --8<-- [start:split]
def split(ts: np.ndarray, user: np.ndarray, mode: str, data_seed: int = 0, k: int = 5) -> tuple[np.ndarray, dict]:
    """每行的切分编号：0 = train，1 = valid，2 = test。行已按时间排序。
    test 固定为 ts ≥ t90；mode 只决定 t90 之前的 pool 怎么分：
      time      train < t80 ≤ valid < t90（训练集里没有"未来"）
      random    pool 内按行号哈希随机 1/9 进 valid（同一用户之后的行为可能在训练集里）
      user_last pool 内每个用户最后 k 条进 valid（行数不到 2k 的用户全留在 train）"""
    t80, t90 = (int(x) for x in np.quantile(ts, [0.8, 0.9]))
    s = np.where(ts >= t90, 2, 0).astype(np.int8)
    pool = s == 0
    if mode == "time":
        s[pool & (ts >= t80)] = 1
    elif mode == "random":
        rng = np.random.default_rng(1_000_003 + data_seed)
        s[pool & (rng.random(len(ts)) < 1 / 9)] = 1
    elif mode == "user_last":
        idx = np.flatnonzero(pool)
        o = idx[np.lexsort((np.arange(len(idx)), user[idx]))]           # pool 内按 (用户, 时间) 排
        u = user[o]
        start = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
        size = np.diff(np.r_[start, len(o)])
        rank_from_end = np.repeat(start + size, size) - np.arange(len(o))  # 1 = 该用户最后一条
        ok = (rank_from_end <= k) & (np.repeat(size, size) >= 2 * k)
        s[o[ok]] = 1
    else:
        raise ValueError(mode)
    return s, {"t80": t80, "t90": t90}
# --8<-- [end:split]


def test_hash(s: np.ndarray) -> str:
    return hashlib.md5(np.flatnonzero(s == 2).astype(np.int64).tobytes()).hexdigest()[:12]


YEAR_BUCKETS = 18


def year_bucket(year: np.ndarray) -> np.ndarray:
    """上映年份 → 5 年一档：1919 年及以前 = 1，1920-1924 = 2，…，1995 年及以后 = 17；未知 = 0。"""
    return np.where(year > 0, np.clip((year - 1915) // 5, 0, YEAR_BUCKETS - 2) + 1, 0).astype(np.int64)


# --8<-- [start:encode]
def vocab(values: np.ndarray) -> tuple[np.ndarray, int]:
    """只用训练集建词表：返回已排序的原始 ID（下标 + 1 就是编码），0 留给训练集没见过的（OOV，冷启动）。"""
    u = np.unique(values)
    return u, len(u) + 1


def encode(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    if len(known) == 0:
        return np.zeros(len(values), dtype=np.int64)
    pos = np.minimum(np.searchsorted(known, values), len(known) - 1)
    return np.where(known[pos] == values, pos + 1, 0).astype(np.int64)
# --8<-- [end:encode]


# --8<-- [start:checks]
def _later_same_key(key: np.ndarray, ts: np.ndarray, s: np.ndarray, target: int) -> np.ndarray:
    """target 集合里的每一行：训练集里同一个 key 有没有时间更晚的行。"""
    last = np.full(int(key.max()) + 1, -1, dtype=np.int64)                # 每个 key 在训练集里的最晚时间
    np.maximum.at(last, key[s == 0], ts[s == 0])
    rows = s == target
    return last[key[rows]] > ts[rows]


def leak_checks(r: dict, s: np.ndarray, uid: np.ndarray, mid: np.ndarray) -> dict:
    """切分的体检：训练集里有没有未来；验证集里多少行的用户在训练集还有更晚的行为；冷启动占比。"""
    ts = r["ts"]
    v, t = s == 1, s == 2
    later_u = _later_same_key(r["user"], ts, s, 1)
    later_i = _later_same_key(r["movie"], ts, s, 1)
    cold = {}
    for name, m in (("valid", v), ("test", t)):
        cold[name] = {"user_rows": round(float((uid[m] == 0).mean()), 4) if m.any() else None,
                      "item_rows": round(float((mid[m] == 0).mean()), 4) if m.any() else None,
                      "users": int(len(np.unique(r["user"][m])))}
    return {"leak": {"train_has_future": bool(v.any() and ts[s == 0].max() > ts[v].min()),
                     "valid_rows_with_later_train_same_user": round(float(later_u.mean()), 4) if len(later_u) else 0.0,
                     "valid_rows_with_later_train_same_item": round(float(later_i.mean()), 4) if len(later_i) else 0.0},
            "cold": cold, "later_user_mask_valid": later_u}
# --8<-- [end:checks]
