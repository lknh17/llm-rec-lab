"""样本读取与构造：原始数据集 -> 带标签、切分、特征编码的数组。

支持四个数据集（参数 dataset），统一成同一套列：
  ratings: user_id, movie_id, rating, timestamp      （movie_id 泛指物品）
  users:   user_id, gender, age, occupation, zip     （四个"用户画像槽位"，不同数据集映射不同字段）
  movies:  movie_id, title, genres(以 | 分隔), year

任务定义（必须说清楚）：
  MovieLens 只有"用户给电影打了分"，没有曝光日志。所以任务是"已知用户 u 在时间 t 给电影 i 打了分，
  预测这个分数是否 >= label_threshold（喜欢）"。负样本是"看过但打了低分"，不是"曝光未点击"。
  KuaiRand 是真实曝光日志：标签 = is_click，负样本才是"曝光未点击"，和线上 CTR 任务一致。
  合成数据的标签按已知概率抽样，能算出理论最优 AUC。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from labkit.paths import assets_dir

from . import synth

PREP_VERSION = 5  # 预处理逻辑一改就加 1，旧缓存自动失效（5：增加 dataset 维度，题材编码向量化）

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
MAX_GENRES = 6
CNT_BUCKETS = 14  # log2 分桶：0, 1, 2-3, 4-7, ..., >= 4096
SMOOTH_A = 20.0   # 喜欢率的贝叶斯平滑强度：相当于先验里有 20 次评分

SPLIT_NAMES = {0: "train", 1: "valid", 2: "test"}

DATASETS = {
    "ml-1m": {"source": "MovieLens-1M（GroupLens）", "kind": "真实公开数据", "label": "threshold", "profile": True,
              "dir": "ml-1m", "fp_file": "ratings.dat",
              "task": "已知用户给电影打了分，预测评分是否 >= {t}。负样本 = 看过但打低分，不是曝光未点击"},
    "ml-20m": {"source": "MovieLens-20M（GroupLens）", "kind": "真实公开数据", "label": "threshold", "profile": False,
               "dir": "ml-20m", "fp_file": "ratings.csv",
               "task": "同 ML-1M（评分是 0.5 星粒度），预测评分是否 >= {t}。没有用户画像，user_profile 特征组不可用"},
    "kuairand-pure": {"source": "KuaiRand-Pure（快手，CIKM'22，CC BY-SA 4.0）", "kind": "真实公开曝光日志",
                      "label": "binary", "profile": True, "dir": "KuaiRand-Pure/data",
                      "fp_file": "log_standard_4_08_to_4_21_pure.csv",
                      "task": "曝光日志：预测 is_click。负样本 = 曝光未点击（标准推荐流量，不含随机曝光）"},
    "synth": {"source": "合成数据（潜在因子 + 已知真实概率）", "kind": "合成数据", "label": "binary", "profile": True,
              "task": "按已知概率抽样的点击；AUC(label, 真实概率) 是理论上限"},
}
KUAIRAND_LOGS = ["log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"]


def dataset_dir(name: str = "ml-1m") -> Path:
    return assets_dir() / "datasets" / DATASETS[name]["dir"]


def dataset_fingerprint(cfg: dict | None = None) -> str:
    name = (cfg or {}).get("dataset", "ml-1m")
    if name == "synth":
        return synth.fingerprint(cfg["synth_users"], cfg["synth_items"], cfg["synth_rows"], cfg["synth_alpha"],
                                 cfg["synth_seed"])
    path = dataset_dir(name) / DATASETS[name]["fp_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"缺少数据集 {name}：{path}。运行 python -m labkit assets {name} 下载（离线环境请事先拷入 {dataset_dir(name)}）")
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# --8<-- [start:load_raw]
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读 MovieLens-1M 的三张原始表。'::' 分隔，latin-1 编码（片名里有欧洲字符）。"""
    d = dataset_dir("ml-1m")
    ratings = pd.read_csv(d / "ratings.dat", sep="::", engine="python", encoding="latin-1",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    users = pd.read_csv(d / "users.dat", sep="::", engine="python", encoding="latin-1",
                        names=["user_id", "gender", "age", "occupation", "zip"])
    movies = pd.read_csv(d / "movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["movie_id", "title", "genres"])
    # 片名末尾的 "(1995)" 是上映年份
    movies["year"] = movies["title"].map(lambda t: int(m.group(1)) if (m := re.search(r"\((\d{4})\)\s*$", t)) else 0)
    return ratings, users, movies
# --8<-- [end:load_raw]


# --8<-- [start:load_other]
def load_ml20m() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """MovieLens-20M：2000 万条评分，0.5 星粒度，csv 带表头。没有用户画像：四个画像槽位全部填"未知"。"""
    d = dataset_dir("ml-20m")
    ratings = pd.read_csv(d / "ratings.csv", dtype={"userId": np.int64, "movieId": np.int64, "rating": np.float32,
                                                    "timestamp": np.int64})
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    movies = pd.read_csv(d / "movies.csv").rename(columns={"movieId": "movie_id"})
    movies["year"] = movies["title"].map(
        lambda t: int(m.group(1)) if (m := re.search(r"\((\d{4})\)\s*$", str(t).strip())) else 0)
    users = pd.DataFrame({"user_id": np.sort(ratings["user_id"].unique())})
    for col in ("gender", "age", "occupation", "zip"):
        users[col] = "unknown"
    return ratings, users, movies


def load_kuairand() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """KuaiRand-Pure 标准流量日志（4/08-5/08，约 144 万次曝光）。

    画像槽位映射：gender <- user_active_degree，age <- register_days_range，
    occupation <- follow_user_num_range，zip <- fans_user_num_range（取档位序号）。
    物品：题材 <- tag（逗号分隔的类目编号），年代 <- upload_dt 的年份。
    """
    d = dataset_dir("kuairand-pure")
    cols = ["user_id", "video_id", "time_ms", "is_click"]
    logs = pd.concat([pd.read_csv(d / f, usecols=cols) for f in KUAIRAND_LOGS], ignore_index=True)
    # 同一用户同一视频可能曝光多次：保留每次曝光（真实日志就是这样），pair_overlap 检查会如实报告
    ratings = pd.DataFrame({"user_id": logs["user_id"].to_numpy(np.int64),
                            "movie_id": logs["video_id"].to_numpy(np.int64),
                            "rating": logs["is_click"].to_numpy(np.float32),
                            "timestamp": (logs["time_ms"] // 1000).to_numpy(np.int64)})
    uf = pd.read_csv(d / "user_features_pure.csv")
    fans_levels = ["0", "[1,10)", "[10,100)", "[100,1k)", "[1k,5k)", "[5k,1w)", "[1w,10w)"]
    users = pd.DataFrame({
        "user_id": uf["user_id"], "gender": uf["user_active_degree"].astype(str),
        "age": uf["register_days_range"].astype(str), "occupation": uf["follow_user_num_range"].astype(str),
        "zip": uf["fans_user_num_range"].astype(str).map(lambda v: str(fans_levels.index(v)) if v in fans_levels else "x"),
    })
    vf = pd.read_csv(d / "video_features_basic_pure.csv", usecols=["video_id", "upload_dt", "tag"])
    movies = pd.DataFrame({
        "movie_id": vf["video_id"], "title": "video " + vf["video_id"].astype(str),
        "genres": vf["tag"].fillna("").astype(str).str.replace(",", "|"),
        "year": pd.to_datetime(vf["upload_dt"], errors="coerce").dt.year.fillna(0).astype(int),
    })
    return ratings, users, movies
# --8<-- [end:load_other]


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    name = cfg.get("dataset", "ml-1m")
    if name == "ml-1m":
        return (*load_raw(), {})
    if name == "ml-20m":
        return (*load_ml20m(), {})
    if name == "kuairand-pure":
        return (*load_kuairand(), {})
    if name == "synth":
        return synth.frame(cfg["synth_users"], cfg["synth_items"], cfg["synth_rows"], cfg["synth_alpha"],
                           cfg["synth_seed"])
    raise ValueError(f"未知数据集 {name}")


def genre_vocab_for(name: str, movies: pd.DataFrame) -> dict:
    if name in ("ml-1m", "ml-20m"):
        return {g: i + 1 for i, g in enumerate(GENRES)}  # ML-20M 多出的 IMAX 等题材当作未知，忽略
    tags = sorted({t for gs in movies["genres"].astype(str) for t in gs.split("|") if t})
    return {g: i + 1 for i, g in enumerate(tags)}


# --8<-- [start:split]
def assign_split(ratings: pd.DataFrame, mode: str, seed: int) -> np.ndarray:
    """返回每行的切分编号：0=train, 1=valid, 2=test。

    time：按全局时间切。t80、t90 是时间戳的 80%、90% 分位点，
          train 全部早于 t80，valid 在 [t80, t90)，test 不早于 t90 —— 训练集里没有"未来"。
    random：随机打乱后 80/10/10。同一个用户 2002 年的评分可能进训练集，用来预测他 2000 年的评分。
    """
    n = len(ratings)
    if mode == "time":
        ts = ratings["timestamp"].to_numpy()
        t80, t90 = np.quantile(ts, [0.8, 0.9])
        return np.where(ts < t80, 0, np.where(ts < t90, 1, 2)).astype(np.int8)
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return np.where(u < 0.8, 0, np.where(u < 0.9, 1, 2)).astype(np.int8)
# --8<-- [end:split]


# --8<-- [start:stats]
def history_stats(df: pd.DataFrame, key: str, prior: float, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """每一行样本的"该用户（或电影）历史评分数"和"平滑后的历史喜欢率"。

    point_in_time：只统计时间戳严格早于本样本的行为。同一秒的多条评分互相看不见，
                   这和线上"特征在请求时刻之前生成"一致。
    full_data：    统计该 key 的全部行为——包括本样本自己的标签和它之后的行为。这是标签泄漏：
                   特征里直接含有答案，离线指标会虚高，上线后拿不到这些信息。
    """
    if mode == "point_in_time":
        g = (df.groupby([key, "timestamp"], sort=True)
               .agg(n=("label", "size"), pos=("label", "sum")).reset_index())
        # 累加到当前时间戳为止，再减掉当前时间戳自己 -> 严格早于本时刻的计数
        g["cnt"] = g.groupby(key)["n"].cumsum() - g["n"]
        g["pos_before"] = g.groupby(key)["pos"].cumsum() - g["pos"]
        merged = df[[key, "timestamp"]].merge(g[[key, "timestamp", "cnt", "pos_before"]],
                                              on=[key, "timestamp"], how="left")
        cnt, pos = merged["cnt"].to_numpy(), merged["pos_before"].to_numpy()
    elif mode == "full_data":
        agg = df.groupby(key)["label"].agg(["size", "sum"])
        cnt = df[key].map(agg["size"]).to_numpy()
        pos = df[key].map(agg["sum"]).to_numpy()
    else:
        raise ValueError(mode)
    # 平滑：历史很少时向全局先验收缩，避免 1 次评分就得到 0% 或 100% 的喜欢率
    rate = (pos + SMOOTH_A * prior) / (cnt + SMOOTH_A)
    return cnt.astype(np.int64), rate.astype(np.float32)
# --8<-- [end:stats]


def cnt_bucket(cnt: np.ndarray) -> np.ndarray:
    """0 -> 0；1 -> 1；2-3 -> 2；4-7 -> 3 ……  log2 分桶，cap 在 CNT_BUCKETS-1。"""
    b = np.zeros_like(cnt)
    nz = cnt > 0
    b[nz] = np.floor(np.log2(cnt[nz])).astype(np.int64) + 1
    return np.minimum(b, CNT_BUCKETS - 1)


def single_feature_auc(score: np.ndarray, label: np.ndarray) -> float:
    from .evaluate import auc

    return auc(label, score)


# --8<-- [start:encode]
def build(cfg: dict, out_dir: Path) -> dict:
    """完整的数据准备：读表 -> 打标签 -> 切分 -> 统计特征 -> 编码 -> 泄漏检查 -> 落盘。"""
    name = cfg.get("dataset", "ml-1m")
    ratings, users, movies, extra = load_dataset(cfg)
    df = ratings.merge(users, on="user_id", how="left").merge(movies, on="movie_id", how="left")
    df = df.sort_values(["timestamp", "user_id", "movie_id"], kind="stable").reset_index(drop=True)
    if DATASETS[name]["label"] == "threshold":
        df["label"] = (df["rating"] >= cfg["label_threshold"]).astype(np.int8)
    else:  # 点击日志 / 合成数据：rating 列就是 0/1 标签，label_threshold 不起作用
        df["label"] = (df["rating"] > 0.5).astype(np.int8)
    df["split"] = assign_split(df, cfg["split"], cfg["seed"])
    train = df[df["split"] == 0]
    prior = float(train["label"].mean())  # 先验只用训练集估计

    # ID 词表只用训练集建：验证/测试里没见过的用户、电影映射到 0（OOV），这就是冷启动
    user_vocab = {u: i + 1 for i, u in enumerate(sorted(train["user_id"].unique()))}
    movie_vocab = {m: i + 1 for i, m in enumerate(sorted(train["movie_id"].unique()))}
    # 属性词表来自属性表本身（不含标签信息），所有用户/电影都能编码
    gender_vocab = {g: i + 1 for i, g in enumerate(sorted(users["gender"].astype(str).unique()))}  # ML-1M: F=1, M=2
    age_vocab = {a: i + 1 for i, a in enumerate(sorted(users["age"].unique()))}
    occ_vocab = {o: i + 1 for i, o in enumerate(sorted(users["occupation"].unique()))}
    genre_vocab = genre_vocab_for(name, movies)

    arrays: dict[str, np.ndarray] = {
        "user_id": df["user_id"].map(user_vocab).fillna(0).to_numpy(np.int64),
        "movie_id": df["movie_id"].map(movie_vocab).fillna(0).to_numpy(np.int64),
        # 画像缺失（日志里有、画像表里没有的用户）映射到 0
        "gender": df["gender"].astype(str).map(gender_vocab).fillna(0).to_numpy(np.int64),
        "age": df["age"].map(age_vocab).fillna(0).to_numpy(np.int64),
        "occupation": df["occupation"].map(occ_vocab).fillna(0).to_numpy(np.int64),
        "zip1": df["zip"].astype(str).str[0].map(lambda c: int(c) + 1 if c.isdigit() else 0).to_numpy(np.int64),
        # 年代：1919 年及以前归一桶，之后每 10 年一桶
        "year_bucket": np.clip((df["year"].fillna(0).to_numpy() - 1910) // 10, 0, 9).astype(np.int64) + 1,
    }
    # 题材：先给每部电影编好 [6] 的题材编号，再按行取（2000 万行时逐行循环要几分钟）
    table = np.zeros((len(movies) + 1, MAX_GENRES), dtype=np.int64)  # [M+1, 6]，最后一行给属性表里没有的物品
    for row, gs in enumerate(movies["genres"].fillna("").astype(str).str.split("|")):
        ids = [genre_vocab[g] for g in gs if g in genre_vocab][:MAX_GENRES]
        table[row, : len(ids)] = ids
    pos = pd.Series(np.arange(len(movies)), index=movies["movie_id"].to_numpy())
    arrays["genres"] = table[df["movie_id"].map(pos).fillna(len(movies)).to_numpy(np.int64)]  # [N, 6]，0 是 padding

    for key, prefix in (("user_id", "user"), ("movie_id", "item")):
        cnt, rate = history_stats(df, key, prior, cfg["stats_mode"])
        arrays[f"{prefix}_cnt_bucket"] = cnt_bucket(cnt)
        arrays[f"{prefix}_like_rate"] = rate
        arrays[f"{prefix}_cnt_feature"] = cnt
        # 分组分析和错例展示一律用"样本时刻之前的真实历史数"，不随特征口径变化，保证不同运行分组一致
        pit_cnt, _ = history_stats(df, key, prior, "point_in_time")
        arrays[f"{prefix}_cnt_raw"] = pit_cnt
    arrays["label"] = df["label"].to_numpy(np.float32)
    arrays["split"] = df["split"].to_numpy(np.int8)
    arrays["raw_user_id"] = df["user_id"].to_numpy(np.int64)
    arrays["raw_movie_id"] = df["movie_id"].to_numpy(np.int64)
    arrays["timestamp"] = df["timestamp"].to_numpy(np.int64)
    arrays["rating"] = df["rating"].to_numpy(np.float32)
    if "oracle_p" in df:
        arrays["oracle_p"] = df["oracle_p"].to_numpy(np.float32)
# --8<-- [end:encode]

    vocab_sizes = {
        "user_id": len(user_vocab) + 1, "movie_id": len(movie_vocab) + 1, "gender": 3,
        "age": len(age_vocab) + 1, "occupation": len(occ_vocab) + 1, "zip1": 11, "year_bucket": 11,
        "genre": len(genre_vocab) + 1, "user_cnt_bucket": CNT_BUCKETS, "item_cnt_bucket": CNT_BUCKETS,
    }
    checks = leakage_checks(df, arrays)
    if "oracle_p" in arrays:  # 合成数据：按真实概率排序的 AUC = 理论上限
        for k, split_name in SPLIT_NAMES.items():
            m = arrays["split"] == k
            checks[f"oracle_auc_{split_name}"] = round(single_feature_auc(arrays["oracle_p"][m], arrays["label"][m]), 4)
    np.savez(out_dir / "arrays.npz", **arrays)
    meta = {
        "dataset": name, "fingerprint": dataset_fingerprint(cfg), "prep_version": PREP_VERSION, "extra": extra,
        "rows": int(len(df)), "prior_like_rate_train": prior, "vocab_sizes": vocab_sizes,
        "split_rows": {SPLIT_NAMES[k]: int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "label_rate": {SPLIT_NAMES[k]: float(v) for k, v in df.groupby("split")["label"].mean().items()},
        "checks": checks,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    movies[["movie_id", "title", "genres"]].to_csv(out_dir / "movies.csv", index=False)
    users.to_csv(out_dir / "users.csv", index=False)
    return meta


# --8<-- [start:leakage_checks]
def leakage_checks(df: pd.DataFrame, arrays: dict) -> dict:
    """数据层面能自动发现的可疑信号。只陈述事实，判断交给人。"""
    out: dict = {}
    split = df["split"].to_numpy()
    ts = df["timestamp"]
    for k, name in SPLIT_NAMES.items():
        part = ts[split == k]
        out[f"time_range_{name}"] = [str(pd.to_datetime(part.min(), unit="s").date()),
                                     str(pd.to_datetime(part.max(), unit="s").date())] if len(part) else None
    # 训练集的最晚时间是否晚于验证集的最早时间：time 切分时必须是 False
    out["train_has_future"] = bool(ts[split == 0].max() > ts[split == 1].min())
    # 同一 (用户, 电影) 是否同时出现在训练和评估中（MovieLens 每对只评一次，应为 0）
    pair = df["user_id"].to_numpy(np.int64) * (int(df["movie_id"].max()) + 1) + df["movie_id"].to_numpy(np.int64)
    out["pair_overlap_train_valid"] = int(len(np.intersect1d(pair[split == 0], pair[split == 1])))
    # 冷启动占比：验证集里有多少行的用户/电影在训练集没出现过
    for k, name in ((1, "valid"), (2, "test")):
        mask = split == k
        out[f"cold_user_share_{name}"] = round(float((arrays["user_id"][mask] == 0).mean()), 4)
        out[f"cold_movie_share_{name}"] = round(float((arrays["movie_id"][mask] == 0).mean()), 4)
    # 统计特征计入的行为数 vs 样本时刻之前真实存在的行为数。前者更大，说明特征"看到"了样本时刻及之后的行为
    for prefix in ("user", "item"):
        extra = arrays[f"{prefix}_cnt_feature"] - arrays[f"{prefix}_cnt_raw"]
        out[f"{prefix}_stats_rows_at_or_after_sample_share"] = round(float((extra > 0).mean()), 4)
    # 单特征 AUC：只用"用户历史喜欢率"一个数去排序。单特征就和整个模型差不多好，要怀疑它含有标签
    label = arrays["label"]
    for feat in ("user_like_rate", "item_like_rate"):
        for k, name in ((0, "train"), (1, "valid")):
            mask = split == k
            out[f"single_feature_auc_{feat}_{name}"] = round(single_feature_auc(arrays[feat][mask], label[mask]), 4)
    return out
# --8<-- [end:leakage_checks]


class RecData:
    """缓存目录里的数组 -> 按切分取出的视图。"""

    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        z = np.load(cache_dir / "arrays.npz")
        self.arrays = {k: z[k] for k in z.files}
        self.meta = json.loads((cache_dir / "meta.json").read_text())
        self.movies = pd.read_csv(cache_dir / "movies.csv").set_index("movie_id")
        self.users = pd.read_csv(cache_dir / "users.csv").set_index("user_id")

    def index(self, split_name: str) -> np.ndarray:
        code = {v: k for k, v in SPLIT_NAMES.items()}[split_name]
        return np.nonzero(self.arrays["split"] == code)[0]

    def raw_row(self, i: int) -> dict:
        """一条样本的原始字段，用于样本追踪和错例展示。"""
        a = self.arrays
        uid, mid = int(a["raw_user_id"][i]), int(a["raw_movie_id"][i])
        u = self.users.loc[uid] if uid in self.users.index else _UNKNOWN_USER
        m = self.movies.loc[mid] if mid in self.movies.index else {"title": f"item {mid}", "genres": ""}
        return {
            "row": int(i), "split": SPLIT_NAMES[int(a["split"][i])], "user_id": uid, "movie_id": mid,
            "title": m["title"], "genres": m["genres"], "gender": u["gender"], "age": _num(u["age"]),
            "occupation": _num(u["occupation"]), "zip": str(u["zip"]), "rating": _num(a["rating"][i]),
            "label": int(a["label"][i]),
            "time": str(pd.to_datetime(int(a["timestamp"][i]), unit="s")),
            "user_history_cnt": int(a["user_cnt_raw"][i]), "item_history_cnt": int(a["item_cnt_raw"][i]),
        }


_UNKNOWN_USER = {"gender": "unknown", "age": "unknown", "occupation": "unknown", "zip": "unknown"}


def _num(v):
    """评分、年龄段这些字段在不同数据集里可能是整数、半星或字符串：能转整数就转整数。"""
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return str(v)
