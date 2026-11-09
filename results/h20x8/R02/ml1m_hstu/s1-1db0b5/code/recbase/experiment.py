"""recbase 执行入口：评分日志 → 样本与标签 → 切分（固定未来留出集）→ 编码 → 统计特征 → Embedding+MLP → 评估。

三个准备阶段都有缓存（labkit.cache）：
  recbase_raw    解析 CSV（数据集 + 预处理版本）
  recbase_split  标签、切分、训练集词表编码、泄漏体检（+ 切分方式、阈值、抽样）
  recbase_feats  四种口径的历史统计（+ 统计口径、平滑强度）
换种子、换模型参数的场景都直接复用。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from labkit.cache import stage_cache

from . import data as D
from . import features as F
from . import metrics as M
from .model import RecModel, param_counts
from .params import PARAMS
from .train import predict, to_tensors, train

RAW_VERSION = 1
SPLIT_VERSION = 1
FEAT_VERSION = 1
ML1M_ROWS, ML1M_USERS, ML1M_MOVIES = 1_000_209, 6_040, 3_883
MODEL_KEYS = tuple(p.name for p in PARAMS.params if p.affects == "model")
BASE_FIELDS = ("user_id", "movie_id", "gender", "age", "occupation", "zip1", "year_bucket", "genres")
STAT_FIELDS = ("user_cnt_bucket", "user_rate", "user_lcnt", "item_cnt_bucket", "item_rate", "item_lcnt")


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def raw_key(cfg: dict) -> dict:
    return {"v": RAW_VERSION, "dataset": cfg["dataset"]}


def split_key(cfg: dict) -> dict:
    k = {"v": SPLIT_VERSION, "raw": raw_key(cfg), "split": cfg["split"], "label_threshold": cfg["label_threshold"],
         "max_users": cfg["max_users"]}
    if cfg["split"] == "random":
        k["data_seed"] = cfg["data_seed"]
    if cfg["split"] == "user_last":
        k["user_last_k"] = cfg["user_last_k"]
    return k


def feat_key(cfg: dict) -> dict:
    return {"v": FEAT_VERSION, "split": split_key(cfg), "stats_mode": cfg["stats_mode"], "stats_smooth": cfg["stats_smooth"]}


def _iso(t: int) -> str:
    return dt.datetime.fromtimestamp(int(t), dt.UTC).strftime("%Y-%m-%d")


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- 准备阶段
def build_raw(ctx, d: Path) -> None:
    files = D.assets.resolve_dataset_files(D.DATASET)
    raw = D.load_raw(files)
    np.savez(d / "ratings.npz", **raw["ratings"])
    np.savez(d / "users.npz", **raw["users"])
    np.savez(d / "movies.npz", **{k: v for k, v in raw["movies"].items() if k not in ("title", "genres_text")})
    ratings_csv = next(f for f in files if f.endswith("ratings.csv"))
    with open(ratings_csv, "rb") as f:
        head = f.read(160).decode("utf-8", "replace")
    r = raw["ratings"]
    info = {"rows": int(len(r["ts"])), "users": int(len(np.unique(r["user"]))), "movies_rated": int(len(np.unique(r["movie"]))),
            "movies_table": int(len(raw["movies"]["movie"])), "csv_head": head,
            "parsed_head": [{k: int(r[k][i]) for k in ("user", "movie", "rating", "ts")} for i in range(2)]}
    (d / "info.json").write_text(json.dumps(info, ensure_ascii=False))
    (d / "titles.json").write_text(json.dumps({"movie": raw["movies"]["movie"].tolist(), "title": raw["movies"]["title"],
                                               "genres": raw["movies"]["genres_text"]}, ensure_ascii=False))
    if (info["rows"], info["users"], info["movies_table"]) != (ML1M_ROWS, ML1M_USERS, ML1M_MOVIES):
        ctx.warn("ml1m_shape", f"行数 / 用户数 / 电影数 = {info['rows']} / {info['users']} / {info['movies_table']}，"
                               f"和 MovieLens-1M 官方 {ML1M_ROWS} / {ML1M_USERS} / {ML1M_MOVIES} 不一致")


def build_split(cfg: dict, rdir: Path, d: Path) -> None:
    raw = {"ratings": _npz(rdir / "ratings.npz"), "users": _npz(rdir / "users.npz"), "movies": _npz(rdir / "movies.npz")}
    raw = D.subsample_users(raw, cfg["max_users"])
    r, users, movies = raw["ratings"], raw["users"], raw["movies"]
    label = (r["rating"] >= cfg["label_threshold"]).astype(np.float32)
    s, cut = D.split(r["ts"], r["user"], cfg["split"], cfg["data_seed"], cfg["user_last_k"])
    known_u, n_u = D.vocab(r["user"][s == 0])                              # 词表只来自训练集
    known_m, n_m = D.vocab(r["movie"][s == 0])
    uid, mid = D.encode(r["user"], known_u), D.encode(r["movie"], known_m)
    urow = np.full(int(users["user"].max()) + 1, 0)
    urow[users["user"]] = np.arange(len(users["user"]))
    mrow = np.full(int(movies["movie"].max()) + 1, 0)
    mrow[movies["movie"]] = np.arange(len(movies["movie"]))
    ui, mi = urow[r["user"]], mrow[r["movie"]]
    lc = D.leak_checks(r, s, uid, mid)
    fut = np.zeros(len(s), bool)
    fut[s == 1] = lc.pop("later_user_mask_valid")
    fut[s == 2] = D._later_same_key(r["user"], r["ts"], s, 2)
    np.savez(d / "rows.npz", label=label, s=s, user=r["user"], movie=r["movie"], ts=r["ts"], rating=r["rating"], fut=fut,
             user_id=uid, movie_id=mid, gender=users["gender"][ui], age=users["age"][ui],
             occupation=users["occupation"][ui], zip1=users["zip1"][ui],
             year_bucket=D.year_bucket(movies["year"])[mi], genres=movies["genres"][mi])
    m = [s == i for i in range(3)]
    meta = {
        "vocab_sizes": {"user_id": n_u, "movie_id": n_m, "gender": 3, "age": len(D.AGES) + 1, "occupation": 22, "zip1": 11,
                        "year_bucket": D.YEAR_BUCKETS, "genre": len(D.GENRES) + 1,
                        "user_cnt_bucket": F.CNT_BUCKETS, "item_cnt_bucket": F.CNT_BUCKETS},
        "data": {"rows": int(len(s)), "users": int(len(np.unique(r["user"]))), "items": int(len(np.unique(r["movie"]))),
                 "split_rows": {n: int(x.sum()) for n, x in zip(D.SPLIT_NAMES, m)},
                 "label_rate": {n: round(float(label[x].mean()), 4) if x.any() else None for n, x in zip(D.SPLIT_NAMES, m)},
                 "time_range": {"t80": _iso(cut["t80"]), "t90": _iso(cut["t90"]),
                                **{n: [_iso(r["ts"][x].min()), _iso(r["ts"][x].max())] if x.any() else None
                                   for n, x in zip(D.SPLIT_NAMES, m)}},
                 "test_hash": D.test_hash(s), "train_vocab": {"users": n_u - 1, "movies": n_m - 1}, **lc},
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))


# --8<-- [start:build_feats]
def build_feats(cfg: dict, rows: dict, d: Path) -> None:
    """训练 / 验证行用 stats_mode 口径；留出行永远用线上口径（serve_mode）。另存一份全部行都用线上口径的版本：
    同一批验证行"换成线上能拿到的特征"再打一次分，就是 valid_serving。"""
    s, ts, y = rows["s"], rows["ts"], rows["label"]
    mode, serve = cfg["stats_mode"], F.serve_mode(cfg["stats_mode"])
    prior = float(y[s == 0].mean())
    alpha = cfg["stats_smooth"]
    out, check = {}, {"mode": mode, "serve_mode": serve, "prior": round(prior, 4), "single_auc": {}}
    offline = s != 2
    for name, key in (("user", rows["user"]), ("item", rows["movie"])):
        pit = F.stats(key, ts, y, "point_in_time")
        cm, pm = F.stats(key, ts, y, mode) if mode != "point_in_time" else pit
        cs, ps = F.stats(key, ts, y, serve) if serve != "point_in_time" else pit
        cnt, pos = np.where(offline, cm, cs), np.where(offline, pm, ps)     # 留出行退回线上口径
        out[f"{name}_cnt_bucket"], out[f"{name}_rate"], out[f"{name}_lcnt"] = \
            F.cnt_bucket(cnt), F.smooth_rate(cnt, pos, prior, alpha), F.log_cnt(cnt)
        out[f"{name}_cnt_bucket_serve"], out[f"{name}_rate_serve"], out[f"{name}_lcnt_serve"] = \
            F.cnt_bucket(cs), F.smooth_rate(cs, ps, prior, alpha), F.log_cnt(cs)
        out[f"{name}_prev"] = pit[0]                                        # 分组用：样本时刻之前的评分数
        upto = F.rows_upto_now(key, ts)
        check[f"{name}_rows_using_future_share"] = round(float((cm[offline] > upto[offline]).mean()), 4)
        check[f"{name}_rows_using_self_share"] = 1.0 if mode in ("include_self", "full") else 0.0
        rate, rate_s = out[f"{name}_rate"], out[f"{name}_rate_serve"]
        check["single_auc"][f"{name}_rate"] = {
            "train": round(M.auc(y[s == 0], rate[s == 0]), 4), "valid": round(M.auc(y[s == 1], rate[s == 1]), 4),
            "valid_serving": round(M.auc(y[s == 1], rate_s[s == 1]), 4)}
    np.savez(d / "feats.npz", **out)
    (d / "check.json").write_text(json.dumps(check, ensure_ascii=False))
# --8<-- [end:build_feats]


# ---------------------------------------------------------------- 主流程
def run(ctx, cfg: dict) -> None:
    ctx.stage("prepare")
    torch.manual_seed(cfg["seed"])
    if not cfg["feature_groups"]:
        raise ValueError("feature_groups 不能为空")
    rdir = stage_cache(ctx, "recbase_raw", raw_key(cfg), lambda d: build_raw(ctx, d))
    sdir = stage_cache(ctx, "recbase_split", split_key(cfg), lambda d: build_split(cfg, rdir, d))
    rows = _npz(sdir / "rows.npz")
    smeta = json.loads((sdir / "meta.json").read_text())
    fdir = stage_cache(ctx, "recbase_feats", feat_key(cfg), lambda d: build_feats(cfg, rows, d))
    feats = _npz(fdir / "feats.npz")
    stats_check = json.loads((fdir / "check.json").read_text())
    rinfo = json.loads((rdir / "info.json").read_text())
    titles = json.loads((rdir / "titles.json").read_text())
    data = smeta["data"]
    s, y = rows["s"], rows["label"]
    idx = {n: np.flatnonzero(s == i) for i, n in enumerate(D.SPLIT_NAMES)}
    ctx.info("data", {"kind": "公开数据集 MovieLens-1M（评分 ≥ 阈值记为喜欢）", "source": "hazemessam/ml-1m（HF 镜像）",
                      "license": "GroupLens 研究用途条款", **{k: data[k] for k in ("rows", "users", "items", "split_rows",
                                                                                "label_rate", "time_range", "test_hash")},
                      "split": cfg["split"], "stats_mode": cfg["stats_mode"]})
    ctx.trace("raw_csv", {"note": "ratings.csv 开头原文：timestamp 被引号包住、里面还有换行，按字符串读再去空白转整数",
                          "text": rinfo["csv_head"], "parsed": rinfo["parsed_head"]})
    ctx.log(f"数据：{data['rows']:,} 行；train / valid / test = {data['split_rows']}；"
            f"验证集冷启动用户行 {data['cold']['valid']['user_rows']:.1%}；训练集含未来：{data['leak']['train_has_future']}")

    # ---------------- 模型 ----------------
    device = pick_device(cfg["device"])
    model = RecModel(smeta["vocab_sizes"], cfg).to(device)
    used = [*model.sparse, *(["genres"] if model.use_genres else []), *model.dense]
    base = {k: rows[k] for k in BASE_FIELDS} | {k: feats[k] for k in STAT_FIELDS}
    serve = base | {k: feats[f"{k}_serve"] for k in STAT_FIELDS}
    X = to_tensors({k: base[k] for k in used} | {"label": y}, device)
    Xs = to_tensors({k: serve[k] for k in used} | {"label": y}, device)
    pc = param_counts(model)
    ctx.info("model", {"arch": {"mlp": "Embedding + MLP", "deepfm": "Embedding + MLP + FM 二阶交叉",
                                "dcn": "Embedding + MLP ∥ DCN-v2 cross ×3"}[cfg["arch"]],
                       "feature_groups": cfg["feature_groups"], "fields": used, "params": pc,
                       "device": str(device), "dtype": "float32", "train_mode": "full"})
    ctx.log(f"模型：{pc['total']:,} 参数，embedding 占 {pc['embedding_share']:.0%}；MLP 输入 {pc['mlp_input_dim']} 维；字段 {used}")

    gen = torch.Generator().manual_seed(cfg["seed"])                     # 和训练循环同一个种子：第一个 batch 的第一条
    i0 = int(idx["train"][torch.randperm(len(idx["train"]), generator=gen)[0].item()])
    mrow = {m: i for i, m in enumerate(titles["movie"])}
    by_mode = {}
    for mode in F.MODES:
        uc, up = F.stats(rows["user"], rows["ts"], y, mode)
        ic, ip = F.stats(rows["movie"], rows["ts"], y, mode)
        by_mode[mode] = {"user_cnt": int(uc[i0]), "user_pos": int(up[i0]),
                         "user_rate": round(float(F.smooth_rate(uc[i0:i0 + 1], up[i0:i0 + 1], stats_check["prior"], cfg["stats_smooth"])[0]), 4),
                         "item_cnt": int(ic[i0]), "item_pos": int(ip[i0]),
                         "item_rate": round(float(F.smooth_rate(ic[i0:i0 + 1], ip[i0:i0 + 1], stats_check["prior"], cfg["stats_smooth"])[0]), 4)}
    m0 = mrow.get(int(rows["movie"][i0]))
    ctx.trace("sample", {
        "note": "被追踪样本 = 训练第 1 个 batch 的第 1 条（真实打乱结果）。stats_by_mode 是这一行在四种统计口径下的值",
        "raw": {"user": int(rows["user"][i0]), "movie": int(rows["movie"][i0]),
                "title": titles["title"][m0] if m0 is not None else None, "genres": titles["genres"][m0] if m0 is not None else None,
                "rating": int(rows["rating"][i0]), "time": dt.datetime.fromtimestamp(int(rows["ts"][i0]), dt.UTC).isoformat(),
                "label": int(y[i0])},
        "encoded": {k: (base[k][i0].tolist() if k == "genres" else round(float(base[k][i0]), 4)) for k in used},
        "stats_mode": cfg["stats_mode"], "stats_by_mode": by_mode})

    # ---------------- 训练 ----------------
    ctx.stage("train")

    def eval_epoch(e: int) -> dict:
        pv, pt = predict(model, X, idx["valid"]), predict(model, Xs, idx["test"])
        uv, ut = rows["user"][idx["valid"]], rows["user"][idx["test"]]
        yv, yt = y[idx["valid"]], y[idx["test"]]
        out = {"valid_auc": round(M.auc(yv, pv), 5), "valid_gauc": round(M.gauc(yv, pv, uv)[0], 5),
               "valid_logloss": round(M.logloss(yv, pv), 5), "test_auc": round(M.auc(yt, pt), 5),
               "test_gauc": round(M.gauc(yt, pt, ut)[0], 5)}
        for k, v in out.items():
            ctx.metric(k, v, step=None, split=k.split("_")[0], epoch=e)
        first = epochs_seen[0] if epochs_seen else out["valid_auc"]
        epochs_seen.append(out["valid_auc"])
        ctx.metric("valid_auc_delta_pp", round((out["valid_auc"] - first) * 100, 4), step=None, split="valid", epoch=e)
        ctx.check()
        return out

    epochs_seen: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tr = train(ctx, cfg, model, X, idx["train"], eval_epoch)
    per_epoch = tr.pop("per_epoch")
    ctx.info("train", {k: tr[k] for k in ("steps", "seconds", "samples_per_s")})

    # ---------------- 评估 ----------------
    ctx.stage("evaluate")
    yv, yt = y[idx["valid"]], y[idx["test"]]
    uv, ut = rows["user"][idx["valid"]], rows["user"][idx["test"]]
    pv = predict(model, X, idx["valid"])
    pvs = predict(model, Xs, idx["valid"])
    pt = predict(model, Xs, idx["test"])
    valid = M.block(yv, pv, uv, cfg["bootstrap"])
    valid_serving = M.block(yv, pvs, uv)
    test = M.block(yt, pt, ut, cfg["bootstrap"])
    gv, gv_rows = M.groups(yv, pv, uv, M.group_masks(rows["user_id"][idx["valid"]], feats["user_prev"][idx["valid"]],
                                                     feats["item_prev"][idx["valid"]], rows["fut"][idx["valid"]]))
    gt, gt_rows = M.groups(yt, pt, ut, M.group_masks(rows["user_id"][idx["test"]], feats["user_prev"][idx["test"]],
                                                     feats["item_prev"][idx["test"]], rows["fut"][idx["test"]]))
    ytr = y[idx["train"]]
    ref = {name: M.reference(ytr, rows["user"][idx["train"]], rows["movie"][idx["train"]], y[idx[name]],
                             rows["user"][idx[name]], rows["movie"][idx[name]]) for name in ("valid", "test")}

    def sub(a, b):
        return round(a - b, 5) if a is not None and b is not None else None

    gap = {"valid_minus_test_auc": sub(valid["auc"], test["auc"]), "valid_minus_test_gauc": sub(valid["gauc"], test["gauc"]),
           "valid_minus_serving_auc": sub(valid["auc"], valid_serving["auc"]),
           "valid_minus_serving_gauc": sub(valid["gauc"], valid_serving["gauc"])}
    best = max(per_epoch, key=lambda e: e["valid_auc"])
    last = per_epoch[-1]
    overfit = {"valid_auc_drop_from_best": round(best["valid_auc"] - last["valid_auc"], 5),
               "test_auc_drop_from_best": round(best["test_auc"] - last["test_auc"], 5),
               "train_valid_logloss_gap": round(last["valid_logloss"] - last["train_logloss"], 5)}
    stats_used = bool({"user_stats", "item_stats"} & set(cfg["feature_groups"]))

    # 追踪：一条冷启动验证行；一个验证用户的 GAUC 怎么算
    cold = np.flatnonzero(rows["user_id"][idx["valid"]] == 0)
    if len(cold):
        j = int(cold[0])
        ctx.trace("cold_sample", {"note": "验证集里训练集没见过的用户：用户 ID 编码成 0，所有冷启动用户共用这一行 embedding。"
                                          "词表只来自训练集，所以训练集里没有任何一行是 0：这一行从没拿到过梯度，一直是随机初始化",
                                  "user_raw": int(uv[j]), "user_id_encoded": 0, "movie_raw": int(rows["movie"][idx["valid"][j]]),
                                  "label": int(yv[j]), "prob": round(float(pv[j]), 4),
                                  "oov_rows_in_train": int((rows["user_id"][idx["train"]] == 0).sum()),
                                  "cold_rows_share": data["cold"]["valid"]["user_rows"]})
    ex = None
    for u in np.unique(uv):
        m = uv == u
        n_pos = int(yv[m].sum())
        if 2 <= n_pos <= m.sum() - 2 and m.sum() <= 20:
            ex = u
            break
    if ex is not None:
        m = uv == ex
        o = np.argsort(-pv[m])
        ctx.trace("gauc_example", {"note": "同一个用户的验证行按模型分数从高到低；单用户 AUC = 正例排在负例前面的比例",
                                   "user": int(ex), "rows": [{"prob": round(float(pv[m][k]), 4), "label": int(yv[m][k])} for k in o],
                                   "n_pos": int(yv[m].sum()), "n_neg": int((1 - yv[m]).sum()),
                                   "user_auc": round(M.auc(yv[m], pv[m]), 4)})

    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None
    display = {"验证 AUC": valid["auc"], "验证 GAUC": valid["gauc"], "验证 LogLoss": valid["logloss"],
               "留出 AUC": test["auc"], "留出 GAUC": test["gauc"], "验证 − 留出 AUC": gap["valid_minus_test_auc"],
               "验证冷启动用户行占比": data["cold"]["valid"]["user_rows"], "embedding 占比": pc["embedding_share"],
               "最好 epoch": best["epoch"]}
    if stats_used and cfg["stats_mode"] != stats_check["serve_mode"]:
        display["验证 − 同行线上口径 AUC"] = gap["valid_minus_serving_auc"]
    ctx.summary({
        "valid": valid, "test": test, "valid_serving": valid_serving, "gap": gap,
        "reference": ref, "groups": {"valid": gv, "test": gt}, "groups_valid": gv_rows, "groups_test": gt_rows,
        "data": data, "stats_check": stats_check | {"used": stats_used}, "params": pc,
        "train": tr | {"peak_gb": peak_gb}, "epochs": per_epoch, "best_epoch": {"epoch": best["epoch"], "valid_auc": best["valid_auc"]},
        "overfit": overfit,
        "headline": {"valid_auc": valid["auc"], "valid_gauc": valid["gauc"], "test_auc": test["auc"],
                     "valid_minus_test_auc": gap["valid_minus_test_auc"], "embedding_share": pc["embedding_share"]},
        "display": display, "display_pct": ["验证冷启动用户行占比", "embedding 占比"],
    })
    ctx.log(f"验证 AUC {valid['auc']} GAUC {valid['gauc']}；留出 AUC {test['auc']}；参照线（电影平均喜欢率）"
            f"验证 {ref['valid']['item_mean']['auc']}；最好 epoch {best['epoch']}")

    # ---------------- 保存与重新加载 ----------------
    path = ctx.artifact("model.pt")
    torch.save({"state_dict": model.state_dict(), "vocab_sizes": smeta["vocab_sizes"],
                "cfg": {k: cfg[k] for k in MODEL_KEYS}}, path)
    ctx.register_artifact("model.pt", "checkpoint", "排序模型权重")
    ck = torch.load(path, map_location=device)
    again = RecModel(ck["vocab_sizes"], {**cfg, **ck["cfg"]}).to(device)
    again.load_state_dict(ck["state_dict"])
    head = idx["valid"][:4096]
    diff = float(np.abs(predict(again, X, head) - pv[:len(head)]).max()) if len(head) else 0.0
    ctx.trace("reload", {"max_prob_diff": diff, "ok": diff < 1e-5})
    ctx.info("checks", {"reload_identical": diff < 1e-5})
