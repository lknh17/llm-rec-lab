"""推荐项目的执行入口：准备数据 -> 训练 -> 评估 -> 保存 -> 重新加载并预测。"""

from __future__ import annotations

import numpy as np
import torch

from labkit.cache import stage_cache
from labkit.store import get_run

from . import data as D
from .evaluate import auc, calibration, error_samples, group_metrics, logloss, predict, reference_baselines
from .model import build_model
from .train import to_tensors, train


def pick_device(name: str) -> torch.device:
    if name == "auto":
        # 小 MLP 在 Apple MPS 上每步的调度开销比计算还大，实测 CPU 更快，所以 auto 不选 mps
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def data_key(cfg: dict) -> dict:
    """决定数据准备产物的全部条件。feature_groups 不在里面：所有特征都预先算好，模型按需取用。"""
    name = cfg["dataset"]
    key = {"dataset": "movielens-1m" if name == "ml-1m" else name, "fingerprint": D.dataset_fingerprint(cfg),
           "prep_version": D.PREP_VERSION, "split": cfg["split"], "stats_mode": cfg["stats_mode"]}
    if D.DATASETS[name]["label"] == "threshold":
        key["label_threshold"] = cfg["label_threshold"]
    if name == "synth":
        key["synth"] = {k: cfg[k] for k in ("synth_users", "synth_items", "synth_rows", "synth_alpha", "synth_seed")}
    if cfg["split"] == "random":
        key["seed"] = cfg["seed"]
    return key


def run(ctx, cfg: dict) -> None:
    # ---------------- 1. 数据 ----------------
    ctx.stage("prepare")
    spec = D.DATASETS[cfg["dataset"]]
    if "user_profile" in cfg["feature_groups"] and not spec["profile"]:
        raise ValueError(f"{cfg['dataset']} 没有用户画像，feature_groups 里不能有 user_profile")
    key = data_key(cfg)
    cache_dir = stage_cache(ctx, "rec_data", key, lambda out: D.build(cfg, out))
    data = D.RecData(cache_dir)
    meta = data.meta
    ctx.info("data", {
        "kind": spec["kind"], "source": spec["source"], "dataset": cfg["dataset"], "version": meta["fingerprint"],
        "task": spec["task"].format(t=cfg["label_threshold"]),
        "split_rule": {"time": "全局时间 80/10/10", "random": "随机 80/10/10"}[cfg["split"]],
        "stats_mode": cfg["stats_mode"], "seed": cfg["seed"], "prep_version": D.PREP_VERSION,
        "rows": meta["rows"], "split_rows": meta["split_rows"], "label_rate": meta["label_rate"],
        "data_key": key,
    })
    ctx.info("checks", {"data": meta["checks"]})
    for name, val in meta["checks"].items():
        ctx.log(f"数据检查 {name} = {val}")

    device = pick_device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    tensors = to_tensors(data, device)
    model = build_model(meta, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ctx.info("model", {"arch": "Embedding + MLP", "feature_groups": cfg["feature_groups"],
                       "sparse_fields": model.sparse, "dense_fields": model.dense, "genre_bag": model.use_genres,
                       "mlp_input_dim": model.in_dim, "params": n_params, "device": str(device)})
    ctx.log(f"模型：特征组 {cfg['feature_groups']}，MLP 输入 {model.in_dim} 维，参数 {n_params:,}，设备 {device}")

    resume = None
    if cfg["resume_from"]:
        src = get_run(cfg["resume_from"])
        if src is None or not (src.dir / "artifacts" / "checkpoint_last.pt").exists():
            raise FileNotFoundError(f"找不到运行 {cfg['resume_from']} 的检查点")
        resume = torch.load(src.dir / "artifacts" / "checkpoint_last.pt", map_location=device, weights_only=False)
        ctx.info("resume", {"mode": "续训（恢复优化器与随机数状态）", "from_run": cfg["resume_from"],
                            "epoch": resume["epoch"], "step": resume["step"]})

    # ---------------- 2. 训练 ----------------
    ctx.stage("train")
    hist = train(ctx, cfg, data, model, tensors, device, resume)

    # ---------------- 3. 评估 ----------------
    ctx.stage("evaluate")
    train_idx, valid_idx = data.index("train"), data.index("valid")
    prob = predict(model, tensors, valid_idx)
    y = data.arrays["label"][valid_idx]
    result = {
        "valid": {"auc": round(auc(y, prob), 4), "logloss": round(logloss(y, prob), 4),
                  "calibration": calibration(y, prob), "n": int(len(valid_idx)), "label_rate": round(float(y.mean()), 4)},
        "reference_baselines_valid": reference_baselines(data, train_idx, valid_idx),
        "groups_valid": group_metrics(data, valid_idx, prob),
        "steps": hist["steps"], "best_epoch": hist["best"],
    }
    if cfg["final_test"]:
        test_idx = data.index("test")
        pt = predict(model, tensors, test_idx)
        yt = data.arrays["label"][test_idx]
        result["test"] = {"auc": round(auc(yt, pt), 4), "logloss": round(logloss(yt, pt), 4), "n": int(len(test_idx))}
        ctx.log(f"保留测试集：AUC={result['test']['auc']} LogLoss={result['test']['logloss']}")
    ctx.log(f"验证集：AUC={result['valid']['auc']} LogLoss={result['valid']['logloss']} "
            f"ECE={result['valid']['calibration']['ece']}")

    rows = error_samples(data, valid_idx, prob)
    rng = np.random.default_rng(0)  # 固定的 50 条随机样本：不同运行之间可以逐条对照
    for j in rng.choice(len(valid_idx), size=50, replace=False):
        r = data.raw_row(int(valid_idx[j]))
        r.update({"prob": round(float(prob[j]), 4), "error_type": "随机抽样（固定种子）"})
        rows.append(r)
    ctx.predictions(rows)

    # ---------------- 4. 保存、重新加载、预测 ----------------
    path = ctx.artifact("model.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "vocab_sizes": meta["vocab_sizes"]}, path)
    ctx.register_artifact("model.pt", "model", "仅权重 + 配置，用于推理；续训请用 checkpoint_last.pt")
    loaded = torch.load(path, map_location=device, weights_only=False)
    model2 = build_model({"vocab_sizes": loaded["vocab_sizes"]}, loaded["cfg"]).to(device)
    model2.load_state_dict(loaded["state_dict"])
    prob2 = predict(model2, tensors, valid_idx[:5000])
    max_diff = float(np.abs(prob2 - prob[:5000]).max())
    probe_row = data.raw_row(int(valid_idx[0]))
    probe_row["prob_in_memory"] = round(float(prob[0]), 6)
    probe_row["prob_reloaded"] = round(float(prob2[0]), 6)
    ctx.trace("reload", {"artifact": "artifacts/model.pt", "checked_rows": 5000, "max_abs_diff": max_diff,
                         "ok": max_diff < 1e-6, "probe": probe_row})
    ctx.info("checks", {"reload_max_abs_diff": max_diff})
    ctx.log(f"重新加载核对：5000 条验证样本最大概率差 {max_diff:.2e}")
    ctx.summary(result)
    ctx.summary({"headline": {"valid_auc": result["valid"]["auc"], "valid_logloss": result["valid"]["logloss"],
                              "valid_ece": result["valid"]["calibration"]["ece"], "steps": hist["steps"]}})
