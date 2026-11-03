"""评估：AUC、LogLoss、校准、分组指标和错例。每个指标的口径写在函数里。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


# --8<-- [start:metrics]
def auc(label: np.ndarray, score: np.ndarray) -> float:
    """AUC = 随机抽一个正样本和一个负样本，正样本得分更高的概率（并列算一半）。

    用秩和公式计算：O(N log N)。分母是 正样本数 x 负样本数；只有一类样本时无定义，返回 nan。
    """
    label = np.asarray(label).astype(bool)
    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(score)).rank(method="average").to_numpy()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def logloss(label: np.ndarray, prob: np.ndarray) -> float:
    """平均交叉熵。分母是样本数。它同时惩罚排序错和概率不准，AUC 只看排序。"""
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(-(label * np.log(p) + (1 - label) * np.log(1 - p)).mean())


def calibration(label: np.ndarray, prob: np.ndarray, bins: int = 10) -> dict:
    """校准：预测 0.7 的那批样本，真实喜欢率是不是也约 0.7。

    ECE = sum_b (n_b / N) * |平均预测_b - 真实比例_b|，按预测值等宽分 10 桶。
    另报 pred/actual：整体平均预测 / 整体真实正例率，1.0 表示整体不偏。
    """
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        mp, ma = float(prob[m].mean()), float(label[m].mean())
        ece += m.mean() * abs(mp - ma)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_pred": round(mp, 4), "actual": round(ma, 4)})
    return {"ece": round(float(ece), 4), "pred_over_actual": round(float(prob.mean() / max(label.mean(), 1e-9)), 4),
            "bins": rows}
# --8<-- [end:metrics]


@torch.no_grad()
def predict(model, tensors: dict[str, torch.Tensor], idx: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(idx), batch_size):
        b = torch.as_tensor(idx[s: s + batch_size], device=tensors["label"].device)
        batch = {k: v[b] for k, v in tensors.items()}
        out.append(torch.sigmoid(model(batch)).float().cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


# --8<-- [start:group_metrics]
def group_metrics(data, idx: np.ndarray, prob: np.ndarray) -> list[dict]:
    """分组指标：整体 AUC 提升可能只来自某一类用户。每组都报样本数，样本太少的组不要下结论。"""
    a = data.arrays
    label = a["label"][idx]
    groups = {
        "用户历史评分数": pd.cut(a["user_cnt_raw"][idx], [-1, 0, 20, 100, 400, 10**9],
                          labels=["0（冷启动）", "1-20", "21-100", "101-400", ">400"]),
        "电影历史评分数": pd.cut(a["item_cnt_raw"][idx], [-1, 0, 50, 500, 10**9],
                          labels=["0（新片）", "1-50", "51-500", ">500"]),
        "用户在训练集出现": np.where(a["user_id"][idx] == 0, "未出现（OOV）", "出现过"),
    }
    rows = []
    for gname, values in groups.items():
        values = np.asarray(values, dtype=object)
        for v in pd.unique(values):
            m = values == v
            if m.sum() == 0:
                continue
            rows.append({"group": gname, "value": str(v), "n": int(m.sum()), "label_rate": round(float(label[m].mean()), 4),
                         "auc": round(auc(label[m], prob[m]), 4), "logloss": round(logloss(label[m], prob[m]), 4)})
    return rows
# --8<-- [end:group_metrics]


def reference_baselines(data, train_idx: np.ndarray, eval_idx: np.ndarray) -> dict:
    """参照系：不训练模型也能拿到的分数。模型必须明显超过它们才算学到了东西。

    - 常数预测：全部预测训练集正例率。AUC=0.5，LogLoss 是"什么都不学"的上限；
    - 电影平均喜欢率：只用训练集统计每部电影的喜欢率（平滑），相当于"热门好片"排序。
    """
    a = data.arrays
    y_tr, y_ev = a["label"][train_idx], a["label"][eval_idx]
    prior = float(y_tr.mean())
    const = np.full(len(eval_idx), prior)
    df = pd.DataFrame({"m": a["raw_movie_id"][train_idx], "y": y_tr}).groupby("m")["y"].agg(["sum", "size"])
    rate = ((df["sum"] + 20 * prior) / (df["size"] + 20)).to_dict()
    item_score = np.array([rate.get(m, prior) for m in a["raw_movie_id"][eval_idx]])
    return {
        "constant": {"auc": 0.5, "logloss": round(logloss(y_ev, const), 4)},
        "item_mean": {"auc": round(auc(y_ev, item_score), 4), "logloss": round(logloss(y_ev, item_score), 4)},
    }


def error_samples(data, idx: np.ndarray, prob: np.ndarray, k: int = 15) -> list[dict]:
    """最自信的错误：预测很高但实际不喜欢、预测很低但实际喜欢，各取 k 条。"""
    label = data.arrays["label"][idx]
    err = np.abs(prob - label)
    rows = []
    for kind, mask in (("高估（预测喜欢，实际低分）", label == 0), ("低估（预测不喜欢，实际高分）", label == 1)):
        cand = np.nonzero(mask)[0]
        top = cand[np.argsort(-err[cand])[:k]]
        for j in top:
            r = data.raw_row(int(idx[j]))
            r.update({"prob": round(float(prob[j]), 4), "error_type": kind})
            rows.append(r)
    return rows
