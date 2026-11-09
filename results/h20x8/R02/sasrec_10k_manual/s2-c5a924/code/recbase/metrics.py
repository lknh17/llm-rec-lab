"""评估：AUC（并列取平均）、GAUC、LogLoss / NE、按用户 bootstrap、分组指标、不训练模型的参照线。全部 numpy。"""

from __future__ import annotations

import numpy as np

# 分组：(键, 维度, 取值)。键不含"."，模板里写 groups.valid.u_unseen.auc
GROUPS = (
    ("u_unseen", "用户在训练集", "没出现（OOV）"), ("u_seen", "用户在训练集", "出现过"),
    ("u_new", "用户此前评分数", "0"), ("u_low", "用户此前评分数", "1-19"), ("u_mid", "用户此前评分数", "20-99"),
    ("u_high", "用户此前评分数", "100-399"), ("u_heavy", "用户此前评分数", "400+"),
    ("i_tail", "电影此前评分数", "<50"), ("i_mid", "电影此前评分数", "50-499"), ("i_head", "电影此前评分数", "500+"),
    ("fut_user", "训练集有该用户更晚的行为", "有"), ("no_fut_user", "训练集有该用户更晚的行为", "没有"),
)


def _r(v: float | None, nd: int = 5) -> float | None:
    return None if v is None or v != v else round(float(v), nd)


# --8<-- [start:auc_gauc]
def _sorted_ties(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按分数升序排好的下标 o 和每个位置的并列组编号 g（分数相同 = 同一组）。"""
    o = np.argsort(score, kind="mergesort")
    s = score[o]
    return o, np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1


def _auc_presorted(o: np.ndarray, g: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """AUC = Σ_正例 w ×（排在它前面的负例权重 + 并列负例权重 / 2）÷（正例总权重 × 负例总权重）。"""
    yy, ww = y[o], w[o]
    pos_g = np.bincount(g, weights=ww * yy)
    neg_g = np.bincount(g, weights=ww * (1 - yy))
    P, N = pos_g.sum(), neg_g.sum()
    if P == 0 or N == 0:
        return float("nan")
    neg_before = np.cumsum(neg_g) - neg_g
    return float((pos_g * (neg_before + 0.5 * neg_g)).sum() / (P * N))


def auc(label: np.ndarray, score: np.ndarray, w: np.ndarray | None = None) -> float:
    """随机抽一个正例和一个负例，正例分数更高的概率（并列算一半）。只有一类样本时无定义，返回 nan。"""
    y = np.asarray(label, np.float64)
    o, g = _sorted_ties(np.asarray(score))
    return _auc_presorted(o, g, y, np.ones(len(y)) if w is None else np.asarray(w, np.float64))


def gauc(label: np.ndarray, score: np.ndarray, user: np.ndarray) -> tuple[float, float]:
    """按用户分别算 AUC，再按用户的行数加权平均。只有正例或只有负例的用户没有定义，排除在外。
    返回 (gauc, 覆盖率 = 参与计算的行数 / 总行数)。"""
    n = len(label)
    if n == 0:
        return float("nan"), 0.0
    o = np.lexsort((score, user))                                        # 先按用户、再按分数
    u, s, y = user[o], score[o], np.asarray(label, np.float64)[o]
    new_u = np.r_[True, u[1:] != u[:-1]]
    u_start = np.maximum.accumulate(np.where(new_u, np.arange(n), 0))   # 每行所在用户的起始位置
    new_t = new_u | np.r_[True, s[1:] != s[:-1]]
    t_start = np.flatnonzero(new_t)
    t_size = np.diff(np.r_[t_start, n])
    tg = np.cumsum(new_t) - 1
    rank = (t_start[tg] - u_start) + (t_size[tg] + 1) / 2               # 用户内的秩（并列取平均），从 1 开始
    uid = np.cumsum(new_u) - 1
    n_all = np.bincount(uid)
    n_pos = np.bincount(uid, weights=y)
    n_neg = n_all - n_pos
    r_pos = np.bincount(uid, weights=rank * y)
    ok = (n_pos > 0) & (n_neg > 0)
    if not ok.any():
        return float("nan"), 0.0
    per_user = (r_pos[ok] - n_pos[ok] * (n_pos[ok] + 1) / 2) / (n_pos[ok] * n_neg[ok])
    return float((n_all[ok] * per_user).sum() / n_all[ok].sum()), float(n_all[ok].sum() / n)
# --8<-- [end:auc_gauc]


def logloss(label: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(-(label * np.log(p) + (1 - label) * np.log(1 - p)).mean())


def ne(label: np.ndarray, prob: np.ndarray) -> float:
    """Normalized Entropy = LogLoss ÷ "只预测这批样本平均正例率"的 LogLoss。正例率不同的两批数据也能比；< 1 才算学到东西。"""
    p = float(np.mean(label))
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p)) if 0 < p < 1 else float("nan")
    return logloss(label, prob) / h


def boot_std(label: np.ndarray, score: np.ndarray, user: np.ndarray, times: int, seed: int = 0) -> float | None:
    """按用户有放回抽样 times 次（同一用户的行一起进出），AUC 的标准差 = 换一批用户时 AUC 会抖多少。"""
    if times <= 1 or len(label) == 0:
        return None
    rng = np.random.default_rng(seed)
    _, inv = np.unique(user, return_inverse=True)
    nu = inv.max() + 1
    y = np.asarray(label, np.float64)
    o, g = _sorted_ties(np.asarray(score))
    vals = [_auc_presorted(o, g, y, np.bincount(rng.integers(0, nu, nu), minlength=nu)[inv].astype(np.float64))
            for _ in range(times)]
    return float(np.nanstd(vals))


def block(label: np.ndarray, prob: np.ndarray, user: np.ndarray, boot: int = 0) -> dict:
    """一批样本的全部指标。"""
    if len(label) == 0:
        return {"n": 0, "auc": None, "gauc": None, "logloss": None, "ne": None, "label_rate": None,
                "gauc_row_cov": None, "auc_boot_std": None}
    g, cov = gauc(label, prob, user)
    return {"n": int(len(label)), "label_rate": _r(np.mean(label), 4), "auc": _r(auc(label, prob)),
            "gauc": _r(g), "gauc_row_cov": _r(cov, 4), "logloss": _r(logloss(label, prob)),
            "ne": _r(ne(label, prob)), "auc_boot_std": _r(boot_std(label, prob, user, boot))}


# --8<-- [start:groups]
def group_masks(uid: np.ndarray, u_prev: np.ndarray, i_prev: np.ndarray, fut: np.ndarray) -> dict[str, np.ndarray]:
    """uid = 训练集词表编码（0 = 没见过）；u_prev / i_prev = 样本时刻之前的评分数（时点正确，不随统计口径变）；
    fut = 训练集里有没有这个用户更晚的行为。"""
    return {"u_unseen": uid == 0, "u_seen": uid != 0,
            "u_new": u_prev == 0, "u_low": (u_prev >= 1) & (u_prev < 20), "u_mid": (u_prev >= 20) & (u_prev < 100),
            "u_high": (u_prev >= 100) & (u_prev < 400), "u_heavy": u_prev >= 400,
            "i_tail": i_prev < 50, "i_mid": (i_prev >= 50) & (i_prev < 500), "i_head": i_prev >= 500,
            "fut_user": fut, "no_fut_user": ~fut}


def groups(label: np.ndarray, prob: np.ndarray, user: np.ndarray, masks: dict[str, np.ndarray]) -> tuple[dict, list]:
    """整体 AUC 的变化可能只来自某一类用户。每组都报行数：行数少的组别下结论。返回 dict（模板 / 断言用）和列表（网页表格用）。"""
    by_key, rows = {}, []
    for key, dim, val in GROUPS:
        m = masks[key]
        n = int(m.sum())
        if n:
            g, _ = gauc(label[m], prob[m], user[m])
            d = {"n": n, "label_rate": _r(label[m].mean(), 4), "auc": _r(auc(label[m], prob[m])), "gauc": _r(g)}
        else:
            d = {"n": 0, "label_rate": None, "auc": None, "gauc": None}
        by_key[key] = d
        rows.append({"key": key, "group": dim, "value": val, **d})
    return by_key, rows
# --8<-- [end:groups]


# --8<-- [start:reference]
def smoothed_mean(key_train: np.ndarray, y_train: np.ndarray, key_eval: np.ndarray, alpha: float = 20.0) -> np.ndarray:
    """只用训练集：每个 key 的平滑喜欢率 (正例 + α × 先验) / (次数 + α)；训练集没见过的 key 给先验。"""
    prior = float(y_train.mean())
    size = int(max(key_train.max(initial=0), key_eval.max(initial=0))) + 1
    n = np.bincount(key_train, minlength=size)
    p = np.bincount(key_train, weights=y_train, minlength=size)
    return ((p + alpha * prior) / (n + alpha))[key_eval]


def reference(y_tr, user_tr, movie_tr, y_ev, user_ev, movie_ev) -> dict:
    """不训练模型的参照线：按"电影平均喜欢率"排序（热门好片）、按"用户平均喜欢率"排序（爱打高分的人）。"""
    out = {}
    for name, ktr, kev in (("item_mean", movie_tr, movie_ev), ("user_mean", user_tr, user_ev)):
        s = smoothed_mean(ktr, y_tr, kev)
        g, _ = gauc(y_ev, s, user_ev)
        out[name] = {"auc": _r(auc(y_ev, s)), "gauc": _r(g), "logloss": _r(logloss(y_ev, s))}
    return out
# --8<-- [end:reference]
