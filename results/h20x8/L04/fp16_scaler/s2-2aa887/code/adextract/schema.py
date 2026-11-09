"""输出格式、解析与评测口径。

抽取式任务：所有字段值都应该摘自广告原文。这让评测可以完全基于规则，不依赖模型评审：
  - 格式合法率   = 能解析成符合 schema 的 JSON 的样本数 / 评估样本数
  - 品类准确率   = 品类匹配的样本数 / 评估样本数（格式错误计为错）
  - 卖点 P/R/F1  = 按短语一对一匹配的 micro 统计
  - 优惠召回     = 被找回的标准优惠数 / 标准优惠总数
  - 条件保留率   = 条件也被正确保留的优惠数 / 带条件的标准优惠总数  （1 - 它 = 关键条件遗漏率）
  - 无依据率     = 原文中找不到的预测值个数 / 预测值总数（只统计格式合法的输出）
匹配规则：归一化（去空白和标点、转小写）后，一方包含另一方即算匹配——抽取边界多一两个字不算错。
适用范围：只衡量"抽得对不对"，不衡量措辞好坏；规则匹配对同义改写（例如把"指定款"写成"部分商品"）会判错，
这类情况在错例里人工抽查。
"""

from __future__ import annotations

import json
import re

PUNCT = re.compile(r"[\s，。！？、；：,.!?;:（）()\[\]【】「」“”\"'《》<>·~～\-—_]+")


def norm(s: str | None) -> str:
    return PUNCT.sub("", (s or "")).lower()


def loose_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return pred is None and gold is None
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return p == g
    return p == g or (len(p) >= 2 and p in g) or (len(g) >= 2 and g in p)


def to_json(target: dict) -> str:
    """训练目标的序列化方式：紧凑 JSON，中文不转义。推理端也按这个格式解析。"""
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def parse_output(text: str) -> tuple[dict | None, str | None]:
    """严格解析：允许首尾空白和 ```json 围栏，其余多余文字一律判格式错误。"""
    t = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, flags=re.S)
    if m:
        t = m.group(1)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败：{e.msg}"
    if not isinstance(obj, dict):
        return None, "不是 JSON 对象"
    missing = {"category", "selling_points", "promotions"} - set(obj)
    if missing:
        return None, f"缺字段 {sorted(missing)}"
    if obj["category"] is not None and not isinstance(obj["category"], str):
        return None, "category 不是字符串或 null"
    if not isinstance(obj["selling_points"], list) or not all(isinstance(x, str) for x in obj["selling_points"]):
        return None, "selling_points 不是字符串列表"
    if not isinstance(obj["promotions"], list):
        return None, "promotions 不是列表"
    for p in obj["promotions"]:
        if not isinstance(p, dict) or "offer" not in p or not isinstance(p.get("offer"), str):
            return None, "promotions 元素缺 offer"
        if p.get("condition") is not None and not isinstance(p["condition"], str):
            return None, "condition 不是字符串或 null"
    return obj, None


def _match_lists(pred: list, gold: list, key=lambda x: x) -> list[tuple[int, int]]:
    """贪心一对一匹配，返回 (pred_idx, gold_idx)。"""
    used, pairs = set(), []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if pi not in used and loose_match(key(p), key(g)):
                used.add(pi)
                pairs.append((pi, gi))
                break
    return pairs


# --8<-- [start:score_one]
def score_one(text: str, pred_raw: str, gold: dict) -> dict:
    """单条样本的评测明细和错误类型。"""
    pred, err = parse_output(pred_raw)
    gold_cond = [g for g in gold["promotions"] if g.get("condition")]
    r = {"format_ok": pred is not None, "format_error": err, "errors": [],
         "n_gold_points": len(gold["selling_points"]), "n_gold_promos": len(gold["promotions"]),
         "n_gold_cond": len(gold_cond), "point_tp": 0, "n_pred_points": 0, "promo_tp": 0, "cond_kept": 0,
         "n_pred_values": 0, "n_ungrounded": 0, "ungrounded_values": []}
    if pred is None:
        r["errors"].append("格式错误")
        r["category_ok"] = False
        return r
    r["category_ok"] = loose_match(pred["category"], gold["category"])
    if not r["category_ok"]:
        r["errors"].append("品类错误" if gold["category"] else "无品类却编造品类")
    pts = _match_lists(pred["selling_points"], gold["selling_points"])
    r["point_tp"], r["n_pred_points"] = len(pts), len(pred["selling_points"])
    if len(pts) < len(gold["selling_points"]):
        r["errors"].append("漏卖点")
    if len(pts) < len(pred["selling_points"]):
        r["errors"].append("多卖点")
    pairs = _match_lists(pred["promotions"], gold["promotions"], key=lambda x: x.get("offer"))
    r["promo_tp"] = len(pairs)
    if len(pairs) < len(gold["promotions"]):
        r["errors"].append("漏优惠")
    if len(pairs) < len(pred["promotions"]):
        r["errors"].append("多优惠（原文无此优惠或把否定句当优惠）")
    for pi, gi in pairs:
        g, p = gold["promotions"][gi], pred["promotions"][pi]
        if g.get("condition"):
            if loose_match(p.get("condition"), g["condition"]):
                r["cond_kept"] += 1
            else:
                r["errors"].append("漏条件")  # 例如把"指定款第二件半价"写成无条件的"第二件半价"
        elif p.get("condition"):
            r["errors"].append("多条件")
    missed_cond_promos = [g for g in gold_cond if not any(gold["promotions"][gi] is g for _, gi in pairs)]
    if missed_cond_promos and "漏条件" not in r["errors"]:
        r["errors"].append("漏条件")
    values = ([pred["category"]] if pred["category"] else []) + list(pred["selling_points"])
    for p in pred["promotions"]:
        values += [p.get("offer")] + ([p["condition"]] if p.get("condition") else [])
    src = norm(text)
    for v in values:
        r["n_pred_values"] += 1
        if norm(v) and norm(v) not in src:
            r["n_ungrounded"] += 1
            r["ungrounded_values"].append(v)
    if r["n_ungrounded"]:
        r["errors"].append("无依据内容")
    return r
# --8<-- [end:score_one]


# --8<-- [start:aggregate]
def aggregate(rows: list[dict]) -> dict:
    """把单条明细汇总成指标。每个指标的分母写在 *_den 里，网页一并展示。"""
    n = len(rows)
    s = lambda k: sum(r[k] for r in rows)  # noqa: E731
    gold_pts, pred_pts, tp = s("n_gold_points"), s("n_pred_points"), s("point_tp")
    prec = tp / pred_pts if pred_pts else 0.0
    rec = tp / gold_pts if gold_pts else 0.0
    error_counts: dict[str, int] = {}
    for r in rows:
        for e in set(r["errors"]):
            error_counts[e] = error_counts.get(e, 0) + 1
    return {
        "n": n,
        "format_ok": round(s("format_ok") / n, 4) if n else 0, "format_ok_den": n,
        "category_acc": round(s("category_ok") / n, 4) if n else 0, "category_acc_den": n,
        "point_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "point_precision": round(prec, 4), "point_recall": round(rec, 4), "point_den_gold": gold_pts,
        "promo_recall": round(s("promo_tp") / s("n_gold_promos"), 4) if s("n_gold_promos") else None,
        "promo_recall_den": s("n_gold_promos"),
        "condition_keep": round(s("cond_kept") / s("n_gold_cond"), 4) if s("n_gold_cond") else None,
        "condition_keep_den": s("n_gold_cond"),
        "ungrounded_rate": round(s("n_ungrounded") / s("n_pred_values"), 4) if s("n_pred_values") else 0.0,
        "ungrounded_den": s("n_pred_values"),
        "exact_all_ok": round(sum(1 for r in rows if r["format_ok"] and not r["errors"]) / n, 4) if n else 0,
        "error_counts": dict(sorted(error_counts.items(), key=lambda kv: -kv[1])),
    }
# --8<-- [end:aggregate]
