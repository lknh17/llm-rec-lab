"""可验证奖励：抽取任务的标准答案是确定的，奖励完全由规则算出，不需要奖励模型。

两种口径（GRPO 用；onpolicy 偏好对用 field 挑最差的一条）：
  - field：格式错误 0 分；否则 0.2·品类 + 0.25·卖点 F1 + 0.25·优惠 F1 + 0.3·条件保留，
           原文找不到的值按比例扣 0.5，多写条件扣 0.1，截到 [0, 1]。全部字段正确 = 1。
  - recall_only：格式合法即可，0.5·卖点召回 + 0.5·优惠召回。不看条件、不罚多写——
           奖励没覆盖的维度，优化时不会变好，甚至会被牺牲。
"""

from __future__ import annotations

from adextract.schema import parse_output, score_one


def _f1(tp: int, n_pred: int, n_gold: int) -> float:
    if n_pred == 0 and n_gold == 0:
        return 1.0
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _n_pred_promos(output: str) -> int:
    parsed, _ = parse_output(output)
    return len(parsed["promotions"]) if parsed else 0


# --8<-- [start:reward]
def reward(text: str, output: str, gold: dict, kind: str = "field") -> tuple[float, dict]:
    sc = score_one(text, output, gold)
    if not sc["format_ok"]:
        return 0.0, sc
    if kind == "recall_only":
        pr = sc["point_tp"] / sc["n_gold_points"] if sc["n_gold_points"] else 1.0
        rr = sc["promo_tp"] / sc["n_gold_promos"] if sc["n_gold_promos"] else 1.0
        return round(0.5 * pr + 0.5 * rr, 4), sc
    point_f1 = _f1(sc["point_tp"], sc["n_pred_points"], sc["n_gold_points"])
    promo_f1 = _f1(sc["promo_tp"], _n_pred_promos(output), sc["n_gold_promos"])
    cond = sc["cond_kept"] / sc["n_gold_cond"] if sc["n_gold_cond"] else 1.0
    r = 0.2 * sc["category_ok"] + 0.25 * point_f1 + 0.25 * promo_f1 + 0.3 * cond
    r -= 0.5 * (sc["n_ungrounded"] / sc["n_pred_values"] if sc["n_pred_values"] else 0.0)
    r -= 0.1 if "多条件" in sc["errors"] else 0.0
    return round(min(max(r, 0.0), 1.0), 4), sc
# --8<-- [end:reward]


def all_ok(sc: dict) -> bool:
    return bool(sc["format_ok"] and not sc["errors"])
