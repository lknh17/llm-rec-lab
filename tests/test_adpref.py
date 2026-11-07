"""偏好优化项目（adpref）：奖励、规则偏好对、DPO / GRPO loss 的数值性质，以及依赖项目的代码哈希。"""

import random

import pytest
import torch

from adextract.schema import to_json
from adextract.synth import make_ad
from adpref.optim import dpo_loss, grpo_step_loss
from adpref.pairs import build_pairs, rule_rejected
from adpref.reward import reward
from labkit import snapshot

GOLD = {"category": "跑鞋", "selling_points": ["轻量", "透气"],
        "promotions": [{"offer": "第二件半价", "condition": "指定款"}]}
TEXT = "跑鞋轻量、透气，指定款第二件半价"


def test_reward_full_marks_and_format_error():
    assert reward(TEXT, to_json(GOLD), GOLD)[0] == 1.0
    assert reward(TEXT, "不是 JSON", GOLD)[0] == 0.0


def test_reward_penalizes_dropped_condition_only_in_field_mode():
    bad = {**GOLD, "promotions": [{"offer": "第二件半价", "condition": None}]}
    field, _ = reward(TEXT, to_json(bad), GOLD, "field")
    recall, _ = reward(TEXT, to_json(bad), GOLD, "recall_only")
    assert field == pytest.approx(0.7)       # 条件占 0.3
    assert recall == 1.0                     # 只看召回：丢条件不扣分


def test_reward_penalizes_ungrounded_value():
    bad = {**GOLD, "selling_points": ["轻量", "透气", "缓震"]}
    assert reward(TEXT, to_json(bad), GOLD)[0] < 1.0
    assert reward(TEXT, to_json(bad), GOLD, "recall_only")[0] == 1.0


def test_rule_rejected_drops_condition_or_adds_ungrounded():
    rng = random.Random(0)
    bad, kind = rule_rejected({"text": TEXT, "gold": GOLD}, rng)
    assert kind == "drop_condition" and bad["promotions"][0]["condition"] is None
    assert GOLD["promotions"][0]["condition"] == "指定款"          # 不改原标注
    no_cond = {"category": "跑鞋", "selling_points": ["轻量"], "promotions": []}
    bad, kind = rule_rejected({"text": "跑鞋轻量", "gold": no_cond}, rng)
    assert kind == "add_ungrounded" and len(bad["selling_points"]) == 2


def test_build_rule_pairs_and_flip():
    rng = random.Random(1)
    rows = [dict(make_ad(rng, k), id=f"r{i}") for i, k in enumerate(["single_promo", "no_promo", "exclusion"])]
    cfg = {"pref_pairs": "rule", "pref_flip": False, "data_seed": 0, "seed": 0}
    pairs, stats = build_pairs(rows, cfg)
    assert stats["pairs"] == 3 and stats["pair_yield"] == 1.0
    assert all(p["chosen"] == to_json(r["gold"]) for p, r in zip(pairs, rows))
    flipped, fstats = build_pairs(rows, dict(cfg, pref_flip=True))
    assert fstats["flipped"] and all(f["rejected"] == p["chosen"] for f, p in zip(flipped, pairs))


def test_dpo_loss_is_log2_at_reference_and_falls_with_margin():
    z = torch.zeros(4)
    loss, _, _ = dpo_loss(z, z, z, z, beta=0.1)
    assert loss.item() == pytest.approx(0.6931, abs=1e-4)          # 策略 = 参考模型时 loss = log 2
    better, rc, rr = dpo_loss(z, z - 10, z, z, beta=0.1)            # rejected 被压低
    assert better.item() < loss.item() and (rc > rr).all()


def test_grpo_zero_advantage_gives_zero_policy_gradient():
    lp = torch.randn(2, 5, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    loss, kl = grpo_step_loss(lp, lp.detach(), mask, torch.zeros(2), kl_coef=0.04)
    loss.backward()
    assert kl.item() == pytest.approx(0.0, abs=1e-7)
    assert torch.allclose(lp.grad, torch.zeros_like(lp.grad))     # 组内奖励全相同：没有学习信号


def test_dependency_changes_code_hash(tmp_path):
    (tmp_path / "dep").mkdir()
    (tmp_path / "dep" / "a.py").write_text("x = 1\n")
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "b.py").write_text("y = 2\n")
    alone = snapshot.tree_manifest(tmp_path / "proj")["code_hash"]
    (tmp_path / "proj" / "deps.txt").write_text("dep\n")
    h1 = snapshot.tree_manifest(tmp_path / "proj")["code_hash"]
    (tmp_path / "dep" / "a.py").write_text("x = 3\n")
    h2 = snapshot.tree_manifest(tmp_path / "proj")["code_hash"]
    assert alone != h1 != h2                                        # 依赖的代码改了，结果同样过期


def test_project_without_deps_keeps_hash_algorithm():
    """adextract 没有 deps.txt：哈希和引入依赖机制之前一致（H20 上已有的 L05/L06 结果不过期）。"""
    m = snapshot.tree_manifest(snapshot.project_dir("adextract"))
    assert "deps" not in m
