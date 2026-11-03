"""LLM 项目：合成数据、评测规则、监督 mask、loss 对齐。"""

import json

import pytest
import torch
import torch.nn.functional as F

from labkit import scenarios
from adextract import synth
from adextract.data import IGNORE, encode_example, load_evalset, rows_hash
from adextract.schema import norm, score_one, to_json
from adextract.train import causal_lm_loss

CFG = scenarios.param_space("adextract").defaults()
EVAL_TEXTS = {r["text"] for s in ("dev", "test") for r in load_evalset(s)}


def test_exclusion_ratio_zero_is_identical():
    a = synth.generate(400, 0.15, 0, exclude=EVAL_TEXTS)
    b = synth.generate(400, 0.15, 0, exclude=EVAL_TEXTS, exclusion_ratio=0.0)
    assert rows_hash(a) == rows_hash(b)


def test_exclusion_ratio_replaces_only_k():
    a = synth.generate(400, 0.15, 0, exclude=EVAL_TEXTS)
    b = synth.generate(400, 0.15, 0, exclude=EVAL_TEXTS, exclusion_ratio=0.15)
    assert len(b) == 400
    diff = [i for i in range(400) if a[i]["text"] != b[i]["text"]]
    assert len(diff) == 60 and all(b[i]["kind"] == "exclusion" for i in diff)


@pytest.mark.parametrize("ratio", [0.0, 0.15])
def test_no_eval_leak_and_labels_grounded(ratio):
    rows = synth.generate(400, 0.4, 3, exclude=EVAL_TEXTS, exclusion_ratio=ratio) + synth.synth_valid()
    assert not EVAL_TEXTS & {r["text"] for r in rows}
    for r in rows + load_evalset("dev") + load_evalset("test"):
        g, src = r["gold"], norm(r["text"])
        vals = ([g["category"]] if g["category"] else []) + g["selling_points"]
        vals += [x for p in g["promotions"] for x in (p["offer"], p.get("condition")) if x]
        assert all(norm(v) in src for v in vals), r


def test_score_one_rules():
    text = "跑鞋轻量透气，第二件半价，指定款参加"
    gold = {"category": "跑鞋", "selling_points": ["轻量", "透气"],
            "promotions": [{"offer": "第二件半价", "condition": "指定款"}]}
    assert score_one(text, to_json(gold), gold)["errors"] == []
    no_cond = {**gold, "promotions": [{"offer": "第二件半价", "condition": None}]}
    assert "漏条件" in score_one(text, to_json(no_cond), gold)["errors"]
    rewritten = {**gold, "promotions": [{"offer": "第二件5折", "condition": "指定款"}]}
    assert "无依据内容" in score_one(text, to_json(rewritten), gold)["errors"]
    assert score_one(text, "不是 JSON", gold)["errors"] == ["格式错误"]


def test_causal_lm_loss_alignment():
    torch.manual_seed(0)
    B, T, V = 2, 7, 11
    logits = torch.randn(B, T, V)
    labels = torch.full((B, T), IGNORE)
    labels[0, 4:] = torch.randint(0, V, (3,))
    labels[1, 5:] = torch.randint(0, V, (2,))
    loss, n = causal_lm_loss(logits, labels)
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=IGNORE)
    assert int(n) == 5 and torch.allclose(loss, ref)


def test_zero_supervision_is_silent():
    """监督 token 为 0 时 loss = 0 而不是 nan：不会报错，这正是 L1 故障难以发现的原因。"""
    loss, n = causal_lm_loss(torch.randn(1, 5, 7, requires_grad=True), torch.full((1, 5), IGNORE))
    assert int(n) == 0 and loss.item() == 0.0 and not loss.isnan()


@pytest.fixture(scope="module")
def tok():
    from adextract.modeling import load_tokenizer, resolve_model_path

    if resolve_model_path(CFG["model"], must_exist=False) is None:
        pytest.skip("本机没有 Qwen 模型缓存")
    return load_tokenizer(CFG["model"])


def test_supervision_mask_is_assistant_only(tok):
    r = synth.synth_valid()[0]
    ex = encode_example(tok, r["text"], r["gold"], CFG)
    assert all(x == IGNORE for x in ex["labels"][: ex["n_prompt"]])
    sup = tok.decode([t for t in ex["labels"] if t != IGNORE])
    assert sup == to_json(r["gold"]) + "<|im_end|>\n"
    assert json.loads(sup.split("<|im_end|>")[0]) == json.loads(to_json(r["gold"]))


def test_truncation_fault_and_drop_long(tok):
    r = synth.synth_valid()[0]
    right = encode_example(tok, r["text"], r["gold"], {**CFG, "max_len": 256, "truncation": "right"})
    assert right["truncated"] and right["n_supervised"] == 0
    assert encode_example(tok, r["text"], r["gold"], {**CFG, "max_len": 256, "truncation": "drop_long"}) is None
