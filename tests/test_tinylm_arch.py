"""tinylm 在 L02-L04 新加的部分：结构变体（GQA / MQA / MLA / NoPE / Post-Norm / MoE）、文档 mask、
token 预算与重复、标签右移的故障、fp16 下溢、梯度累积的等价性，以及新参数不改变 L01 的初始化和配置哈希。"""

import numpy as np
import pytest
import torch

from labkit import scenarios
from tinylm.experiment import budget_steps
from tinylm.model import GPT, doc_layout, flops_per_token, kv_cache_stats, param_counts
from tinylm.params import PARAMS
from tinylm.train import lr_at, make_batch, plan_steps

SMALL = {"d_model": 32, "n_layer": 2, "n_head": 2, "d_ff": 64, "seq_len": 16}
MID = {"d_model": 384, "n_layer": 6, "n_head": 6, "d_ff": 1024, "seq_len": 512}


def test_default_structure_init_is_bitwise_unchanged():
    """新参数取默认值时，模块的创建顺序和 L01 一致：同种子的初始化和前向逐位相同（金标值来自 L01 的代码）。"""
    torch.manual_seed(0)
    m = GPT(257, SMALL)
    s = sum(p.double().sum().item() * (i + 1) for i, p in enumerate(m.parameters()))
    x = torch.randint(0, 257, (2, 16), generator=torch.Generator().manual_seed(1))
    assert repr(s) == "1163.155247777907"
    assert repr(m(x).double().sum().item()) == "21.418252611032585"


def test_new_params_are_omitted_from_hash_at_default():
    cfg = PARAMS.resolve({})
    old_keys = {"n_kv_head", "kv_lora_rank", "pos_emb", "ffn", "doc_mask", "data_frac", "epochs", "target_offset",
                "grad_accum", "grad_scaler", "lr_schedule", "flops_budget_frac", "moe_aux_coef"}
    assert old_keys.isdisjoint(PARAMS.hash_view(cfg))
    assert "n_kv_head" in PARAMS.hash_view(PARAMS.resolve({"n_kv_head": 2}))


@pytest.mark.parametrize("extra,kind,elems", [({}, "mha", 768), ({"n_kv_head": 2}, "gqa", 256),
                                              ({"n_kv_head": 1}, "mqa", 128), ({"kv_lora_rank": 96}, "mla", 128)])
def test_kv_cache_elements(extra, kind, elems):
    with torch.device("meta"):
        m = GPT(16384, {**MID, **extra})
    kv = kv_cache_stats(m)
    assert kv["kind"] == kind and kv["elems_per_token_layer"] == elems
    assert kv["bytes_per_token"] == elems * 6 * 2


@pytest.mark.parametrize("extra", [{"n_kv_head": 1}, {"kv_lora_rank": 8, "mla_rope_dim": 4}, {"pos_emb": "none"},
                                   {"pos_emb": "learned"}, {"norm_pos": "post"}, {"ffn": "moe", "moe_experts": 4},
                                   {"doc_mask": True, "eos_id": 3}])
def test_variants_forward_backward(extra):
    torch.manual_seed(0)
    m = GPT(100, {**SMALL, **extra})
    x = torch.randint(0, 100, (2, 16))
    out = m(x)
    assert out.shape == (2, 16, m.V)
    out.float().logsumexp(-1).mean().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad and p.dim() > 1 and "wpe" not in str(p.shape))


def test_gqa_matches_mha_when_kv_heads_are_copied():
    """GQA 等价于"把每组 KV 复制给组内每个查询头"的 MHA：用复制后的权重构造 MHA，输出必须相同。"""
    torch.manual_seed(0)
    cfg = {"d_model": 32, "n_layer": 1, "n_head": 4, "d_ff": 64, "seq_len": 8}
    g = GPT(50, {**cfg, "n_kv_head": 2}).eval()
    h = GPT(50, cfg).eval()
    h.load_state_dict({k: v for k, v in g.state_dict().items() if "qkv" not in k}, strict=False)
    D, C = 8, 32
    wq, wk, wv = g.blocks[0].qkv.weight.split([C, 2 * D, 2 * D])
    rep = lambda w: w.view(2, D, C).repeat_interleave(2, dim=0).reshape(4 * D, C)  # noqa: E731
    h.blocks[0].qkv.weight.data = torch.cat([wq, rep(wk), rep(wv)])
    x = torch.randint(0, 50, (2, 8))
    assert torch.allclose(g(x), h(x), atol=1e-5)


def test_doc_mask_isolates_documents():
    """文档 mask：后一篇文档的输出不受前一篇内容影响，且等于把它单独从位置 0 开始算。"""
    torch.manual_seed(0)
    eos = 1
    m = GPT(50, {**SMALL, "doc_mask": True, "eos_id": eos}).eval()
    a = torch.tensor([[eos, 5, 6, 7, eos, 8, 9, 10]])
    b = torch.tensor([[eos, 20, 21, 22, eos, 8, 9, 10]])
    alone = torch.tensor([[eos, 8, 9, 10]])
    ya, yb, y0 = m(a), m(b), m(alone)
    assert torch.allclose(ya[0, 4:], yb[0, 4:], atol=1e-5)
    assert torch.allclose(ya[0, 4:], y0[0], atol=1e-5)
    doc, pos, mask = doc_layout(a, eos)
    assert doc.tolist() == [[1, 1, 1, 1, 2, 2, 2, 2]] and pos.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3]]
    assert mask.shape == (1, 1, 8, 8) and not mask[0, 0, 5, 2]


def test_moe_params_flops_and_balance_loss():
    with torch.device("meta"):
        d, e = GPT(16384, MID), GPT(16384, {**MID, "ffn": "moe"})
    pd, pe = param_counts(d), param_counts(e)
    assert 2.1 < pe["total"] / pd["total"] < 2.4
    assert abs(pe["active"] - pd["total"]) / pd["total"] < 0.01  # top-2 × 512 = 1024：激活参数和 Dense 相同
    fd, fe = flops_per_token(d, MID), flops_per_token(e, MID)
    assert 1.0 <= fe["train_flops_per_token"] / fd["train_flops_per_token"] < 1.01
    torch.manual_seed(0)
    m = GPT(100, {**SMALL, "ffn": "moe", "moe_experts": 4})
    m(torch.randint(0, 100, (4, 16)))
    assert 0.9 < float(m.aux_loss().detach()) < 4.0  # 完全均衡 = 1，坍缩到一个专家 → E
    st = m.moe_stats()
    assert abs(sum(st["load"][0]) - 1) < 1e-5 and st["max_over_mean"] >= 1


def test_make_batch_shift_and_fault():
    stream = np.arange(100, dtype=np.int32)
    x, y = make_batch(stream, [0, 10], 8)
    assert (y[:, :-1] == x[:, 1:]).all() and (y[:, -1] == x[:, -1] + 1).all()
    x0, y0 = make_batch(stream, [0, 10], 8, offset=0)
    assert (x0 == y0).all()


def test_plan_steps_prefix_epochs_and_budget():
    cfg = {"batch_size": 4, "seed": 0, "max_steps": 0, "data_frac": 0.25, "epochs": 4}
    p = plan_steps(400, cfg)
    assert p["n_used"] == 100 and p["total"] == 100 and set(p["order"]) == set(range(100))
    assert len(p["order"]) == 400 and not (p["order"][:100] == p["order"][100:200]).all()  # 每遍重新打乱
    assert plan_steps(400, {**cfg, "data_frac": 1.0, "epochs": 1}, budget_steps=30)["total"] == 30


def test_budget_steps_equal_compute():
    """等算力：小模型每 token 便宜，步数更多；总 FLOPs = 比例 × 参考模型过一遍训练流的 FLOPs。"""
    cfg = PARAMS.resolve({"flops_budget_frac": 0.5})
    n_tok = 10_000_000
    with torch.device("meta"):
        ref = GPT(16384, cfg)
    fpt_ref = flops_per_token(ref, cfg)["train_flops_per_token"]
    steps = budget_steps(cfg, 16384, n_tok, fpt_ref)
    assert abs(steps * cfg["batch_size"] * cfg["seq_len"] - 0.5 * n_tok) / n_tok < 0.01
    small = PARAMS.resolve({"flops_budget_frac": 0.5, "d_model": 320, "n_layer": 5, "n_head": 5, "d_ff": 864})
    with torch.device("meta"):
        fpt_s = flops_per_token(GPT(16384, small), small)["train_flops_per_token"]
    assert budget_steps(small, 16384, n_tok, fpt_s) > 1.3 * steps


def test_constant_schedule_and_no_warmup():
    cfg = {"lr": 1.0, "warmup_frac": 0.1, "lr_schedule": "constant"}
    assert lr_at(50, 100, cfg) == 1.0 and lr_at(99, 100, cfg) == 1.0
    assert lr_at(0, 100, {"lr": 1.0, "warmup_frac": 0.0}) == 1.0


def test_fp16_cross_entropy_gradient_underflows_without_scaling():
    """交叉熵对 logits 的梯度 (p − y) / N：N 很大时在 fp16 里大部分变成 0；乘上 2^16 再算就保留下来。"""
    torch.manual_seed(0)
    N, V = 4096, 4096
    logits = 3 * torch.randn(N, V)  # 训练中的模型分布是尖的：多数词的概率远小于 1 / V
    y = torch.randint(0, V, (N,))
    g = (logits.softmax(-1) - torch.nn.functional.one_hot(y, V).float()) / N
    assert (g.half() == 0).float().mean() > 0.5
    assert (((g * 65536).half()) == 0).float().mean() < 0.05


def test_scenarios_l02_l04_expand_and_l01_baseline_shared():
    base = {c: next(s for s in scenarios.expand(c, "h20x8", seeds=[0]) if s.scenario == "baseline") for c in
            ("L01", "L02", "L03", "L04")}
    assert len({b.config_hash for b in base.values()}) == 1, "L02-L04 的基线必须和 L01 同配置，H20 上只跑一次"
    for c in ("L02", "L03", "L04"):
        assert len({s.scenario for s in scenarios.expand(c, "h20x8")}) == 10
