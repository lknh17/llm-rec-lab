"""recops（R05 / R06）：Zipf id、DLRM 形状、两种特征处理等价、优化器状态、attention 参照与分块镜像（含故障）、计时与 roofline、端到端。"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from labkit import scenarios
from recops import attn as A
from recops import bench as BN
from recops import embops as E
from recops.params import PARAMS


def test_zipf_ids_skewed_and_in_range():
    u = E.ids_for(50000, 10000, "uniform", 1.05, 0, "cpu")
    z = E.ids_for(50000, 10000, "zipf", 1.05, 0, "cpu")
    assert int(z.min()) >= 0 and int(z.max()) < 10000
    assert torch.unique(z).numel() < 0.5 * torch.unique(u).numel()


def test_feature_processing_python_equals_vectorized():
    raw = E.RawShard(500, 5, 7, seed=0)
    idx = np.array([3, 17, 499, 0])
    a, b = E.process_vectorized(raw, idx, 1000), E.process_python(raw, idx, 1000)
    for k in a:
        assert np.allclose(a[k], b[k])


def test_dlrm_shapes_and_sparse_grads():
    m = E.DLRM(rows=100, n_tables=4, d=8)
    B = 6
    b = {"dense": torch.rand(B, 13), "cat": torch.randint(0, 100, (B, 4)), "mh": torch.randint(0, 100, (B, 2, 5)),
         "label": torch.rand(B)}
    y = m(b)
    assert y.shape == (B,)
    y.sum().backward()
    assert m.tables[0].weight.grad.is_sparse
    f = 4 + 3
    assert m.top[0].in_features == 8 + f * (f - 1) // 2


def test_optim_bench_dense_state_is_two_tables():
    cfg = PARAMS.resolve({"emb_batch": 64, "iters": 2, "warmup": 1, "min_time_ms": 0})
    r = E.optim_bench(cfg, torch.device("cpu"), BN.Timer(torch.device("cpu"), cfg), rows=5000, dim=8)
    assert r["adam"]["state_gb"] == pytest.approx(2 * r["adam"]["table_gb"], rel=0.01)   # exp_avg + exp_avg_sq
    assert r["sparse_adam"]["state_gb"] == pytest.approx(2 * r["sparse_adam"]["table_gb"], rel=0.01)


# ---------------------------------------------------------------- attention
def qkv(B=1, H=2, L=50, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, L, D, generator=g) for _ in range(3)]


def test_naive_softmax_matches_sdpa():
    q, k, v = qkv()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(A.naive(q, k, v, "softmax", None), ref, atol=1e-5)


def test_naive_pointwise_loop():
    q, k, v = qkv(L=9)
    rab = torch.randn(4) * 0.1
    got = A.naive(q, k, v, "pointwise", rab)
    D, L = 16, 9
    for i in (0, 4, 8):
        acc = torch.zeros(D)
        for j in range(i + 1):
            s = (q[0, 0, i] @ k[0, 0, j]) / math.sqrt(D) + rab[min(i - j, 3)]
            acc += torch.nn.functional.silu(s) / L * v[0, 0, j]
        assert torch.allclose(got[0, 0, i], acc, atol=1e-5)


@pytest.mark.parametrize("kind", ["softmax", "pointwise"])
@pytest.mark.parametrize("L", [64, 50, 130])
def test_block_mirror_matches_naive(kind, L):
    q, k, v = qkv(L=L)
    rab = torch.randn(32) * 0.1
    ref = A.naive(q, k, v, kind, rab)
    got = A.block_mirror(q, k, v, kind, rab, block_m=32, block_n=16)
    assert torch.allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("kind", ["softmax", "pointwise"])
def test_block_mirror_fault_leaks_future(kind):
    q, k, v = qkv(L=64)
    rab = torch.randn(32) * 0.1
    ref = A.naive(q, k, v, kind, rab)
    bad = A.block_mirror(q, k, v, kind, rab, block_m=32, block_n=32, diag_causal=False)
    err = float((bad - ref).norm() / ref.norm())
    assert err > 0.05
    # 每个对角块的最后一行看到的正好是完整的因果窗口：那一行不受影响
    assert torch.allclose(bad[0, 0, 31], ref[0, 0, 31], atol=1e-5)


def test_accounting_and_roofline():
    assert A.attn_flops(1, 1, 100, 64) == 2 * 100 * 100 * 64
    assert A.attn_bytes(1, 1, 100, 64, 2, "naive") > A.attn_bytes(1, 1, 100, 64, 2, "triton")
    r = BN.roofline(4e9, 1e9, 148, 4000)                                   # AI = 4 FLOP/B：带宽受限，4 × 4 TB/s = 16 TFLOPS
    assert r["ridge"] == 37.0 and r["bound_tflops"] == 16.0
    assert BN.roofline(1e12, 1e9, 148, 4000)["bound_tflops"] == 148         # AI = 1000：算力受限


def test_timer_cpu():
    cfg = PARAMS.resolve({"iters": 5, "warmup": 1, "min_time_ms": 0})
    t = BN.Timer(torch.device("cpu"), cfg).time(lambda: torch.randn(100, 100) @ torch.randn(100, 100))
    assert t["n"] >= 5 and t["p10_ms"] <= t["median_ms"] <= t["p90_ms"]


# ---------------------------------------------------------------- 端到端
class Ctx:
    def __init__(self, d: Path):
        self.run_dir, self.run_id, self.s, self.tr, self.m = d, "t", {}, {}, []

    def stage(self, n): pass
    def log(self, m): pass
    def check(self): pass
    def info(self, *a): pass
    def warn(self, *a): pass
    def metric(self, name, value, step=None, **kw): self.m.append(name)
    def trace(self, k, p): self.tr[k] = json.loads(json.dumps(p, default=str))
    def summary(self, d): self.s.update(d)


def test_end_to_end_r05_cpu(tmp_path):
    from recops import experiment as X

    cfg = PARAMS.resolve({"device": "cpu", "emb_rows": 5000, "emb_batch": 128, "bag_lens": "1,10", "iters": 2, "warmup": 1,
                          "min_time_ms": 0, "e2e_rows": 2000, "batch_size": 64, "e2e_steps": 5, "e2e_profile_steps": 2,
                          "num_workers": 0, "parts": "emb,grid,optim,loader,e2e", "grid_rows": "1e3,1e4", "grid_dims": "8"})
    ctx = Ctx(tmp_path)
    X.run(ctx, cfg)
    for p in ("emb.bag10_uniform.eff_gbs", "emb.bag10_zipf.unique_frac", "optim.dense_over_sparse", "loader.samples_per_s",
              "e2e.stage_share.wait_data", "e2e.samples_per_s", "grid"):
        d = ctx.s
        for k in p.split("."):
            d = d[k]
    assert set(ctx.s["e2e"]["stage_ms"]) == {"wait_data", "h2d", "forward_backward", "optimizer"}


def test_end_to_end_r06_cpu(tmp_path):
    from recops import experiment as X

    cfg = PARAMS.resolve({"device": "cpu", "suite": "r06", "attn_impl": "naive", "attn_dtype": "fp32", "seq_lens": "32,64", "ragged_len": 40,
                          "head_dims": "16", "attn_batch": 1, "attn_heads": 2, "iters": 2, "warmup": 1, "min_time_ms": 0,
                          "layer_len": 32, "layer_d_model": 32})
    ctx = Ctx(tmp_path)
    X.run(ctx, cfg)
    assert ctx.s["check"]["max_rel_err"] < 1e-4
    assert {"softmax_L32_D16", "pointwise_L64_D16", "softmax_L40_D16"} <= set(ctx.s["attn"])
    assert 0 < ctx.s["layer"]["attn_share"] < 1


def test_scenarios_r05_r06():
    for c in ("R05", "R06"):
        specs = scenarios.expand(c, "h20x8")
        assert specs
        for s in scenarios.load_course(c)["scenarios"]:
            for e in s.get("expect", []):
                scenarios.check_expr(e)
