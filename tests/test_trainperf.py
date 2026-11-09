"""trainperf（L08）：参数量与 Qwen2.5 公开值、FLOPs / MFU、显存估算、naive 与 SDPA 一致、profiler 区间运算、
CPU 2 进程 gloo DDP 端到端（同步次数）、单卡 OOM 记录；labkit 多卡槽位和项目自报显存构成。"""

import json
from pathlib import Path

import pytest
import torch

from labkit import batch, evidence, scenarios
from trainperf import measure as M
from trainperf.data import GreedyTokenizer, TextDataset, bpe_merges, normalize
from trainperf.model import LM, flops_per_token, param_counts
from trainperf.params import PARAMS
from trainperf.train import mfu


@pytest.mark.parametrize("preset,want", [("qwen0.5b", 494_032_768), ("qwen1.5b", 1_543_714_304), ("qwen7b", 7_615_616_512)])
def test_param_counts_match_qwen25(preset, want):
    assert param_counts(preset)["params_total"] == want
    with torch.device("meta"):
        m = LM(preset, seq_len=16)
    assert sum(p.numel() for p in m.parameters()) == want


def test_flops_and_mfu():
    f = flops_per_token("qwen7b", 2048)
    assert f["model"] == pytest.approx(6 * param_counts("qwen7b")["params_matmul"] + 6 * 28 * 2048 * 3584)
    cfg = PARAMS.resolve({"preset": "qwen7b"})
    m = mfu(cfg, 650.0, 148.0, None)
    assert 0.18 < m["mfu"] < 0.25                                             # 讲义里的估算：约 20%
    assert mfu({**cfg, "ckpt": "full"}, 650.0, 148.0, None)["hfu"] > m["mfu"]


def test_memory_estimate():
    e = M.estimate(1_543_714_304, "ddp", 8, "bf16")
    assert e["static_gb"] == pytest.approx(16 * 1.5437, rel=1e-3)
    f = M.estimate(1_543_714_304, "fsdp_full", 8, "bf16")
    assert f["static_gb"] == pytest.approx(e["static_gb"] / 8, rel=1e-3)
    assert M.breakdown({"after_build": 6.0, "after_backward": 12.0, "after_step": 24.0}, 40.0) == \
        {"params_gb": 6.0, "grads_gb": 6.0, "optimizer_gb": 12.0, "activations_other_gb": 16.0}


def test_naive_attention_matches_sdpa():
    torch.manual_seed(0)
    a = LM("tiny", "sdpa", 32).eval()
    b = LM("tiny", "naive", 32).eval()
    b.load_state_dict(a.state_dict())
    ids = torch.randint(0, 1024, (2, 32))
    assert torch.allclose(a(ids), b(ids), atol=1e-4)


def test_interval_math():
    u = M.union([(0, 2), (1, 3), (5, 6)])
    assert u == [(0, 3), (5, 6)] and M.length(u) == 4
    assert M.subtract([(0, 10)], [(2, 4), (6, 7)]) == 7                       # 通信 0-10，计算覆盖 3：暴露 7
    st = M.kernel_stats([("gemm", 0, 4000), ("ncclKernel_AllReduce", 3000, 6000), ("gemm", 7000, 8000)], wall_ms=10.0, steps=1)
    assert st["gpu_busy_frac"] == pytest.approx(0.7) and st["nccl_exposed_ms_per_step"] == pytest.approx(2.0)


def test_cpu_pipeline_full_length_and_deterministic():
    ds = TextDataset(128, 1024, 5, 4, seed=0)
    x, y = ds[1], ds[1]
    assert x.shape == (129,) and torch.equal(x, y) and int((x > 0).sum()) == 129
    assert normalize("Café Ｆｕｌｌ") == "cafe full"
    assert len(bpe_merges([1, 2, 1, 2, 1, 2], 1, 100)) == 3
    assert GreedyTokenizer(1024).encode("recommend") != []


# ---------------------------------------------------------------- 端到端（CPU）
class Ctx:
    def __init__(self):
        self.s, self.tr, self.m, self.w = {}, {}, [], []

    def stage(self, n): pass
    def log(self, m): pass
    def check(self): pass
    def info(self, *a): pass
    def warn(self, *a): self.w.append(a)
    def metric(self, name, value, step=None, **kw): self.m.append((name, kw))
    def trace(self, k, p): self.tr[k] = json.loads(json.dumps(p, default=str))
    def summary(self, d): self.s.update(d)


SMOKE = {"preset": "tiny", "device": "cpu", "seq_len": 32, "mbs": 2, "steps": 24, "profile_steps": 0, "bench_hw": False, "prep_passes": 1}


def run(over):
    from trainperf import experiment as X

    ctx = Ctx()
    X.run(ctx, PARAMS.resolve({**SMOKE, **over}))
    return ctx


def test_single_end_to_end():
    ctx = run({"strategy": "none", "world_size": 1})
    s = ctx.s
    for p in ("throughput.tokens_per_s_per_gpu", "mfu.mfu", "time.data_frac", "time.step_ms", "memory.est.static_gb", "model.params_total"):
        d = s
        for k in p.split("."):
            d = d[k]
    assert s["time"]["data_ms"] + s["time"]["fwd_ms"] + s["time"]["bwd_ms"] + s["time"]["opt_ms"] == pytest.approx(s["time"]["step_ms"], abs=0.05)
    assert {"step_timeline", "memory_timeline"} <= set(ctx.tr)
    assert all(kw.get("tokens_per_s") for n, kw in ctx.m if n == "train_loss")


@pytest.mark.parametrize("gas,no_sync,want", [(1, True, 1.0), (4, True, 1.0), (4, False, 4.0)])
def test_ddp_gloo_two_ranks_sync_count(gas, no_sync, want):
    ctx = run({"strategy": "ddp", "world_size": 2, "backend": "gloo", "grad_accum": gas, "no_sync": no_sync})
    assert ctx.s["parallel"]["world_size"] == 2
    assert ctx.s["comm"]["syncs_per_opt_step"] == want
    assert len(ctx.s["memory"]["per_rank_peak_gb"]) == 2 and not ctx.w


def test_oom_recorded_as_success(monkeypatch):
    """模拟单卡在 optimizer 阶段 OOM：summary 里记阶段序号 3，run 不抛异常。"""
    import trainperf.train as T

    real = torch.optim.AdamW.step

    def boom(self, *a, **k):
        raise torch.cuda.OutOfMemoryError("模拟")

    monkeypatch.setattr(torch.optim.AdamW, "step", boom)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    ctx = run({"strategy": "none", "world_size": 1})
    assert ctx.s["memory"]["oom"] is True and ctx.s["memory"]["oom_stage"] == "optimizer" and ctx.s["memory"]["oom_stage_idx"] == 3
    monkeypatch.setattr(torch.optim.AdamW, "step", real)
    del T


# ---------------------------------------------------------------- labkit
def test_make_slots_multi_gpu(monkeypatch):
    monkeypatch.setenv("LABKIT_GPUS", "0,1,2,3,4,5,6,7")
    s8 = batch.make_slots({"slots": {"gpus_per_job": 8}})
    assert [(s.name, s.gpu) for s in s8] == [("gpu0-7-0", "0,1,2,3,4,5,6,7")]
    s2 = batch.make_slots({"slots": {"gpus_per_job": 2}})
    assert [s.gpu for s in s2] == ["0,1", "2,3", "4,5", "6,7"]
    assert len(batch.make_slots({"slots": {"gpus_per_job": 1}})) == 8
    monkeypatch.setenv("LABKIT_GPUS", "0,1")
    with pytest.raises(RuntimeError):
        batch.make_slots({"slots": {"gpus_per_job": 8}})


def test_evidence_uses_project_breakdown():
    own = {"memory": {"breakdown": {"params_gb": 6.2, "grads_gb": 6.2, "optimizer_gb": 12.4, "activations_other_gb": 18.0},
                      "peak_allocated_gb": 42.8, "method": "实测快照", "oom": False}}
    out = evidence.memory({"model": {"params": {"total": 1, "trainable": 1}}}, {}, own)
    assert out["breakdown"]["params_gb"] == 6.2 and out["peak_allocated_gb"] == 42.8 and out["oom"] is False
    lora = evidence.memory({"model": {"params": {"total": 1000, "trainable": 10}, "dtype": "bfloat16"}}, {}, {})
    assert "weights_gb" in lora["breakdown"]


def test_scenarios_l08():
    h = scenarios.expand("L08", "h20x8")
    assert scenarios.load_course("L08")["profiles"]["h20x8"]["slots"]["gpus_per_job"] == 8
    ids = {s.scenario for s in h}
    assert {"fsdp_full", "qwen7b_single", "compile"} <= ids
    m = {s.scenario for s in scenarios.expand("L08", "mac-smoke")}
    assert not ({"fsdp_full", "fsdp_grad_op", "compile", "qwen7b_single", "qwen7b_fsdp8"} & m)
    for s in scenarios.load_course("L08")["scenarios"]:
        for e in s.get("expect", []):
            scenarios.check_expr(e)
