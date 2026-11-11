"""vllmserve（L09）：请求构造的长度和共享前缀、到达时刻、延迟口径、vLLM 日志 / 计数器解析、屋顶线数字（和讲义一致）、
子进程协议（用假的测量进程）、vllm-env 资产、场景清单；Mac 上有 qwen2.5-0.5b 时用 transformers 后端端到端跑一遍。"""

import json
import sys
import textwrap

import pytest

from labkit import assets, scenarios
from vllmserve import experiment as X
from vllmserve import prompts as P
from vllmserve import stats as S
from vllmserve.params import PARAMS


class CharTok:
    """一个字符一个 token 的假分词器：长度好算，测试不依赖模型文件。"""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(map(chr, ids))


def test_prompts_exact_length_and_shared_prefix():
    tok = CharTok()
    head, mh = P.build(tok, 600, 4, 200, "head", seed=0)
    assert all(len(p) == 600 for p in head) and mh["content_len"] == 600 - 200 - mh["fixed_len"]
    pre = len(tok.encode(P.PRE))
    common = lambda a, b: next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)  # noqa: E731
    assert common(head[0], head[1]) >= pre + 200                         # 规范在前：开头 S 个 token 完全相同
    tail, mt = P.build(tok, 600, 4, 200, "tail", seed=0)
    assert all(len(p) == 600 for p in tail) and common(tail[0], tail[1]) < pre + 20   # 内容在前：开头马上不同
    assert head[0][-len(tok.encode(P.INSTRUCTION + P.POST)):] == tail[0][-len(tok.encode(P.INSTRUCTION + P.POST)):]
    assert P.build(tok, 600, 2, 200, "head", seed=0)[0] == head[:2]       # 确定性
    assert P.build(tok, 600, 2, 200, "head", seed=1)[0][0] != head[0]     # 种子换内容
    with pytest.raises(ValueError):
        P.build(tok, 100, 1, 200, "head", seed=0)


def test_arrivals():
    assert P.arrivals(5, 0.0, 0) == [0.0] * 5
    a = P.arrivals(4000, 2.0, 0)
    assert a[0] == 0.0 and all(y > x for x, y in zip(a, a[1:]))
    assert (a[-1] - a[0]) / (len(a) - 1) == pytest.approx(0.5, rel=0.08)


def test_latency_definitions():
    recs = [{"arrival": 0.0, "chunk_times": [0.5, 0.6, 0.7, 0.8], "n_out": 4, "n_in": 100, "num_cached": 0},
            {"arrival": 1.0, "chunk_times": [1.2, 1.3, 2.3], "n_out": 4, "n_in": 100, "num_cached": 50}]   # 第二条最后一块含 2 个 token
    out = S.latency(recs, 2.5, 4)
    assert out["throughput"]["req_per_s"] == 0.8 and out["throughput"]["completed"] == 2
    assert out["latency"]["ttft_ms_p50"] == pytest.approx(350.0)           # 500 和 200 的中位数
    assert out["latency"]["tpot_ms_p99"] == pytest.approx(364.0, abs=0.1)  # 100 和 (2.3 − 1.2) / 3 ≈ 366.7 的 p99（线性插值）
    assert out["latency"]["itl_ms_max"] == pytest.approx(1000.0)           # 被卡住的那一下
    assert S.pct([1, 2, 3, 4], 50) == 2.5 and S.pct([], 50) is None


def test_log_and_counter_parsing():
    log = ["INFO 11-11 10:00:00 [kv_cache_utils.py:1087] GPU KV cache size: 1,134,512 tokens",
           "INFO [kv_cache_utils.py:1091] Maximum concurrency for 4,162 tokens per request: 272.59x",
           "INFO [gpu_model_runner.py:2602] Model loading took 14.2488 GiB and 5.1 seconds",
           "INFO [gpu_worker.py:298] Available KV cache memory: 60.59 GiB",
           "INFO [loggers.py:123] Engine 000: Avg prompt throughput: 5000.0 tokens/s, Prefix cache hit rate: 47.5%",
           "INFO [metrics.py:96] SpecDecoding metrics: Mean acceptance length: 2.31, Accepted throughput: 10 tokens/s"]
    p = S.parse_log(log)
    assert p == {"kv_tokens": 1134512.0, "max_concurrency": 272.59, "model_load_gb": 14.2488, "kv_cache_gib": 60.59,
                 "prefix_hit_rate_log": 47.5, "mean_accept_len_log": 2.31}
    recs = [{"n_in": 100, "num_cached": None, "chunk_times": [1], "n_out": 1, "arrival": 0}]
    prom = S.prom_delta({"vllm:prefix_cache_queries_total": 100.0, "vllm:spec_decode_num_drafts_total": 0.0},
                        {"vllm:prefix_cache_queries_total": 300.0, "vllm:prefix_cache_hits_total": 50.0,
                         "vllm:spec_decode_num_drafts_total": 40.0, "vllm:spec_decode_num_draft_tokens_total": 160.0,
                         "vllm:spec_decode_num_accepted_tokens_total": 60.0, "vllm:num_preemptions_total": 3.0})
    m = S.engine_metrics(prom, p, recs)
    assert m["cache"]["prefix_hit_rate"] == 0.25 and m["cache"]["source"].startswith("prometheus")
    assert m["spec"]["mean_accept_len"] == 2.5 and m["spec"]["accept_rate"] == 0.375 and m["sched"]["preemptions"] == 3.0
    recs[0]["num_cached"] = 40                                              # 每条请求自报的优先
    assert S.engine_metrics(prom, p, recs)["cache"]["prefix_hit_rate"] == 0.4
    assert S.engine_metrics({}, p, [dict(recs[0], num_cached=None)])["cache"]["prefix_hit_rate"] == 0.475


def test_roofline_matches_lesson_numbers():
    """讲义 S2：Qwen2.5-7B 单条 4096 prefill 在 H20 上约 414 ms（算力瓶颈），batch=1 decode 一步约 3.9 ms（带宽瓶颈）。"""
    cfg = PARAMS.resolve({})
    e = X.roofline_est(cfg)
    assert e["prefill_ms"] == pytest.approx(414, abs=3) and e["prefill_compute_bound"]
    assert e["decode_ms_b1"] == pytest.approx(3.87, abs=0.05) and e["weights_gb"] == pytest.approx(15.23, abs=0.01)
    assert e["kv_bytes_per_token"] == 57344 and 290 < e["max_batch_est"] < 305    # (96 × 0.9 − 15.2) GB ÷ (4146 × 56 KB)
    f = X.roofline_est(PARAMS.resolve({"quantization": "fp8"}))
    assert f["prefill_ms"] == pytest.approx(e["prefill_ms"] / 2, rel=0.02) and f["decode_ms_b1"] < e["decode_ms_b1"] * 0.55
    a = X.roofline_est(PARAMS.resolve({"model": "qwen2.5-7b-awq"}))
    assert a["prefill_ms"] == pytest.approx(e["prefill_ms"], rel=0.01) and a["weights_gb"] == pytest.approx(15.23 / 4, abs=0.01)
    cmp = X.compare_roofline(e, {"ttft_ms_p50": 600.0, "tpot_ms_p50": 6.0}, {"duration_s": 100.0}, cfg)
    assert cmp["ttft_vs_est"] == pytest.approx(600 / e["prefill_ms"], rel=1e-3) and 0.6 < cmp["prefill_mfu"] < 0.7
    assert 0.6 < cmp["decode_mbu"] < 0.7


def test_roofline_copy_is_byte_identical_to_lesson_tool():
    """projects/vllmserve/roofline.py 由私有仓库 tools/export_lessonbench.py 同步；公开仓库里只检查它还能被导入、公式没被改坏。"""
    from vllmserve import roofline as RL

    assert RL.HARDWARE["H20"].peak_tflops == 148 and RL.HARDWARE["H20"].bandwidth_gbs == 4000
    assert "--8<-- [start:prefill_fn]" in (X.HERE / "roofline.py").read_text()


FAKE_WORKER = textwrap.dedent('''
    import json, sys
    job = json.load(open(sys.argv[1]))
    M = "@@VLLMSERVE "
    print(M + json.dumps({"event": "env", "vllm": "0.0-fake", "gpu": "FakeGPU", "gpu_mem_gb": 96.0}), flush=True)
    print("INFO GPU KV cache size: 1,000 tokens", flush=True)
    print(M + json.dumps({"event": "loaded", "load_s": 1.0}), flush=True)
    n = len(job["prompts"])
    recs = [{"arrival": 0.0, "chunk_times": [0.1 * (i + 1) + 0.01 * k for k in range(job["output_len"])], "n_out": job["output_len"],
             "n_in": len(p), "num_cached": 0} for i, p in enumerate(job["prompts"])]
    if job["seed"] == 99:
        print(M + json.dumps({"event": "error", "error": "Traceback ... CUDA out of memory"}), flush=True)
        sys.exit(1)
    print(M + json.dumps({"event": "result", "records": recs, "duration_s": 1.0, "max_decoding": 2, "mean_decoding": 1.5,
                          "prom_before": {}, "prom_after": {}, "load_s": 1.0}), flush=True)
''')


class Ctx:
    def __init__(self, run_dir):
        self.run_dir, self.logs, self.tr, self.s, self.inf = run_dir, [], {}, {}, {}

    def stage(self, s): pass
    def log(self, m): self.logs.append(m)
    def check(self): pass
    def info(self, k, d): self.inf[k] = d
    def trace(self, k, p): self.tr[k] = json.loads(json.dumps(p, default=str))
    def summary(self, d): self.s.update(d)


def test_subprocess_protocol(tmp_path, monkeypatch):
    fake = tmp_path / "proj"
    fake.mkdir()
    (fake / "bench_worker.py").write_text(FAKE_WORKER)
    monkeypatch.setattr(X, "HERE", fake)
    monkeypatch.setenv("LABKIT_VLLM_PYTHON", sys.executable)
    ctx = Ctx(tmp_path)
    job = {"prompts": [[1, 2, 3], [4, 5]], "arrivals": [0.0, 0.0], "output_len": 3, "seed": 0}
    res = X.run_vllm(ctx, job)
    assert res["env"]["vllm"] == "0.0-fake" and res["max_decoding"] == 2 and len(res["records"]) == 2
    assert S.parse_log(res["log"])["kv_tokens"] == 1000 and not (tmp_path / "vllm_job.json").exists()
    assert any("KV cache" in m for m in ctx.logs)
    with pytest.raises(RuntimeError, match="out of memory"):
        X.run_vllm(ctx, {**job, "seed": 99})


def test_vllm_env_asset(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_ASSETS_DIR", str(tmp_path))
    monkeypatch.delenv("LABKIT_VLLM_PYTHON", raising=False)
    assert assets.resolve_env_python("vllm-env", must_exist=False) is None
    assert assets.env_dir("vllm-env").name == "vllm-env--vllm-0.11.0"
    monkeypatch.setenv("LABKIT_VLLM_SPEC", "vllm==0.12.0")
    assert assets.env_dir("vllm-env").name == "vllm-env--vllm-0.12.0"      # 改版本 = 新目录，旧环境不动
    monkeypatch.setenv("LABKIT_VLLM_PYTHON", sys.executable)
    assert assets.resolve_env_python("vllm-env") == sys.executable
    st = assets.status()["vllm-env"]
    assert st["kind"] == "env" and st["ok"]
    with pytest.raises(SystemExit):
        assets.ensure(["no-such-asset"])


def test_uv_found_outside_path(monkeypatch, tmp_path):
    """systemd 服务的 PATH 里没有 ~/.local/bin 时，也要找到 uv（`uv run` 设的 UV 变量或默认安装位置）。"""
    fake = tmp_path / "uv"
    fake.write_text("")
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setenv("UV", str(fake))
    assert assets.uv_executable() == str(fake)
    monkeypatch.delenv("UV")
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".local" / "bin").mkdir(parents=True)
    (tmp_path / ".local" / "bin" / "uv").write_text("")
    assert assets.uv_executable() == str(tmp_path / ".local" / "bin" / "uv")


def test_scenarios_l09():
    h = scenarios.expand("L09", "h20x8")
    ids = {s.scenario for s in h}
    assert {"env_probe", "baseline", "seqs_1", "prefix_on", "chunk_2k", "fp8", "awq", "spec_ngram_c1", "kv_util_05",
            "fault_prefix_tail", "fix_prefix_head"} <= ids
    info = scenarios.profile_info("L09", "h20x8")
    assert info["slots"]["gpus_per_job"] == 1 and "vllm-env" in info["assets"]
    for s in h:
        assert s.cfg["shared_prefix"] < s.cfg["input_len"]
    m = scenarios.expand("L09", "mac-smoke")
    assert {s.scenario for s in m}.isdisjoint({"env_probe", "fp8", "awq", "awq_c1", "spec_ngram_c1", "spec_draft_c1", "kv_util_05"})
    for s in m:
        assert s.cfg["backend"] == "hf" and s.cfg["shared_prefix"] < s.cfg["input_len"] and not s.expect
    for s in scenarios.load_course("L09")["scenarios"]:
        for e in s.get("expect", []):
            scenarios.check_expr(e)


def _local_model():
    return assets.resolve_model_path("Qwen/Qwen2.5-0.5B-Instruct", must_exist=False)


@pytest.mark.skipif(not _local_model(), reason="本机没有 qwen2.5-0.5b")
def test_hf_backend_end_to_end(tmp_path):
    cfg = PARAMS.resolve({"backend": "hf", "model": "qwen2.5-0.5b", "device": "cpu", "input_len": 96, "output_len": 4,
                          "num_prompts": 2, "shared_prefix": 32})
    ctx = Ctx(tmp_path)
    X.run(ctx, cfg)
    s = ctx.s
    assert s["throughput"]["completed"] == 2 and s["latency"]["ttft_ms_p50"] > 0 and s["latency"]["tpot_ms_p50"] > 0
    assert s["load"]["input_len"] == 96 and s["roofline"]["prefill_ms"] > 0 and s["env"]["backend"] == "hf"
    assert ctx.tr["prompt"]["same_head_tokens_0_vs_1"] >= 32 and len(ctx.tr["requests"]["rows"]) == 2
    assert "条/s" in s["display"]
