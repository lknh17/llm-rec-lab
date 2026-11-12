"""vllmserve 执行入口：构造请求 → 屋顶线估算 → 子进程里跑 vLLM（或 Mac 冒烟用 transformers）→ 汇总吞吐、TTFT / TPOT / ITL、
前缀缓存命中、并发与抢占、投机接受长度，和屋顶线对照 → summary。

vLLM 装在独立环境 vllm-env 里（labkit assets 自动准备），和主环境的 torch 互不影响；主进程只负责构造输入和分析。
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
from pathlib import Path

from . import prompts as P
from . import roofline as RL
from . import stats as S

HERE = Path(__file__).resolve().parent
MARK = "@@VLLMSERVE "
REPOS = {"qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-awq": "Qwen/Qwen2.5-7B-Instruct-AWQ",
         "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct"}
SHAPES = {"qwen2.5-7b": RL.MODELS["qwen2_5_7b"], "qwen2.5-7b-awq": RL.MODELS["qwen2_5_7b"],
          "qwen2.5-0.5b": RL.ModelShape("Qwen2.5-0.5B", 24, 896, 14, 2, 4864, 151936, tie_embeddings=True)}
DRAFT_MODEL = "qwen2.5-0.5b"
# 转发到运行日志的 vLLM 日志行（其余只用来解析，不刷屏）
LOG_KEEP = ("KV cache", "Maximum concurrency", "Model loading took", "ERROR", "Error", "Traceback", "out of memory",
            "Prefix cache hit rate", "acceptance", "Graph capturing finished", "torch.compile takes", "init engine")


def weight_bytes(cfg: dict) -> float:
    """每个权重元素的字节数：bf16 2、FP8 1、AWQ INT4 0.5（忽略 scale / zero point）。"""
    return 0.5 if cfg["model"].endswith("awq") else 1.0 if cfg["quantization"] == "fp8" else 2.0


# --8<-- [start:roofline]
def roofline_est(cfg: dict, gpu_mem_gb: float = 96.0) -> dict:
    """讲义 S2 的屋顶线公式（roofline.py 是 tools/roofline.py 的拷贝）：单条 prefill、batch=1 和 batch=max_num_seqs 的 decode 一步。"""
    m, wb = SHAPES[cfg["model"]], weight_bytes(cfg)
    T, O = cfg["input_len"], cfg["output_len"]
    ctx_len = T + O // 2                                              # decode 时的平均上下文长度
    peak = cfg["peak_tflops"] * (2 if cfg["quantization"] == "fp8" else 1)   # FP8 Tensor Core 标称算力是 bf16 的 2 倍
    hw = RL.Hardware("nominal", peak, cfg["hbm_gbps"], gpu_mem_gb)
    pf = RL.prefill(m, 1, T, hw, w_bytes=wb)                          # 61 TFLOPs（7B、4096）：算力瓶颈
    d1 = RL.decode(m, 1, ctx_len, hw, w_bytes=wb)                     # 读一遍 15.2 GB 权重 + 0.23 GB KV：带宽瓶颈
    db = RL.decode(m, cfg["max_num_seqs"], ctx_len, hw, w_bytes=wb)
    work = cfg["num_prompts"] * (pf.flops + O * d1.flops)             # 整批的模型 FLOPs（每条从头 prefill，不扣缓存省下的）
    return {"prefill_ms": round(pf.time_s * 1e3, 2), "prefill_flops": pf.flops, "prefill_intensity": round(pf.intensity, 1),
            "prefill_compute_bound": pf.compute_time_s >= pf.memory_time_s,
            "decode_ms_b1": round(d1.time_s * 1e3, 3), "decode_bytes_b1": d1.bytes, "decode_intensity_b1": round(d1.intensity, 2),
            "decode_ms_at_max_seqs": round(db.time_s * 1e3, 3), "decode_intensity_at_max_seqs": round(db.intensity, 2),
            "decode_compute_bound_at_max_seqs": db.compute_time_s >= db.memory_time_s,
            "work_flops": work, "peak_tflops_used": peak, "w_bytes": wb,
            "kv_bytes_per_token": m.kv_bytes_per_token(), "weights_gb": round(m.total_params * wb / 1e9, 2),
            "max_batch_est": RL.max_batch(m, T + O, hw, w_bytes=wb, mem_util=cfg["gpu_memory_utilization"])}
# --8<-- [end:roofline]


def compare_roofline(est: dict, lat: dict, thr: dict, cfg: dict) -> dict:
    """实测 ÷ 估算。单条请求（max_num_seqs = 1）时 TTFT 就是 prefill、TPOT 就是 batch=1 的 decode 一步，比值最有意义。"""
    out: dict = {}
    ttft, tpot = lat.get("ttft_ms_p50"), lat.get("tpot_ms_p50")
    peak_bf16 = cfg["peak_tflops"] * 1e12
    if ttft:
        out["ttft_vs_est"] = round(ttft / est["prefill_ms"], 3)
        out["prefill_mfu"] = round(est["prefill_flops"] / (ttft / 1e3) / peak_bf16, 4)       # 按 bf16 标称
    if tpot:
        out["tpot_vs_est"] = round(tpot / est["decode_ms_b1"], 3)
        out["decode_mbu"] = round(est["decode_bytes_b1"] / (tpot / 1e3) / (cfg["hbm_gbps"] * 1e9), 4)
    if thr.get("duration_s"):
        out["work_mfu"] = round(est["work_flops"] / thr["duration_s"] / peak_bf16, 4)      # 整批等效 MFU（不扣前缀缓存）
    return out


def _clean_env() -> dict:
    """子进程用 vllm-env 的解释器：去掉主环境的 PYTHONPATH / VIRTUAL_ENV，免得混进主环境的 torch。"""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")}
    env.update({"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "VLLM_LOGGING_LEVEL": "INFO"})
    return env


def _kill(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)                                  # 整个进程组：vLLM 的引擎核心是测量进程的子进程
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
        except subprocess.TimeoutExpired:
            pass


# --8<-- [start:subprocess]
def run_vllm(ctx, job: dict) -> dict:
    """用 vllm-env 的解释器跑 bench_worker.py（同一份代码快照里的），逐行读输出：事件行解析，日志行留着解析 KV 容量等。"""
    from labkit.assets import resolve_env_python

    py = resolve_env_python("vllm-env")
    job_path = ctx.run_dir / "vllm_job.json"                          # 含全部 token ID，跑完删掉，不进结果
    job_path.write_text(json.dumps(job))
    proc = subprocess.Popen([py, str(HERE / "bench_worker.py"), str(job_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=_clean_env(), start_new_session=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: ([lines.put(ln) for ln in proc.stdout], lines.put(None)), daemon=True).start()
    log: list[str] = []
    res, err, env = None, None, {}
    try:
        while True:
            ctx.check()                                               # 取消、超时：杀掉整个进程组
            try:
                ln = lines.get(timeout=1.0)
            except queue.Empty:
                continue
            if ln is None:
                break
            if ln.startswith(MARK):
                ev = json.loads(ln[len(MARK):])
                kind = ev.pop("event")
                if kind == "env":
                    env = ev
                    ctx.log(f"vLLM {ev.get('vllm')} · torch {ev.get('torch')} · CUDA {ev.get('cuda')} · {ev.get('gpu')}")
                elif kind == "loaded":
                    ctx.log(f"引擎就绪，用时 {ev['load_s']} 秒（加载权重、编译、CUDA graph、分配 KV cache）")
                elif kind == "progress":
                    ctx.log(f"已完成 {ev['done']}/{ev['total']} 条（{ev['t']} 秒）")
                elif kind == "result":
                    res = ev
                elif kind == "error":
                    err = ev["error"]
                continue
            log.append(ln.rstrip())
            if any(k in ln for k in LOG_KEEP):
                ctx.log("[vllm] " + ln.rstrip()[-300:])
        proc.wait(60)
    except BaseException:
        _kill(proc)
        raise
    finally:
        job_path.unlink(missing_ok=True)
    if res is None:
        raise RuntimeError(f"vLLM 测量进程失败（退出码 {proc.returncode}）：\n{err or chr(10).join(log[-40:])}")
    res["env"], res["log"] = {"backend": "vllm", **env}, log
    return res
# --8<-- [end:subprocess]


def run(ctx, cfg: dict) -> None:
    from transformers import AutoTokenizer

    from labkit.assets import resolve_model_path

    ctx.stage("prepare")
    path = resolve_model_path(REPOS[cfg["model"]])
    tok = AutoTokenizer.from_pretrained(path)
    prompts, pmeta = P.build(tok, cfg["input_len"], cfg["num_prompts"], cfg["shared_prefix"], cfg["prefix_position"], cfg["seed"])
    arrivals = P.arrivals(cfg["num_prompts"], cfg["request_rate"], cfg["seed"])
    shape = SHAPES[cfg["model"]]
    ctx.info("data", {"kind": "合成中文商品内容 + 共享打标规范（token 精确控制长度）", "input_len": cfg["input_len"],
                      "output_len": cfg["output_len"], "num_prompts": cfg["num_prompts"], "shared_prefix": cfg["shared_prefix"],
                      "prefix_position": cfg["prefix_position"], "request_rate": cfg["request_rate"]})
    ctx.info("model", {"arch": shape.name, "repo": REPOS[cfg["model"]], "params": {"total": shape.total_params, "trainable": 0},
                       "train_mode": "inference", "dtype": {2.0: "bf16", 1.0: "fp8 W8A8", 0.5: "int4 W4A16（AWQ）"}[weight_bytes(cfg)],
                       "backend": cfg["backend"]})
    same = next((i for i, (a, b) in enumerate(zip(prompts[0], prompts[1])) if a != b), len(prompts[0])) if len(prompts) > 1 else None
    ctx.trace("prompt", {"note": "第 0 条请求的开头和结尾（解码回文本）；后两项是构造说明和前两条请求开头相同的 token 数",
                         "head_text": tok.decode(prompts[0][:96]), "tail_text": tok.decode(prompts[0][-48:]),
                         **pmeta, "same_head_tokens_0_vs_1": same})
    ctx.log(f"{cfg['num_prompts']} 条 × {cfg['input_len']} token 输入 / {cfg['output_len']} token 输出；共享前缀 {cfg['shared_prefix']}"
            f"（{cfg['prefix_position']}），前两条开头相同 {same} 个 token；到达率 {cfg['request_rate'] or '全部同时'}")

    ctx.stage("train")
    if cfg["backend"] == "vllm":
        job = {k: cfg[k] for k in ("seed", "max_num_seqs", "max_num_batched_tokens", "enable_prefix_caching", "gpu_memory_utilization",
                                   "quantization", "spec", "num_speculative_tokens", "output_len")}
        job.update({"model_path": path, "prompts": prompts, "arrivals": arrivals,
                    "max_model_len": cfg["input_len"] + cfg["output_len"] + (cfg["num_speculative_tokens"] + 8 if cfg["spec"] != "none" else 8),
                    "draft_path": resolve_model_path(REPOS[DRAFT_MODEL]) if cfg["spec"] == "draft" else None})
        res = run_vllm(ctx, job)
    else:
        from . import hf_worker

        res = hf_worker.run(ctx, cfg, path, prompts, arrivals)

    ctx.stage("evaluate")
    recs = res["records"]
    base = S.latency(recs, res["duration_s"], cfg["output_len"])
    log = S.parse_log(res.get("log", []))
    eng = S.engine_metrics(S.prom_delta(res["prom_before"], res["prom_after"]), log, recs)
    cache_cfg = res["prom_after"].get("cache_config") or {}
    est = roofline_est(cfg, res["env"].get("gpu_mem_gb") or 96.0)
    roof = {**est, **compare_roofline(est, base["latency"], base["throughput"], cfg)}
    engine = {"max_num_seqs": cfg["max_num_seqs"], "max_num_batched_tokens": cfg["max_num_batched_tokens"],
              "enable_prefix_caching": cfg["enable_prefix_caching"], "gpu_memory_utilization": cfg["gpu_memory_utilization"],
              "quantization": cfg["quantization"], "spec": cfg["spec"], "num_speculative_tokens": cfg["num_speculative_tokens"],
              "kv_tokens": log.get("kv_tokens"), "kv_cache_gib": log.get("kv_cache_gib"), "weights_gib": log.get("model_load_gb"),
              "max_concurrency_est": log.get("max_concurrency"), "block_size": cache_cfg.get("block_size"),
              "num_gpu_blocks": cache_cfg.get("num_gpu_blocks"), "load_s": round(res["load_s"], 1)}
    sched = {**eng["sched"], "max_decoding": res["max_decoding"],
             "mean_decoding": round(res["mean_decoding"], 2) if res.get("mean_decoding") is not None else None}
    lat, thr = base["latency"], base["throughput"]
    first = [{"ttft_ms": round((r["chunk_times"][0] - r["arrival"]) * 1e3, 1), "n_out": r["n_out"], "chunks": len(r["chunk_times"]),
              "num_cached": r["num_cached"], "arrival_s": round(r["arrival"], 3)} for r in recs[:5] if r["chunk_times"]]
    ctx.trace("requests", {"note": "前 5 条请求：TTFT（含排队）、输出 token 数、分几次收到、命中缓存的输入 token 数", "rows": first})
    disp = {"条/s": thr["req_per_s"], "输入 tok/s": thr["input_tok_per_s"], "TTFT p50 ms": lat["ttft_ms_p50"],
            "TPOT p50 ms": lat["tpot_ms_p50"], "ITL p99 ms": lat["itl_ms_p99"], "前缀命中率": eng["cache"]["prefix_hit_rate"],
            "同时 decode 最多": sched["max_decoding"], "等效 MFU": roof.get("work_mfu")}
    ctx.summary({"env": res["env"], "load": {**pmeta, "output_len": cfg["output_len"], "num_prompts": cfg["num_prompts"],
                                             "request_rate": cfg["request_rate"]},
                 "engine": engine, **base, "cache": eng["cache"], "sched": sched, "spec": eng["spec"], "roofline": roof,
                 "headline": {"req_per_s": thr["req_per_s"], "ttft_ms_p50": lat["ttft_ms_p50"], "tpot_ms_p50": lat["tpot_ms_p50"],
                              "prefix_hit_rate": eng["cache"]["prefix_hit_rate"]},
                 "display": {k: v for k, v in disp.items() if v is not None}, "display_pct": ["前缀命中率", "等效 MFU"]})
    ctx.log(f"{thr['req_per_s']} 条/s（输入 {thr['input_tok_per_s']:,.0f} tok/s），TTFT p50 {lat['ttft_ms_p50']} ms，"
            f"TPOT p50 {lat['tpot_ms_p50']} ms，ITL p99 {lat['itl_ms_p99']} ms；前缀命中 {eng['cache']['prefix_hit_rate']}；"
            f"同时 decode 最多 {sched['max_decoding']}；屋顶线 prefill {est['prefill_ms']} ms / decode {est['decode_ms_b1']} ms")
