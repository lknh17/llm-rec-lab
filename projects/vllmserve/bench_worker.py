"""在 vLLM 独立环境里运行的测量进程：python bench_worker.py <job.json>

主环境（labkit worker）构造好请求（token ID）和到达时刻写进 job.json，这里只做三件事：
建引擎 → 按到达时刻把请求送进去、记录每次收到输出的时刻 → 把原始记录和引擎计数器用一行 JSON 交回去。
分析（分位数、屋顶线）都在主环境做，这个文件只依赖 vLLM 和标准库，不 import 项目里的其他模块。

每个配置一个进程：vLLM 引擎退出时显存不一定还干净，下一个配置在新进程里建引擎最稳。
和主进程的约定：以 "@@VLLMSERVE " 开头的行是事件（JSON），其余都是日志（vLLM 自己的日志也打在 stdout）。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
import traceback

MARK = "@@VLLMSERVE "


def emit(event: str, **kw) -> None:
    print(MARK + json.dumps({"event": event, **kw}, ensure_ascii=False), flush=True)


def env_info() -> dict:
    import torch
    import vllm

    info = {"vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0]}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "gpu_mem_gb": round(p.total_memory / 1e9, 1), "capability": f"{p.major}.{p.minor}"})
    return info


# --8<-- [start:engine_args]
def engine_args(job: dict):
    """job 里的参数一一对应 vLLM 的引擎参数；不写的都用 vLLM 默认值。"""
    from vllm import AsyncEngineArgs

    kw = dict(model=job["model_path"], dtype="auto", seed=job["seed"],
              max_model_len=job["max_model_len"],                    # 输入 + 输出 + 投机余量，决定每条请求最多占多少 KV 块
              max_num_seqs=job["max_num_seqs"],                      # 同时在跑的请求上限
              max_num_batched_tokens=job["max_num_batched_tokens"],  # 每步 token 预算：prefill 按它切块，和 decode 拼在一步里
              enable_prefix_caching=job["enable_prefix_caching"],
              gpu_memory_utilization=job["gpu_memory_utilization"],  # 权重 + 激活之外剩下的都给 KV cache
              disable_log_stats=False)
    if job["quantization"] != "none":
        kw["quantization"] = job["quantization"]                     # fp8：加载 bf16 权重后在线转成 FP8 W8A8
    if job["spec"] == "ngram":
        kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": job["num_speculative_tokens"],
                                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    elif job["spec"] == "draft":
        kw["speculative_config"] = {"model": job["draft_path"], "num_speculative_tokens": job["num_speculative_tokens"]}
    return AsyncEngineArgs(**kw)
# --8<-- [end:engine_args]


def prom_snapshot() -> dict:
    """本进程 Prometheus 注册表里所有 vllm: 开头的数（计数器、仪表、直方图的 _sum/_count），同名不同标签的求和。"""
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        return {}
    out: dict = {}
    for fam in REGISTRY.collect():
        if not fam.name.startswith("vllm"):
            continue
        for s in fam.samples:
            if s.name.endswith(("_created", "_bucket")):
                continue
            if "cache_config_info" in s.name:
                out["cache_config"] = {k: v for k, v in s.labels.items() if k in ("block_size", "num_gpu_blocks", "cache_dtype")}
                continue
            out[s.name] = out.get(s.name, 0.0) + float(s.value)
    return out


# --8<-- [start:client]
async def bench(engine, prompts: list[list[int]], arrivals: list[float], output_len: int, closed_loop: bool = False) -> dict:
    """按到达时刻发请求（流式），记录每次收到新 token 的时刻。时间都相对第一条请求的发出时刻。

    closed_loop：上一条请求完成后才发下一条（忽略 arrivals）。一次只跑一条（max_num_seqs = 1）时用它：
    如果 16 条同时到达，后面的请求都在排队，TTFT 量到的是排队时间而不是 prefill（2026-11-12 H20 第一版就是这样，
    TTFT 中位数约 7 秒，其实单条 prefill 只有 0.5 秒左右）。"""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)   # 贪心 + 固定输出长度：各配置的工作量完全相同
    t0 = time.perf_counter()
    recs: list = [None] * len(prompts)
    state = {"decoding": 0, "max_decoding": 0, "done": 0}

    async def one(i: int) -> None:
        delay = arrivals[i] - (time.perf_counter() - t0)
        if delay > 0 and not closed_loop:
            await asyncio.sleep(delay)
        arrival = time.perf_counter() - t0
        chunks, n, cached = [], 0, None
        async for out in engine.generate(TokensPrompt(prompt_token_ids=prompts[i]), sp, request_id=f"r{i}"):
            k = len(out.outputs[0].token_ids)                     # 默认是累计输出：k 是到目前为止生成的 token 数
            if k > n:
                if n == 0:                                        # 第一个 token：这条请求从 prefill 进入 decode
                    state["decoding"] += 1
                    state["max_decoding"] = max(state["max_decoding"], state["decoding"])
                chunks.append(time.perf_counter() - t0)
                n = k
            if cached is None:
                cached = getattr(out, "num_cached_tokens", None)  # 这条请求的输入里有多少 token 命中了前缀缓存
        state["decoding"] -= 1
        state["done"] += 1
        recs[i] = {"arrival": arrival, "chunk_times": chunks, "n_out": n, "n_in": len(prompts[i]), "num_cached": cached}

    async def sampler() -> list:
        """每 100 ms 记一次"同时在 decode 的请求数"，每 10 s 报一次进度。"""
        samples, last = [], 0.0
        while state["done"] < len(prompts):
            samples.append(state["decoding"])
            now = time.perf_counter() - t0
            if now - last > 10:
                emit("progress", done=state["done"], total=len(prompts), t=round(now, 1))
                last = now
            await asyncio.sleep(0.1)
        return samples

    sampling = asyncio.create_task(sampler())
    if closed_loop:
        for i in range(len(prompts)):                             # 一条接一条：TTFT 就是这条请求自己的 prefill
            await one(i)
    else:
        await asyncio.gather(*(one(i) for i in range(len(prompts))))
    duration = time.perf_counter() - t0
    samples = await sampling
    return {"records": recs, "duration_s": duration, "max_decoding": state["max_decoding"],
            "mean_decoding": sum(samples) / len(samples) if samples else None}
# --8<-- [end:client]


async def main_async(job: dict) -> dict:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt

    t = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(engine_args(job))
    load_s = time.perf_counter() - t
    emit("loaded", load_s=round(load_s, 1))
    try:
        # 热身：几条短的随机请求（token 和正式请求不重叠，不会预热前缀缓存），触发各种形状的编译和 CUDA graph
        rng = random.Random(12345)
        warm = [[rng.randint(1000, 30000) for _ in range(64)] for _ in range(4)]
        sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)

        async def w(i: int) -> None:
            async for _ in engine.generate(TokensPrompt(prompt_token_ids=warm[i]), sp, request_id=f"warm{i}"):
                pass

        await asyncio.gather(*(w(i) for i in range(len(warm))))
        await engine.do_log_stats()
        before = prom_snapshot()
        res = await bench(engine, job["prompts"], job["arrivals"], job["output_len"], closed_loop=job.get("closed_loop", False))
        await engine.do_log_stats()                               # 让引擎把缓存命中率、投机接受长度打进日志（备用口径）
        res.update({"prom_before": before, "prom_after": prom_snapshot(), "load_s": load_s})
        return res
    finally:
        getattr(engine, "shutdown", lambda: None)()


def main() -> None:
    try:
        with open(sys.argv[1]) as f:
            job = json.load(f)
        emit("env", **env_info())
        res = asyncio.run(main_async(job))
        emit("result", **res)
    except BaseException:  # noqa: BLE001
        emit("error", error=traceback.format_exc()[-4000:])
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)                                                   # 不等引擎子进程的析构，避免退出时卡住


if __name__ == "__main__":
    main()
