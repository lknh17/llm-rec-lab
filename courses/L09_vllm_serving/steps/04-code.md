<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 18 步：代码导读

> 阶段 1 / 7：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 构造请求

规范（S 个 token）、内容（C 个 token）、固定的开头和指令拼起来正好 input_len 个。head 时所有请求前 S + 3 个 token 相同；tail 时只有聊天模板的 3 个 token 相同，不到一个 KV 块（16 token）。

[`vllmserve/prompts.py#L83-L98`](../../../projects/vllmserve/prompts.py#L83-L98)

```python
def build(tok, input_len: int, num_prompts: int, shared_prefix: int, position: str, seed: int) -> tuple[list[list[int]], dict]:
    """返回 num_prompts 条 token ID 序列（每条正好 input_len 个）和构造说明。"""
    pre, post = _ids(tok, PRE), _ids(tok, INSTRUCTION + POST)
    content_len = input_len - shared_prefix - len(pre) - len(post)
    if content_len < 1:
        raise ValueError(f"input_len={input_len} 放不下共享前缀 {shared_prefix} + 固定部分 {len(pre) + len(post)} 个 token")
    guide = _take(tok, guide_text, shared_prefix)                       # [S]，所有请求相同
    prompts = []
    for i in range(num_prompts):
        rng = random.Random(seed * 1_000_003 + i)
        body = _take(tok, lambda n: content_text(rng, n), content_len)  # [C]，每条不同
        mid = guide + body if position == "head" else body + guide       # tail：开头就不同，前缀缓存无从命中
        prompts.append(pre + mid + post)
    meta = {"input_len": input_len, "shared_prefix": shared_prefix, "content_len": content_len, "fixed_len": len(pre) + len(post),
            "position": position, "same_head_tokens": len(pre) + (shared_prefix if position == "head" else 0)}
    return prompts, meta
```

### ② 引擎参数

max_num_seqs 限制同时在跑的请求数；max_num_batched_tokens 是每步的 token 预算，prefill 按它切块；gpu_memory_utilization 决定权重和激活之外还剩多少给 KV。

[`vllmserve/bench_worker.py#L40-L58`](../../../projects/vllmserve/bench_worker.py#L40-L58)

```python
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
```

### ③ 流式计时

每条请求一个协程，按到达时刻发出；每次收到新 token 记时刻。第一个时刻 − 到达 = TTFT（含排队）。

[`vllmserve/bench_worker.py#L83-L138`](../../../projects/vllmserve/bench_worker.py#L83-L138)

```python
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
```

### ④ 口径

TPOT 按条算再取分位数；ITL 是所有相邻输出间隔放在一起的分位数，被别人的 prefill 卡住的那一步会出现在 ITL 的长尾里。

[`vllmserve/stats.py#L31-L57`](../../../projects/vllmserve/stats.py#L31-L57)

```python
def latency(records: list[dict], duration_s: float, output_len: int) -> dict:
    """records：每条请求 {arrival, chunk_times: [每次收到新 token 的时刻], n_out}，时间单位秒、同一个起点。"""
    ttft, tpot, itl, e2e = [], [], [], []
    for r in records:
        ts = r["chunk_times"]
        if not ts:
            continue
        ttft.append(ts[0] - r["arrival"])                       # 含排队：请求到了但还没轮到 prefill 的时间也算
        e2e.append(ts[-1] - r["arrival"])
        if r["n_out"] > 1:
            tpot.append((ts[-1] - ts[0]) / (r["n_out"] - 1))    # 每条请求的平均每 token 间隔
        itl += [b - a for a, b in zip(ts, ts[1:])]              # 相邻两次输出的间隔：被别人的 prefill 卡住时这里出现长尾
    ms = 1000.0
    n_done = sum(1 for r in records if r["n_out"] >= output_len)
    in_tok = sum(r["n_in"] for r in records)
    out_tok = sum(r["n_out"] for r in records)
    return {
        "throughput": {"req_per_s": _r(len(records) / duration_s, 3), "input_tok_per_s": _r(in_tok / duration_s, 1),
                       "output_tok_per_s": _r(out_tok / duration_s, 1), "total_tok_per_s": _r((in_tok + out_tok) / duration_s, 1),
                       "duration_s": _r(duration_s, 3), "completed": n_done, "num_requests": len(records)},
        "latency": {"ttft_ms_p50": _r(pct(ttft, 50) * ms if ttft else None), "ttft_ms_p99": _r(pct(ttft, 99) * ms if ttft else None),
                    "ttft_ms_mean": _r(statistics.fmean(ttft) * ms if ttft else None),
                    "tpot_ms_p50": _r(pct(tpot, 50) * ms if tpot else None), "tpot_ms_p99": _r(pct(tpot, 99) * ms if tpot else None),
                    "itl_ms_p50": _r(pct(itl, 50) * ms if itl else None), "itl_ms_p99": _r(pct(itl, 99) * ms if itl else None),
                    "itl_ms_max": _r(max(itl) * ms if itl else None),
                    "e2e_ms_p50": _r(pct(e2e, 50) * ms if e2e else None), "e2e_ms_p99": _r(pct(e2e, 99) * ms if e2e else None)},
    }
```

### ⑤ 屋顶线

prefill 用单条 4096 的 FLOPs ÷ 算力；decode 用权重 + KV 字节 ÷ 带宽。FP8 的算力按 2 倍算，AWQ 的权重按 0.5 字节/参数算。

[`vllmserve/experiment.py#L39-L61`](../../../projects/vllmserve/experiment.py#L39-L61)

```python
def roofline_est(cfg: dict, gpu_mem_gb: float = 96.0) -> dict:
    """讲义 S2 的屋顶线公式（roofline.py 是 tools/roofline.py 的拷贝）：单条 prefill、batch=1 和 batch=max_num_seqs 的 decode 一步。

    prefill_ms 沿用讲义口径（lm_head 按全部 T 个位置算，61 TFLOPs、414 ms），和讲义数字一致；
    推理引擎只对最后一个位置算 logits，所以"实际要做的 FLOPs"（prefill_flops_served，用来算 MFU）扣掉 (T − 1) 个位置的 lm_head。"""
    m, wb = SHAPES[cfg["model"]], weight_bytes(cfg)
    T, O = cfg["input_len"], cfg["output_len"]
    ctx_len = T + O // 2                                              # decode 时的平均上下文长度
    peak = cfg["peak_tflops"] * (2 if cfg["quantization"] == "fp8" else 1)   # FP8 Tensor Core 标称算力是 bf16 的 2 倍
    hw = RL.Hardware("nominal", peak, cfg["hbm_gbps"], gpu_mem_gb)
    pf = RL.prefill(m, 1, T, hw, w_bytes=wb)                          # 61 TFLOPs（7B、4096）：算力瓶颈
    d1 = RL.decode(m, 1, ctx_len, hw, w_bytes=wb)                     # 读一遍 15.2 GB 权重 + 0.23 GB KV：带宽瓶颈
    db = RL.decode(m, cfg["max_num_seqs"], ctx_len, hw, w_bytes=wb)
    lm_head = 2 * m.vocab_size * m.d_model                            # 每个位置的 lm_head FLOPs（7B：1.09 GFLOPs）
    served = pf.flops - (T - 1) * lm_head                             # 只算最后一个位置的 logits：57 TFLOPs
    return {"prefill_ms": round(pf.time_s * 1e3, 2), "prefill_flops": pf.flops, "prefill_flops_served": served,
            "prefill_intensity": round(pf.intensity, 1), "prefill_compute_bound": pf.compute_time_s >= pf.memory_time_s,
            "decode_ms_b1": round(d1.time_s * 1e3, 3), "decode_bytes_b1": d1.bytes, "decode_intensity_b1": round(d1.intensity, 2),
            "decode_ms_at_max_seqs": round(db.time_s * 1e3, 3), "decode_intensity_at_max_seqs": round(db.intensity, 2),
            "decode_compute_bound_at_max_seqs": db.compute_time_s >= db.memory_time_s,
            "decode_flops_per_token": d1.flops, "peak_tflops_used": peak, "w_bytes": wb,
            "kv_bytes_per_token": m.kv_bytes_per_token(), "weights_gb": round(m.total_params * wb / 1e9, 2),
            "max_batch_est": RL.max_batch(m, T + O, hw, w_bytes=wb, mem_util=cfg["gpu_memory_utilization"])}
```

---

[← 追踪](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
