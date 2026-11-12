"""测量口径：从每条请求的到达时间和输出到达时间算吞吐与延迟；从 vLLM 的 Prometheus 计数器和日志取缓存、调度、投机解码指标。

纯 Python，主环境和测试都能用。口径和 vLLM 自带的 benchmark_serving 一致：
- TTFT = 第一个输出到达 − 请求到达（含排队）；
- TPOT = (最后一个 token 到达 − 第一个 token 到达) / (输出 token 数 − 1)，每条请求一个值，再取分位数；
- ITL = 相邻两次输出到达的间隔（流式输出的一个"块"可能含多个 token，投机解码时常见），所有请求的间隔放在一起取分位数。
"""

from __future__ import annotations

import re
import statistics


def pct(xs: list[float], q: float) -> float | None:
    """线性插值分位数（q 取 0-100），和 numpy.percentile 默认一致。"""
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


# --8<-- [start:latency]
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
# --8<-- [end:latency]


def prom_delta(before: dict, after: dict) -> dict:
    """两次 Prometheus 快照的差（计数器）；只在 after 里出现的量直接取 after。"""
    return {k: v - before.get(k, 0.0) for k, v in after.items() if isinstance(v, (int, float))}


def _pick(d: dict, *needles: str) -> float | None:
    """按名字片段找计数器（vLLM 版本之间名字后缀会变，例如 _total），多个匹配时求和。"""
    vals = [v for k, v in d.items() if all(n in k for n in needles) and not k.endswith(("_created", "_bucket"))]
    return sum(vals) if vals else None


LOG_PATTERNS = {
    "kv_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    "max_concurrency": re.compile(r"Maximum concurrency for\s*[\d,]+\s*tokens per request:\s*([\d.,]+)x"),
    "prefix_hit_rate_log": re.compile(r"Prefix cache hit rate:\s*(?:GPU:\s*)?([\d.]+)%"),
    "mean_accept_len_log": re.compile(r"Mean acceptance length:\s*([\d.]+)"),
    "model_load_gb": re.compile(r"Model loading took\s*([\d.]+)\s*GiB"),
    "kv_cache_gib": re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB"),
}


def parse_log(lines: list[str]) -> dict:
    """从 vLLM 日志里取引擎自报的数（每个模式取最后一次出现的值）。"""
    out: dict = {}
    for ln in lines:
        for k, pat in LOG_PATTERNS.items():
            m = pat.search(ln)
            if m:
                out[k] = float(m.group(1).replace(",", ""))
    return out


# --8<-- [start:engine_metrics]
def engine_metrics(prom: dict, log: dict, records: list[dict]) -> dict:
    """缓存、调度、投机解码。优先用每条请求自报的 num_cached_tokens，其次 Prometheus 计数器，最后是日志。"""
    in_tok = sum(r["n_in"] for r in records)
    cached = [r.get("num_cached") for r in records]
    if records and all(c is not None for c in cached):
        hit, src = sum(cached) / in_tok if in_tok else 0.0, "request.num_cached_tokens"
    elif (q := _pick(prom, "prefix_cache_queries")) and (h := _pick(prom, "prefix_cache_hits")) is not None:
        hit, src = h / q, "prometheus prefix_cache_hits / queries"
    elif "prefix_hit_rate_log" in log:
        hit, src = log["prefix_hit_rate_log"] / 100, "日志 Prefix cache hit rate"
    else:
        hit, src = None, None
    drafts, draft_tok, acc = (_pick(prom, "spec_decode_num_drafts"), _pick(prom, "spec_decode_num_draft_tokens"),
                              _pick(prom, "spec_decode_num_accepted_tokens"))
    spec = {"drafts": drafts, "draft_tokens": draft_tok, "accepted_tokens": acc,
            # 每次"验证"平均产出的 token 数 = 1（目标模型自己的那个）+ 平均接受的草稿数
            "mean_accept_len": _r(1 + acc / drafts, 3) if drafts and acc is not None else _r(log.get("mean_accept_len_log"), 3),
            "accept_rate": _r(acc / draft_tok, 4) if draft_tok and acc is not None else None}
    return {"cache": {"prefix_hit_rate": _r(hit, 4), "cached_tokens": sum(c for c in cached if c) if any(cached) else None,
                      "prompt_tokens": in_tok, "source": src},
            "sched": {"preemptions": _pick(prom, "num_preemptions")},
            "spec": spec}
# --8<-- [end:engine_metrics]
