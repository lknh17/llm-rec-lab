"""transformers 后端（只用于 Mac 冒烟）：在主环境里逐条 prefill + 逐 token decode，产出和 vLLM 测量进程同样格式的记录。

没有 continuous batching、没有前缀缓存、没有投机解码：这些旋钮在这里不生效，只验证请求构造、计时和汇总口径。
"""

from __future__ import annotations

import time


def _device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(ctx, cfg: dict, model_path: str, prompts: list[list[int]], arrivals: list[float]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    dev = _device(cfg["device"])
    dtype = torch.bfloat16 if dev == "cuda" else torch.float16 if dev == "mps" else torch.float32
    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(dev).eval()
    load_s = time.perf_counter() - t
    sync = torch.cuda.synchronize if dev == "cuda" else torch.mps.synchronize if dev == "mps" else (lambda: None)
    recs = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, ids in enumerate(prompts):
            ctx.check()
            wait = arrivals[i] - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
            arrival = max(arrivals[i], time.perf_counter() - t0) if cfg["request_rate"] > 0 else 0.0
            x = torch.tensor([ids], device=dev)                       # [1, T]
            out = model(input_ids=x, use_cache=True)                  # prefill：logits [1, T, V]
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)          # [1, 1]
            sync()
            chunks = [time.perf_counter() - t0]
            past = out.past_key_values
            for _ in range(cfg["output_len"] - 1):                    # decode：每步 1 个 token，读一遍权重和 KV
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
                sync()
                chunks.append(time.perf_counter() - t0)
            recs.append({"arrival": arrival, "chunk_times": chunks, "n_out": cfg["output_len"], "n_in": len(ids), "num_cached": None})
    duration = time.perf_counter() - t0
    env = {"backend": "hf", "torch": torch.__version__, "device": dev}
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return {"records": recs, "duration_s": duration, "max_decoding": 1, "mean_decoding": 1.0, "prom_before": {}, "prom_after": {},
            "load_s": load_s, "env": env, "log": []}
