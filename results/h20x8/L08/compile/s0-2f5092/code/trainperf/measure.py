"""显存：解析估算（按策略和卡数切分）、实测快照、四部分构成；profiler 区间求并集算 GPU 忙闲和暴露的通信；硬件自测。"""

from __future__ import annotations

import time

import torch

GB = 1e9


# --8<-- [start:breakdown]
def estimate(params: int, strategy: str, world: int, precision: str) -> dict:
    """静态显存（每卡，GB）。PyTorch 混合精度的实际口径：fp32 主参数 4 + fp32 梯度 4 + AdamW m、v 8 = 16 字节/参数。
    （讲义里常见的 2 + 2 + 12 口径是 bf16 参数 / 梯度 + fp32 主参数和 m、v，合计同样 16。）
    DDP：每卡一整份；FSDP（ZeRO-3）：三项都 ÷ N；ZeRO-2（fsdp_grad_op）存储同样 ÷ N，但前向后保留整份 bf16 参数。"""
    n = world if strategy.startswith("fsdp") else 1
    p, g, o = params * 4 / n, params * 4 / n, params * 8 / n
    out = {"params_gb": p / GB, "grads_gb": g / GB, "optimizer_gb": o / GB}
    if strategy == "fsdp_grad_op":
        out["unsharded_params_gb"] = params * (2 if precision == "bf16" else 4) / GB
    out["static_gb"] = sum(v for k, v in out.items() if k != "unsharded_params_gb")
    return {k: round(v, 3) for k, v in out.items()}


def snapshot(device) -> float:
    return torch.cuda.memory_allocated(device) / GB if device.type == "cuda" else 0.0


def breakdown(snaps: dict, peak_gb: float | None) -> dict:
    """实测快照：建模后 = 参数；首个反向后 − 建模后 = 梯度；首次 step 后 − 首个反向后 = 优化器状态；峰值 − 三者 = 激活与其他
    （logits、通信缓冲、FSDP 的整份参数、分配器碎片都在这一项）。"""
    p = snaps["after_build"]
    g = max(snaps["after_backward"] - p, 0.0)
    o = max(snaps["after_step"] - snaps["after_backward"], 0.0)
    out = {"params_gb": round(p, 3), "grads_gb": round(g, 3), "optimizer_gb": round(o, 3)}
    if peak_gb is not None:
        out["activations_other_gb"] = round(max(peak_gb - p - g - o, 0.0), 3)
    return out
# --8<-- [end:breakdown]


# --8<-- [start:prof]
def union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def length(iv: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in iv)


def subtract(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """a 的并集里不被 b 覆盖的长度（"暴露的通信" = 通信区间 − 计算区间）。"""
    total, j = 0.0, 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                total += b[k][0] - cur
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            total += e - cur
    return total


def kernel_stats(events: list[tuple[str, float, float]], wall_ms: float, steps: int) -> dict:
    """events：(kernel 名, 开始 μs, 结束 μs)。名字里有 nccl 的算通信，其余算计算。"""
    comp = union([(s, e) for n, s, e in events if "nccl" not in n.lower()])
    comm = union([(s, e) for n, s, e in events if "nccl" in n.lower()])
    busy = length(union([(s, e) for _, s, e in events])) / 1000
    by = {}
    for n, s, e in events:
        by[n] = by.get(n, 0.0) + (e - s) / 1000
    top = sorted(by.items(), key=lambda kv: -kv[1])[:5]
    return {"gpu_busy_frac": round(min(busy / wall_ms, 1.0), 4) if wall_ms else None,
            "gpu_idle_frac": round(max(1 - busy / wall_ms, 0.0), 4) if wall_ms else None,
            "compute_ms_per_step": round(length(comp) / 1000 / steps, 3), "nccl_ms_per_step": round(length(comm) / 1000 / steps, 3),
            "nccl_exposed_ms_per_step": round(subtract(comm, comp) / 1000 / steps, 3),
            "kernels_per_step": round(len(events) / steps, 1),
            "top_kernels": [{"name": n[:80], "ms_per_step": round(t / steps, 3)} for n, t in top]}
# --8<-- [end:prof]


def hwbench(device, world: int, rank: int) -> dict:
    """bf16 8192³ GEMM 的 TFLOPS；512 MB all-reduce 的 busbw（= algbw × 2(N−1)/N）；4 GB 拷贝的带宽。"""
    out = {}
    if device.type != "cuda":
        return out

    def timeit(fn, n=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    out["matmul_tflops_measured"] = round(2 * 8192 ** 3 / timeit(lambda: a @ a) / 1e12, 1)
    x = torch.empty(2 ** 30, dtype=torch.uint8, device=device)
    y = torch.empty_like(x)
    out["hbm_gbs_measured"] = round(2 * 2 ** 30 / timeit(lambda: y.copy_(x)) / 1e9, 0)
    del a, x, y
    if world > 1:
        import torch.distributed as dist
        buf = torch.empty(2 ** 27, dtype=torch.float32, device=device)       # 512 MB
        t = timeit(lambda: dist.all_reduce(buf), n=5)
        out["allreduce_busbw_gbs"] = round(buf.numel() * 4 / t / 1e9 * 2 * (world - 1) / world, 1)
        del buf
    torch.cuda.empty_cache()
    return out
