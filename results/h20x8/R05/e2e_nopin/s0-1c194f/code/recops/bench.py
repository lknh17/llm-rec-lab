"""微基准工具：热身、L2 冲刷、CUDA event 计时（中位数 + p10/p90）、增量峰值显存、本机参照带宽 / 算力、roofline。"""

from __future__ import annotations

import os
import subprocess
import time

import numpy as np
import torch


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    def __init__(self, device, cfg: dict, l2_flush: bool | None = None):
        self.device = device
        self.warmup, self.iters, self.min_ms = cfg["warmup"], cfg["iters"], cfg["min_time_ms"]
        flush = cfg["l2_flush"] if l2_flush is None else l2_flush
        self.buf = None
        if flush and device.type == "cuda":
            l2 = torch.cuda.get_device_properties(device).L2_cache_size
            self.buf = torch.empty(2 * l2, dtype=torch.uint8, device=device)

    # --8<-- [start:timer]
    def time(self, fn, inner: int = 1) -> dict:
        """先热身，再至少计时 iters 次、总时长至少 min_time_ms；每次之前冲刷 L2（冲刷不计时）。返回每次调用的毫秒统计。"""
        for _ in range(self.warmup):
            fn()
        sync(self.device)
        ts = []
        if self.device.type == "cuda":
            total, n = 0.0, 0
            while n < self.iters or total < self.min_ms:
                evs = []
                for _ in range(max(1, self.iters)):
                    if self.buf is not None:
                        self.buf.zero_()
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(inner):
                        fn()
                    e.record()
                    evs.append((s, e))
                torch.cuda.synchronize()                                   # 一轮只同步一次
                batch = [s.elapsed_time(e) / inner for s, e in evs]
                ts += batch
                total += sum(batch) * inner
                n += len(batch)
                if n >= 10 * self.iters:
                    break
        else:
            t_end = time.perf_counter() + self.min_ms / 1000
            while len(ts) < self.iters or time.perf_counter() < t_end:
                t0 = time.perf_counter()
                for _ in range(inner):
                    fn()
                sync(self.device)
                ts.append((time.perf_counter() - t0) * 1000 / inner)
                if len(ts) >= 10 * self.iters:
                    break
        a = np.array(ts)
        med = float(np.median(a))
        p10, p90 = float(np.percentile(a, 10)), float(np.percentile(a, 90))
        return {"median_ms": round(med, 5), "p10_ms": round(p10, 5), "p90_ms": round(p90, 5),
                "cv": round((p90 - p10) / med, 4) if med else None, "n": len(a)}
    # --8<-- [end:timer]


def peak_delta(fn, device) -> float | None:
    """fn 执行期间的增量峰值显存（GB）：峰值 − 调用前已分配的量。"""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return round((torch.cuda.max_memory_allocated() - base) / 1e9, 4)


def hw_info(device) -> dict:
    info = {"device": str(device), "torch": torch.__version__, "cpu_count": os.cpu_count()}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        info["triton"] = None
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update({"gpu": p.name, "sms": p.multi_processor_count, "l2_mb": round(p.L2_cache_size / 2 ** 20, 1),
                     "mem_gb": round(p.total_memory / 1e9, 1)})
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                                "--format=csv,noheader,nounits", "-i", str(device.index or 0)],
                               capture_output=True, text=True, timeout=10).stdout.strip().split(",")
            info.update({"sm_clock": float(q[0]), "sm_clock_max": float(q[1]), "temp_c": float(q[2]), "power_w": float(q[3])})
        except Exception:  # noqa: BLE001
            pass
    return info


# --8<-- [start:roofline]
def roofline(flops: float, nbytes: float, peak_tflops: float, hbm_gbs: float) -> dict:
    """算术强度 AI = FLOPs / 字节；可达上限 = min(峰值算力, AI × 带宽)。拐点 = 峰值 / 带宽（H20：148 / 4.0 ≈ 37 FLOP/B）。"""
    ai = flops / max(nbytes, 1)
    bound = min(peak_tflops, ai * hbm_gbs / 1000)
    return {"ai": round(ai, 3), "bound_tflops": round(bound, 2), "bound_ms": round(flops / (bound * 1e12) * 1000, 4) if bound else None,
            "ridge": round(peak_tflops * 1000 / hbm_gbs, 1)}
# --8<-- [end:roofline]


def ref_copy_gbs(device, cfg) -> float | None:
    """顺序拷贝 1 GB（读 + 写）的有效带宽：本机 HBM 的实际天花板。"""
    if device.type != "cuda":
        return None
    n = 2 ** 28
    a = torch.empty(n, dtype=torch.float32, device=device)
    b = torch.empty_like(a)
    t = Timer(device, cfg, l2_flush=False).time(lambda: b.copy_(a))
    return round(2 * n * 4 / (t["median_ms"] / 1000) / 1e9, 1)


def ref_matmul_tflops(device, cfg) -> float | None:
    if device.type != "cuda":
        return None
    n = 8192
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    t = Timer(device, cfg, l2_flush=False).time(lambda: a @ b)
    return round(2 * n ** 3 / (t["median_ms"] / 1000) / 1e12, 1)


def ref_h2d_gbs(device, cfg, pinned: bool) -> float | None:
    if device.type != "cuda":
        return None
    x = torch.empty(64 * 2 ** 20, dtype=torch.uint8, pin_memory=pinned)
    t = Timer(device, cfg, l2_flush=False).time(lambda: x.to(device, non_blocking=pinned))
    return round(x.numel() / (t["median_ms"] / 1000) / 1e9, 1)
