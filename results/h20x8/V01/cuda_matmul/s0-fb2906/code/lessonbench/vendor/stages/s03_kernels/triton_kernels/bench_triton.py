"""在 GPU 上对比 Triton 实现与 PyTorch/cuBLAS/FlashAttention 的性能，并换算成 roofline 利用率。

运行（公司 GPU）：uv run python -m stages.s03_kernels.triton_kernels.bench_triton --hw H100
输出：stages/s03_kernels/experiments/triton_bench_<hw>.md
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tabulate import tabulate

from stages.s03_kernels.triton_kernels.flash_attn import flash_attention
from stages.s03_kernels.triton_kernels.fused_softmax import softmax
from stages.s03_kernels.triton_kernels.matmul import matmul
from stages.s03_kernels.triton_kernels.vector_add import add
from tools.bench import benchmark
from tools.roofline import HARDWARE

EXP = Path(__file__).resolve().parents[1] / "experiments"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hw", default="H100", choices=sorted(HARDWARE))
    args = p.parse_args()
    hw = HARDWARE[args.hw]
    dev = torch.device("cuda")
    rows = []

    n = 64 * 1024 * 1024
    x, y = torch.randn(n, device=dev, dtype=torch.bfloat16), torch.randn(n, device=dev, dtype=torch.bfloat16)
    for name, fn in [("vector add / torch", lambda: x + y), ("vector add / triton", lambda: add(x, y))]:
        ms = benchmark(fn, dev).median_ms
        rows.append([name, f"{ms:.3f}", f"{3 * n * 2 / ms / 1e6:.0f} GB/s", f"{3 * n * 2 / ms / 1e6 / hw.bandwidth_gbs:.0%}"])

    s = torch.randn(8192, 4096, device=dev, dtype=torch.bfloat16)
    for name, fn in [("softmax / torch", lambda: F.softmax(s, -1)), ("softmax / triton", lambda: softmax(s))]:
        ms = benchmark(fn, dev).median_ms
        gbs = 2 * s.numel() * 2 / ms / 1e6
        rows.append([name, f"{ms:.3f}", f"{gbs:.0f} GB/s", f"{gbs / hw.bandwidth_gbs:.0%}"])

    for size in [1024, 4096, 8192]:
        a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
        b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
        for name, fn in [(f"matmul {size} / cuBLAS", lambda: a @ b), (f"matmul {size} / triton", lambda: matmul(a, b))]:
            ms = benchmark(fn, dev).median_ms
            tf = 2 * size**3 / ms / 1e9
            rows.append([name, f"{ms:.3f}", f"{tf:.0f} TFLOPS", f"{tf / hw.peak_tflops:.0%}"])

    for t in [1024, 4096, 16384]:
        q, k, v = (torch.randn(4, 32, t, 128, device=dev, dtype=torch.bfloat16) for _ in range(3))
        flops = 4 * 4 * 32 * t * t * 128 / 2  # 因果减半
        for name, fn in [(f"attention T={t} / SDPA(FA2)", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
                         (f"attention T={t} / triton", lambda: flash_attention(q, k, v))]:
            ms = benchmark(fn, dev).median_ms
            tf = flops / ms / 1e9
            rows.append([name, f"{ms:.3f}", f"{tf:.0f} TFLOPS", f"{tf / hw.peak_tflops:.0%}"])

    table = tabulate(rows, headers=["kernel", "ms", "达到", "占峰值"], tablefmt="github")
    report = f"# Triton kernel benchmark（{torch.cuda.get_device_name()}，roofline 按 {hw.name}）\n\n{table}\n"
    print(report)
    EXP.mkdir(exist_ok=True)
    (EXP / f"triton_bench_{args.hw}.md").write_text(report)


if __name__ == "__main__":
    main()
