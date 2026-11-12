"""在 GPU 上测 v1-v5 的 TFLOPS，对比 cuBLAS（torch.matmul），并打印每版的理论分析（来自 code/matmul_model.py）。

运行（公司 GPU）：
    uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.cuda.bench_matmul --hw H100
    uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.cuda.bench_matmul --hw H100 --size 2048
输出：stages/s03_kernels/experiments/cuda_matmul_<hw>.md
"""

import argparse
import os
from pathlib import Path

import torch
from tabulate import tabulate

from stages.s03_kernels.code.matmul_model import VERSIONS, global_reads_per_output, roofline_cap_tflops
from stages.s03_kernels.cuda.matmul_ext import load, matmul
from tools.bench import benchmark
from tools.roofline import HARDWARE

EXP = Path(__file__).resolve().parents[1] / "experiments"


# --8<-- [start:bench]
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hw", default="H100", choices=sorted(HARDWARE))
    p.add_argument("--size", type=int, default=4096, help="M = N = K")
    args = p.parse_args()
    dev = torch.device("cuda")
    s = args.size
    flops = 2 * s**3

    # 编译一次，打印 ptxas 的寄存器用量（记下来代入 tools/occupancy.py）
    os.environ.setdefault("S3_CUDA_VERBOSE", "1")
    load()
    # cuBLAS FP32 基线必须关掉 TF32：否则 torch.matmul 会偷偷用 Tensor Core（TF32），不是同一个对手
    torch.backends.cuda.matmul.allow_tf32 = False
    a32, b32 = torch.randn(s, s, device=dev), torch.randn(s, s, device=dev)
    a16, b16 = a32.half(), b32.half()

    cublas = {
        torch.float32: benchmark(lambda: a32 @ b32, dev).median_ms,
        torch.float16: benchmark(lambda: a16 @ b16, dev).median_ms,
    }
    rows = [["cuBLAS FP32（SIMT，关 TF32）", f"{cublas[torch.float32]:.3f}", f"{flops / cublas[torch.float32] / 1e9:.1f}", "100%",
             "", "", "", ""],
            ["cuBLAS FP16（Tensor Core）", f"{cublas[torch.float16]:.3f}", f"{flops / cublas[torch.float16] / 1e9:.1f}", "100%",
             "", "", "", ""]]
    for v in VERSIONS:
        a, b = (a16, b16) if v.tensor_core else (a32, b32)
        ref = (a.float() @ b.float()) if v.tensor_core else (a32 @ b32)
        out = matmul(a, b, v.name)
        err = (out - ref).abs().max().item()
        # v1 在 4096 上要几百 ms，少测几次
        iters = 5 if v.name in ("v1", "v2") else 20
        ms = benchmark(lambda: matmul(a, b, v.name), dev, warmup=2, iters=iters).median_ms
        tflops = flops / ms / 1e9
        base = cublas[torch.float16 if v.tensor_core else torch.float32]
        intensity, cap, bound = roofline_cap_tflops(v, s, s, s, args.hw)
        rows.append([f"{v.name} {v.desc}", f"{ms:.3f}", f"{tflops:.1f}", f"{base / ms:.0%}",
                     f"{global_reads_per_output(v, s):.0f}", f"{intensity:.2f}", f"{cap:.1f}（{bound}）", f"{err:.1e}"])
    # --8<-- [end:bench]

    table = tabulate(rows, headers=["kernel", "实测 ms", "实测 TFLOPS", "相对同精度 cuBLAS", "每输出全局读次数",
                                    "算术强度（无缓存）", f"{args.hw} 模型上限 TFLOPS", "最大误差"], tablefmt="github")
    report = f"# CUDA 矩阵乘 5 个版本（{torch.cuda.get_device_name()}，M=N=K={s}，模型按 {args.hw}）\n\n{table}\n"
    print(report)
    EXP.mkdir(exist_ok=True)
    (EXP / f"cuda_matmul_{args.hw}.md").write_text(report)


if __name__ == "__main__":
    main()
