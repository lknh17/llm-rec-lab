"""在 GPU 上测 W4A16 GEMM：扫 M=1..256，对比 FP16 torch.matmul（cuBLAS），和字节/算力模型对照。

运行（公司 GPU，加锁）：
    uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.triton_kernels.bench_w4a16 --hw H100
输出：stages/s03_kernels/experiments/w4a16_bench_<hw>.md
"""

import argparse
from pathlib import Path

import torch
from tabulate import tabulate

from stages.s03_kernels.code.w4a16_ref import CUDA_CORE_TFLOPS, pack_int4, scales_for_kernel, w4a16_model
from stages.s03_kernels.triton_kernels.w4a16_gemm import w4a16_matmul
from stages.s05_inference.code.quant import quantize
from tools.bench import benchmark
from tools.roofline import HARDWARE

EXP = Path(__file__).resolve().parents[1] / "experiments"
G = 128


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hw", default="H100", choices=sorted(CUDA_CORE_TFLOPS))
    args = p.parse_args()
    hw = HARDWARE[args.hw]
    dev = torch.device("cuda")
    rows = []
    # 4096×4096：通用方阵；37888×3584：Qwen2.5-7B 的 gate_up 投影（N=2·18944，K=3584）
    for n, k in [(4096, 4096), (37888, 3584)]:
        w = torch.randn(n, k, device=dev) * 0.02
        q, s = quantize(w, 4, G)
        packed, sk = pack_int4(q), scales_for_kernel(s)
        wt = w.half().T.contiguous()  # [K, N]，FP16 基线
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            x = torch.randn(m, k, device=dev, dtype=torch.float16)
            t16 = benchmark(lambda: x @ wt, dev, warmup=10, iters=50).median_ms * 1e3
            t4 = benchmark(lambda: w4a16_matmul(x, packed, sk, G), dev, warmup=10, iters=50).median_ms * 1e3
            est = w4a16_model(m, n, k, args.hw, G)
            gbs16 = 2 * n * k / (t16 * 1e-6) / 1e9
            gbs4 = (0.5 * n * k + 2 * n * k / G) / (t4 * 1e-6) / 1e9
            rows.append([f"{n}x{k}", m, f"{t16:.1f}", f"{gbs16:.0f}", f"{t4:.1f}", f"{gbs4:.0f}",
                         f"{t16 / t4:.2f}x", f"{est['speedup']:.2f}x"])

    table = tabulate(rows, headers=["N×K", "M", "FP16 μs", "FP16 权重 GB/s", "W4A16 μs", "W4 权重 GB/s",
                                    "实测提速", "模型提速"], tablefmt="github")
    report = (f"# W4A16 GEMM benchmark（{torch.cuda.get_device_name()}，模型按 {hw.name}，g={G}）\n\n"
              f"权重 GB/s = 必须读的权重字节 / 耗时；HBM 标称 {hw.bandwidth_gbs:.0f} GB/s。\n\n{table}\n")
    print(report)
    EXP.mkdir(exist_ok=True)
    (EXP / f"w4a16_bench_{args.hw}.md").write_text(report)


if __name__ == "__main__":
    main()
