"""在 GPU 上测 MoE 层：Triton fused_moe vs 逐专家循环 vs 按最大负载 padding 的 bmm，和 roofline 估算对照。

形状按 Qwen3-30B-A3B 的一层 MoE：E=128、k=8、C=2048、I=768。
运行（公司 GPU，加锁）：
    uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.triton_kernels.bench_moe --hw H100
输出：stages/s03_kernels/experiments/moe_bench_<hw>.md
"""

import argparse
from pathlib import Path

import torch
from tabulate import tabulate

from stages.s03_kernels.code.moe_ref import moe_grouped_ref, moe_padded_bmm, padding_waste, zipf_routing
from stages.s03_kernels.triton_kernels.moe_grouped_gemm import fused_moe
from tools.bench import benchmark
from tools.roofline import HARDWARE

EXP = Path(__file__).resolve().parents[1] / "experiments"
E, K_TOP, C, I_FF = 128, 8, 2048, 768


def roofline_us(t: int, active: int, hw) -> float:
    """理想耗时：被选中专家的权重各读一遍（3·C·I·2 字节/专家）+ T·k 行的 3 个 GEMM 的 FLOPs，取 max。"""
    w_bytes = active * 3 * C * I_FF * 2
    act_bytes = t * C * 2 * 2 + t * K_TOP * (2 * I_FF + I_FF + C) * 2 * 2
    flops = 2 * t * K_TOP * 3 * C * I_FF
    return max(flops / (hw.peak_tflops * 1e12), (w_bytes + act_bytes) / (hw.bandwidth_gbs * 1e9)) * 1e6


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hw", default="H100", choices=sorted(HARDWARE))
    args = p.parse_args()
    hw = HARDWARE[args.hw]
    dev = torch.device("cuda")
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * I_FF, C, device=dev, dtype=torch.float16) * 0.02
    w2 = torch.randn(E, C, I_FF, device=dev, dtype=torch.float16) * 0.02
    rows = []
    for t in [1, 16, 64, 256, 1024, 4096]:
        x = torch.randn(t, C, device=dev, dtype=torch.float16)
        for alpha in [0.0, 1.2]:
            ids = zipf_routing(t, E, K_TOP, alpha).to(dev)
            tw = torch.softmax(torch.randn(t, K_TOP, device=dev), -1)
            info = padding_waste(ids.cpu(), E, 16)
            t_fused = benchmark(lambda: fused_moe(x, w1, w2, tw, ids), dev, warmup=5, iters=20).median_ms * 1e3
            t_loop = benchmark(lambda: moe_grouped_ref(x, tw, ids, w1, w2), dev, warmup=3, iters=10).median_ms * 1e3
            t_pad = benchmark(lambda: moe_padded_bmm(x, tw, ids, w1, w2), dev, warmup=3, iters=10).median_ms * 1e3
            est = roofline_us(t, info["active"], hw)
            rows.append([t, alpha, info["active"], f"{info['max_over_mean']:.1f}", f"{est:.1f}",
                         f"{t_fused:.1f}", f"{est / t_fused:.0%}", f"{t_loop:.1f}", f"{t_pad:.1f}"])

    table = tabulate(rows, headers=["T", "alpha", "被选中专家数", "最大/平均负载", "roofline μs", "fused_moe μs",
                                    "达到 roofline", "逐专家循环 μs", "padding bmm μs"], tablefmt="github")
    report = (f"# MoE benchmark（{torch.cuda.get_device_name()}，roofline 按 {hw.name}）\n\n"
              f"E={E}、k={K_TOP}、C={C}、I={I_FF}，FP16，含 moe_align 预处理和 silu_and_mul。\n\n{table}\n")
    print(report)
    EXP.mkdir(exist_ok=True)
    (EXP / f"moe_bench_{args.hw}.md").write_text(report)


if __name__ == "__main__":
    main()
