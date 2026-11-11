"""GPU 环境检查 + 实测校准。在公司 GPU 机器上运行：

    uv run python -m tools.gpu_check

做三件事：
1. 环境信息：GPU 型号、显存、驱动/CUDA 版本、torch 版本，以及 triton、vllm 是否可用。
2. 实测矩阵乘峰值算力（BF16）：和厂商标称值对比，得到"可达峰值"。
3. 实测显存带宽（大块拷贝）：和标称值对比，得到"可达带宽"。

后两项是 S2 roofline 估算要用的真实上限，结果写入 tools/gpu_calibration.md。
"""

import importlib.util
import subprocess
from pathlib import Path

import torch

from tools.bench import benchmark

OUT = Path(__file__).resolve().parent / "gpu_calibration.md"


def env_info() -> list[str]:
    lines = [f"- torch {torch.__version__}，CUDA（torch 编译版本）{torch.version.cuda}"]
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        for i, row in enumerate(smi.splitlines()):
            lines.append(f"- GPU{i}: {row}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        lines.append("- nvidia-smi 不可用")
    for pkg in ["triton", "vllm", "flash_attn", "tensorrt_llm"]:
        lines.append(f"- {pkg}: {'已安装' if importlib.util.find_spec(pkg) else '未安装'}")
    return lines


# --8<-- [start:calibrate]
# --8<-- [start:calibrate_matmul]
def measure_matmul_tflops(device: torch.device, n: int = 8192) -> float:
    # 两个 8192×8192 的 BF16 方阵，各 8192² × 2 B ≈ 134 MB；强度 = 2N³ / (3 · 2N²) = N/3 ≈ 2730 FLOP/B，远超任何卡的拐点
    # 所以这个矩阵乘一定是算力瓶颈，测出来的就是"可达峰值算力"
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)  # [N, N]
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)  # [N, N]
    # benchmark 内部：先 warmup（让 cuBLAS 选好算法、显存分配好），再同步计时 30 次取中位数
    # lambda 把"要测的那一行"包成一个无参函数传进去
    # 2 × 8192³ ≈ 1.1 TFLOP，H100 上按可达 600-750 TFLOPS 约 1.5-2 ms，比 kernel 启动开销大三个数量级
    res = benchmark(lambda: a @ b, device, warmup=10, iters=30)
    # 2·N³ FLOPs ÷ 秒数 ÷ 1e12 = TFLOPS；H100 通常测到 600-800（标称 989）
    return 2 * n**3 / (res.median_ms * 1e-3) / 1e12
# --8<-- [end:calibrate_matmul]


# --8<-- [start:calibrate_bandwidth]
def measure_bandwidth_gbs(device: torch.device, size_gb: float = 2.0) -> float:
    # BF16 每个元素 2 字节，2 GB 对应 10 亿个元素；远大于 L2（50 MB），保证真的走 HBM
    n = int(size_gb * 1e9 / 2)
    # torch.empty 只分配不初始化，省掉一次填充；数值是什么不影响拷贝速度
    src = torch.empty(n, device=device, dtype=torch.bfloat16)  # [n]
    # empty_like：形状、dtype、设备都和 src 相同
    dst = torch.empty_like(src)  # [n]
    # copy_ 带下划线表示原地写入 dst，不分配新张量；它是纯访存，没有计算
    res = benchmark(lambda: dst.copy_(src), device, warmup=5, iters=30)
    # 字节数 = 读 src 2 GB + 写 dst 2 GB = 4 GB；H100 上按可达约 3 TB/s 约 1.3 ms
    return 2 * n * 2 / (res.median_ms * 1e-3) / 1e9  # 读一遍加写一遍
# --8<-- [end:calibrate_bandwidth]
# --8<-- [end:calibrate]


def main() -> None:
    lines = ["# GPU 校准结果", "", "## 环境", *env_info(), ""]
    if not torch.cuda.is_available():
        lines.append("CUDA 不可用：检查驱动和 torch 的 CUDA 版本是否匹配（见 stages/s00_setup/gpu_access.md）。")
    else:
        device = torch.device("cuda")
        tflops = measure_matmul_tflops(device)
        gbs = measure_bandwidth_gbs(device)
        lines += [
            "## 实测上限（GPU0）",
            f"- BF16 矩阵乘（8192³）：{tflops:.0f} TFLOPS",
            f"- 显存拷贝带宽：{gbs:.0f} GB/s",
            f"- 实测拐点算术强度：{tflops * 1e12 / (gbs * 1e9):.0f} FLOP/Byte",
            "",
            "与 GLOSSARY.md 里的标称值对比：可达值通常是标称算力的 60-80%、标称带宽的 80-90%。",
        ]
    report = "\n".join(lines) + "\n"
    print(report)
    OUT.write_text(report)
    print(f"已保存到 {OUT}")


if __name__ == "__main__":
    main()
