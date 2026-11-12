"""S3 第 5 课：CUDA 矩阵乘 5 个版本的访存 / 算力模型（纯计算，Mac 上就能跑）。

对 cuda/matmul_kernels.cu 里的 v1-v5，算出：
- 每个输出元素要从全局内存读几个数；
- 不考虑 L1/L2 缓存复用时，按 32 字节 sector 计的全局访存量和算术强度；
- 在 H100 roofline 上的理论上限（带宽上限和算力上限取小）；
- 每个版本的 block 配置在 H100 上的理论 occupancy（寄存器数是估算，GPU 上用 -Xptxas -v 核对）。
第 6 步旋钮：v4 的 block tile 大小、v3 的 tile 大小、decode 形状（M=16）。
运行：uv run python -m stages.s03_kernels.code.matmul_model
"""

import math
from dataclasses import dataclass
from pathlib import Path

from tabulate import tabulate

from tools.occupancy import occupancy
from tools.roofline import HARDWARE

OUT = Path(__file__).resolve().parents[1] / "experiments" / "matmul_model.md"
# 非 Tensor Core 的 FP32 峰值（CUDA Core FFMA）：v1-v4 只能用这个；FP16 Tensor Core 峰值用 tools/roofline.py 的 peak_tflops
FP32_SIMT_TFLOPS = {"H100": 67.0, "H800": 67.0, "A100": 19.5, "H20": 44.0}


# --8<-- [start:version_spec]
@dataclass(frozen=True)
class KernelVersion:
    name: str
    desc: str
    # 片上复用的 tile 大小：一个从全局读进来的 A 元素被 tile_n 个输出用到，B 元素被 tile_m 个输出用到
    # v1/v2 没有显式复用（1x1）；v3 是 smem 的 32x32；v4 是 block tile 128x128；v5 是每个 warp 的 32x32
    tile_m: int
    tile_n: int
    elem_bytes: int  # 输入元素字节数：FP32 4，FP16 2
    tensor_core: bool
    threads_per_block: int
    regs_est: int  # 每线程寄存器数（估算，待 -Xptxas -v 实测替换）
    smem_bytes: int  # 每个 block 的静态 shared memory
    # 一个 block 负责的 C 小块边长：决定一共启动多少个 block（够不够填满 132 个 SM）
    block_tile: int = 32
    # 只对 v1/v2 有意义：每个输出、每个 k，一个 warp 平均摊到每个线程的 sector 字节数（见 sector_bytes_v1_v2）
    sector_bytes_per_k: float | None = None


def sector_bytes_v1_v2(coalesced: bool) -> float:
    """v1/v2 每算一次乘加（每个输出、每个 k），平均每个线程从 L2/DRAM 搬多少字节（无 L1 复用）。

    一个 warp 32 个线程算 32 个输出，每个 k 发两条 load：
    v1：A 的 32 个地址在 32 个不同的行 -> 32 个 sector；B 32 个线程同一地址 -> 1 个 sector。共 33 × 32 B / 32 = 33 B
    v2：A 同一地址 -> 1 个 sector；B 连续 128 字节 -> 4 个 sector。共 5 × 32 B / 32 = 5 B"""
    sectors = (1 + 4) if coalesced else (32 + 1)
    return sectors * 32 / 32


VERSIONS = [
    KernelVersion("v1", "naive（threadIdx.x -> 行）", 1, 1, 4, False, 1024, 32, 0, 32, sector_bytes_v1_v2(False)),
    KernelVersion("v2", "合并访存（threadIdx.x -> 列）", 1, 1, 4, False, 1024, 32, 0, 32, sector_bytes_v1_v2(True)),
    KernelVersion("v3", "smem 分块 32x32", 32, 32, 4, False, 1024, 32, 2 * 32 * 32 * 4, 32),
    KernelVersion("v4", "寄存器分块 128x128，每线程 8x8", 128, 128, 4, False, 256, 128, (128 * 8 + 8 * 128) * 4, 128),
    KernelVersion("v5", "WMMA 16x16x16，每 warp 32x32", 32, 32, 2, True, 128, 64, 0, 64),
]
# --8<-- [end:version_spec]


# --8<-- [start:traffic_model]
def global_reads_per_output(v: KernelVersion, k: int) -> float:
    """每个输出元素要从全局内存读几个数（线程级请求，不看合并）：K·(1/tile_m + 1/tile_n)。
    v1/v2：2K（A 一行 + B 一列）；v3：K/16；v4：K/64。"""
    return k * (1 / v.tile_m + 1 / v.tile_n)


def global_bytes(v: KernelVersion, m: int, n: int, k: int) -> float:
    """整个矩阵乘从 L2/DRAM 搬的字节数（不考虑 L1/L2 缓存复用，写 C 按 FP32 算）。
    tile 版本：A 被读 ceil(N/tile_n) 遍、B 被读 ceil(M/tile_m) 遍（和第 2 课 tile_traffic 同一个公式）。
    v1/v2：按 sector 粒度数，每个输出每个 k 搬 sector_bytes_per_k 字节。"""
    c_write = m * n * 4
    if v.sector_bytes_per_k is not None:
        return m * n * k * v.sector_bytes_per_k + c_write
    a_reads = m * k * math.ceil(n / v.tile_n)
    b_reads = k * n * math.ceil(m / v.tile_m)
    return v.elem_bytes * (a_reads + b_reads) + c_write


def roofline_cap_tflops(v: KernelVersion, m: int, n: int, k: int, hw: str = "H100") -> tuple[float, float, str]:
    """返回 (算术强度 FLOP/B, 理论上限 TFLOPS, 瓶颈)。上限 = min(算力峰值, 强度 × HBM 带宽)。"""
    flops = 2 * m * n * k
    intensity = flops / global_bytes(v, m, n, k)
    peak = HARDWARE[hw].peak_tflops if v.tensor_core else FP32_SIMT_TFLOPS[hw]
    mem_cap = intensity * HARDWARE[hw].bandwidth_gbs / 1e3  # FLOP/B × GB/s = GFLOP/s -> /1e3 得 TFLOPS
    return intensity, min(peak, mem_cap), "算力" if peak <= mem_cap else "带宽"
# --8<-- [end:traffic_model]


def n_blocks(v: KernelVersion, m: int, n: int) -> int:
    return math.ceil(m / v.block_tile) * math.ceil(n / v.block_tile)


def version_rows(m: int, n: int, k: int, hw: str = "H100") -> list[list]:
    rows = []
    for v in VERSIONS:
        intensity, cap, bound = roofline_cap_tflops(v, m, n, k, hw)
        occ = occupancy(v.threads_per_block, v.regs_est, v.smem_bytes, hw)
        rows.append([v.name, v.desc, f"{global_reads_per_output(v, k):.0f}", f"{global_bytes(v, m, n, k) / 1e6:,.0f}",
                     f"{intensity:.2f}", f"{cap:.1f}", bound, n_blocks(v, m, n), f"{occ.occupancy:.0%}"])
    return rows


def _occ_str(threads: int, regs: int, smem: int) -> str:
    if threads > 1024:
        return "无法启动（>1024 线程）"
    occ = occupancy(threads, regs, smem, "H100")
    return f"{occ.occupancy:.0%}" if occ.active_blocks else f"无法启动（{threads} 线程 × {regs} 寄存器 > 64K）"


def main() -> None:
    size = 4096
    rows = version_rows(size, size, size)

    # 旋钮 A：v4 的 block tile（TM=TN=8 不变，所以线程数 = (BM/8)·(BN/8)）
    rows_a = []
    for bm in [64, 128, 256]:
        threads = (bm // 8) ** 2
        smem = (bm * 8 + 8 * bm) * 4
        v = KernelVersion(f"v4-{bm}", "", bm, bm, 4, False, threads, 128, smem, bm)
        intensity, cap, bound = roofline_cap_tflops(v, size, size, size)
        blocks = n_blocks(v, size, size)
        rows_a.append([f"{bm}x{bm}", threads, f"{smem / 1024:.0f}", f"{bm * bm * 4 // threads}", f"{intensity:.0f}",
                       f"{cap:.0f}", bound, _occ_str(threads, 128, smem), blocks, f"{blocks / 132:.1f}"])

    # 旋钮 B：v3 的 tile（线程数 = tile²）
    rows_b = []
    for t in [8, 16, 32, 64]:
        v = KernelVersion(f"v3-{t}", "", t, t, 4, False, t * t, 32, 2 * t * t * 4, t)
        intensity, cap, bound = roofline_cap_tflops(v, size, size, size)
        rows_b.append([f"{t}x{t}", t * t, f"{2 * t * t * 4 / 1024:.1f}", f"{intensity:.0f}", f"{cap:.1f}",
                       _occ_str(t * t, 32, 2 * t * t * 4)])

    # 旋钮 C：decode 形状 M=16（batch=16 的一次 decode 里的线性层），N=K=4096
    rows_c = version_rows(16, size, size)

    headers = ["版本", "写法", "每个输出的全局读次数", "全局访存 MB（无缓存）", "算术强度 FLOP/B", "H100 上限 TFLOPS", "瓶颈",
               "block 数", "H100 理论 occupancy（寄存器为估算）"]
    report = (
        f"# CUDA 矩阵乘 5 个版本的模型（计算，H100：HBM {HARDWARE['H100'].bandwidth_gbs} GB/s，"
        f"FP32 {FP32_SIMT_TFLOPS['H100']} TFLOPS，FP16 Tensor Core {HARDWARE['H100'].peak_tflops} TFLOPS）\n\n"
        f"## M = N = K = {size}\n\n"
        + tabulate(rows, headers=headers, tablefmt="github")
        + "\n\n## 旋钮 A：v4 的 block tile（TM = TN = 8，寄存器按 128 估）\n\n"
        + tabulate(rows_a, headers=["block tile", "线程数", "smem KB", "每线程累加器字节", "算术强度", "H100 上限 TFLOPS", "瓶颈",
                                    "occupancy", "block 数", "block 数 / 132 SM"], tablefmt="github")
        + "\n\n## 旋钮 B：v3 的 tile 大小（每线程 1 个输出，线程数 = tile²）\n\n"
        + tabulate(rows_b, headers=["tile", "线程数", "smem KB", "算术强度", "H100 上限 TFLOPS", "occupancy"], tablefmt="github")
        + f"\n\n## 旋钮 C：decode 形状 M = 16，N = K = {size}\n\n"
        + tabulate(rows_c, headers=headers, tablefmt="github")
        + "\n"
    )
    print(report)
    OUT.write_text(report)
    print(f"已保存到 {OUT}")


if __name__ == "__main__":
    main()
