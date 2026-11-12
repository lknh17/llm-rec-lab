"""Occupancy 计算器：纯计算，不需要 GPU。和 NVIDIA CUDA Occupancy Calculator（cudaOccupancy.h）同一套规则。

一个 SM 上能同时驻留多少个线程块（block），由四种资源里最紧的那一种决定：
    每 SM 活跃 block 数 = min(按 warp/线程数上限, 按寄存器, 按 shared memory, 按 block 数上限)
occupancy = 每 SM 活跃 warp 数 / 每 SM 最多 warp 数（Hopper、Ampere 都是 64）。

用法：
    uv run python -m tools.occupancy --gpu H100 --threads 256 --regs 64
    uv run python -m tools.occupancy --gpu A100 --threads 256 --regs 32 --smem 49152
    uv run python -m tools.occupancy --gpu H100 --threads 256 --sweep-regs 32 64 128 255
寄存器数从哪来：编译时加 `-Xptxas -v`（或 nvcc `--resource-usage`）会打印 "Used N registers"；
Triton 可以看 `kernel.n_regs`；ncu 的 Launch Statistics 里有 "Registers Per Thread"。
"""

import argparse
import math
from dataclasses import dataclass


# --8<-- [start:gpu_spec]
@dataclass(frozen=True)
class SmSpec:
    """一个 SM 的资源上限。数值来自 CUDA 编程指南的 "Compute Capabilities" 表和各卡白皮书。"""

    name: str
    # 整卡 SM 数：只影响"整卡能同时跑多少 block"，不影响单个 SM 的 occupancy
    sm_count: int
    # 每 SM 最多驻留的线程数；2048 线程 = 64 个 warp
    max_threads_per_sm: int = 2048
    max_warps_per_sm: int = 64
    # 每 SM 最多驻留的 block 数：block 太小（比如 32 线程）时它会先卡住
    max_blocks_per_sm: int = 32
    # 寄存器文件：64K 个 32 位寄存器 = 256 KB，平分给 4 个 SM 子分区（SMSP），每个 16K
    regs_per_sm: int = 65536
    sub_partitions: int = 4
    # 单线程最多 255 个寄存器，超过的部分溢出（spill）到 local memory（实际在 HBM，经 L1/L2 缓存）
    max_regs_per_thread: int = 255
    # 寄存器按 warp 分配，粒度 256 个：每线程寄存器数实际上向上取整到 8 的倍数
    reg_alloc_unit: int = 256
    # shared memory：每 SM 可配置的最大容量（和 L1 共用一块 SRAM，剩下的给 L1）
    smem_per_sm: int = 228 * 1024
    # 单个 block 最多能申请的 shared memory（动态 smem 要 cudaFuncSetAttribute 才能超过 48 KB）
    max_smem_per_block: int = 227 * 1024
    # sm_80 以后，系统给每个 block 额外预留 1 KB shared memory，申请 0 字节也占 1 KB
    smem_reserved_per_block: int = 1024
    # shared memory 分配粒度 128 字节
    smem_alloc_unit: int = 128
    warp_size: int = 32


GPUS = {
    # A100（sm_80）：每 SM 最多 164 KB shared memory，单 block 最多 163 KB
    "A100": SmSpec("A100 SXM", 108, smem_per_sm=164 * 1024, max_smem_per_block=163 * 1024),
    # H100 SXM（sm_90）：132 个 SM，每 SM 228 KB shared memory
    "H100": SmSpec("H100 SXM", 132),
    # H800 和 H100 是同一颗芯片，单 SM 资源完全一样（只砍了 NVLink）
    "H800": SmSpec("H800 SXM", 132),
    # H20 也是 Hopper（sm_90），单 SM 资源同 H100，只是只开了 78 个 SM
    "H20": SmSpec("H20", 78),
}
# --8<-- [end:gpu_spec]


@dataclass
class OccupancyResult:
    gpu: str
    threads_per_block: int
    regs_per_thread: int
    smem_per_block: int
    warps_per_block: int
    # 四种资源各自允许的每 SM block 数
    limits: dict[str, int]
    active_blocks: int
    active_warps: int
    occupancy: float
    # 起决定作用的资源（可能并列）
    limiters: list[str]

    def report(self) -> str:
        lim = "、".join(self.limiters)
        detail = "，".join(f"{k} {v}" for k, v in self.limits.items())
        return (f"[{self.gpu}] {self.threads_per_block} 线程/block、{self.regs_per_thread} 寄存器/线程、"
                f"{self.smem_per_block / 1024:.1f} KB smem/block -> 每 SM {self.active_blocks} 个 block、"
                f"{self.active_warps} 个 warp，occupancy {self.occupancy:.1%}，限制因素：{lim}（各资源允许的 block 数：{detail}）")


def _round_up(x: int, unit: int) -> int:
    return math.ceil(x / unit) * unit


# --8<-- [start:occupancy]
def occupancy(threads_per_block: int, regs_per_thread: int, smem_per_block: int = 0, gpu: str | SmSpec = "H100") -> OccupancyResult:
    """计算一个 kernel 配置在一个 SM 上的理论 occupancy。smem_per_block 单位是字节（静态 + 动态）。"""
    spec = GPUS[gpu] if isinstance(gpu, str) else gpu
    if not 1 <= threads_per_block <= 1024:
        raise ValueError("每个 block 的线程数必须在 1 到 1024 之间")
    if not 1 <= regs_per_thread <= spec.max_regs_per_thread:
        raise ValueError(f"每线程寄存器数必须在 1 到 {spec.max_regs_per_thread} 之间（超出部分编译器会 spill）")

    # 线程按 warp 分配：100 个线程也占 4 个 warp，最后一个 warp 有 28 个线程空转
    warps_per_block = math.ceil(threads_per_block / spec.warp_size)

    # 限制 1：warp 数。64 个 warp 槽位能装几个 block
    by_warps = spec.max_warps_per_sm // warps_per_block
    # 限制 2：block 数硬上限（32）
    by_blocks = spec.max_blocks_per_sm
# --8<-- [end:occupancy]

    # --8<-- [start:occupancy_regs]
    # 限制 3：寄存器。一个 warp 要 regs × 32 个寄存器，向上取整到 256 的倍数
    # 例：64 寄存器 -> 2048 个/warp；255 寄存器 -> 8160 取整到 8192
    regs_per_warp = _round_up(regs_per_thread * spec.warp_size, spec.reg_alloc_unit)
    # 寄存器在 4 个子分区里各自分配，一个 warp 只能用它所在子分区的 16K 个
    # 所以先算每个子分区放几个 warp（向下取整），再乘 4；这比直接 65536 / regs_per_warp 更准
    # 例：40 寄存器 -> 1280/warp -> 每子分区 12 个 -> 48 个 warp（直接除会得到 51）
    regs_per_sub = spec.regs_per_sm // spec.sub_partitions
    warps_by_regs = (regs_per_sub // regs_per_warp) * spec.sub_partitions
    # 整个 block 的 warp 必须同时驻留，所以再按 block 取整
    # 结果为 0 表示这个 block 根本发射不了（"too many resources requested for launch"）
    by_regs = warps_by_regs // warps_per_block
    # --8<-- [end:occupancy_regs]

    # --8<-- [start:occupancy_smem]
    # 限制 4：shared memory。每个 block 实际占用 = 申请量 + 1 KB 系统预留，再按 128 字节取整
    if smem_per_block > spec.max_smem_per_block:
        by_smem = 0
    else:
        smem_cost = _round_up(smem_per_block + spec.smem_reserved_per_block, spec.smem_alloc_unit)
        by_smem = spec.smem_per_sm // smem_cost

    limits = {"warp 数": by_warps, "寄存器": by_regs, "shared memory": by_smem, "block 数上限": by_blocks}
    # 四个限制取最小：最紧的那种资源决定能驻留几个 block
    active_blocks = min(limits.values())
    active_warps = active_blocks * warps_per_block
    limiters = [k for k, v in limits.items() if v == active_blocks]
    return OccupancyResult(spec.name, threads_per_block, regs_per_thread, smem_per_block, warps_per_block,
                           limits, active_blocks, active_warps, active_warps / spec.max_warps_per_sm, limiters)
    # --8<-- [end:occupancy_smem]


def max_regs_for_occupancy(threads_per_block: int, target_warps: int, gpu: str = "H100") -> int:
    """反过来问：想让每 SM 至少有 target_warps 个活跃 warp，每线程最多能用几个寄存器（只看寄存器限制）。
    对应 CUDA 的 __launch_bounds__(maxThreads, minBlocks) 和 Triton 的 num_warps 取舍。"""
    best = 0
    for r in range(8, GPUS[gpu].max_regs_per_thread + 1):
        res = occupancy(threads_per_block, r, 0, gpu)
        if res.limits["寄存器"] * res.warps_per_block >= target_warps:
            best = r
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Occupancy 计算器（纯计算）")
    p.add_argument("--gpu", default="H100", choices=sorted(GPUS))
    p.add_argument("--threads", type=int, default=256, help="每个 block 的线程数")
    p.add_argument("--regs", type=int, default=32, help="每线程寄存器数")
    p.add_argument("--smem", type=int, default=0, help="每个 block 的 shared memory 字节数")
    p.add_argument("--sweep-regs", nargs="+", type=int, help="依次计算多个寄存器数")
    args = p.parse_args()
    for r in args.sweep_regs or [args.regs]:
        print(occupancy(args.threads, r, args.smem, args.gpu).report())


if __name__ == "__main__":
    main()
