"""S3 第 4 课：访存模型（纯计算）+ Mac 上的连续读 / 跨步读 / 随机 gather 实验。

1. warp_sectors / strided_traffic：一个 warp 的 32 个地址落在几个 32 字节 sector 里，决定"读 4 字节要搬多少字节"
2. bank_conflict_ways：一个 warp 访问 shared memory 时有几路 bank conflict（32 个 bank，每个 4 字节宽）
3. Mac 实测：固定有效数据量，改变步长，看 MPS 上的有效带宽怎么掉
运行：uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.code.memory_access
"""

from pathlib import Path

import torch
from tabulate import tabulate

from tools.bench import benchmark
from tools.device import describe, get_device

OUT = Path(__file__).resolve().parents[1] / "experiments" / "memory_access.md"
WARP = 32
SECTOR = 32  # NVIDIA L1/L2 与 HBM 之间的最小搬运单位：32 字节
N_BANKS = 32  # shared memory 有 32 个 bank


# --8<-- [start:warp_sectors]
def warp_sectors(byte_addrs: list[int], sector: int = SECTOR) -> int:
    """一个 warp 的一条访存指令（每个线程一个地址）会触发多少个 sector 的传输。

    硬件把 32 个线程的地址合并（coalesce）：落在同一个 32 字节 sector 里的只搬一次。
    所以 32 个线程读连续的 4 字节 = 128 字节 = 4 个 sector，是最好情况；
    32 个线程各读一个相距很远的地址 = 32 个 sector，搬了 1024 字节只用 128 字节。"""
    # 地址整除 sector 大小，就是它所在 sector 的编号；去重后数个数
    return len({a // sector for a in byte_addrs})


def coalescing_efficiency(byte_addrs: list[int], elem_bytes: int = 4, sector: int = SECTOR) -> float:
    """有效字节 / 实际搬运字节。1.0 表示完美合并；广播（所有线程读同一地址）按去重后的有效字节算。"""
    useful = len(set(byte_addrs)) * elem_bytes
    return useful / (warp_sectors(byte_addrs, sector) * sector)
# --8<-- [end:warp_sectors]


# --8<-- [start:strided_traffic]
def strided_traffic(n: int, stride: int, elem_bytes: int = 4, sector: int = SECTOR) -> int:
    """读 n 个元素、相邻元素相隔 stride 个元素时，从 DRAM/L2 实际搬运的字节数（不考虑缓存复用）。

    一个 sector 能装 sector/elem_bytes 个元素（FP32 时 8 个）：
    - stride 小于 8：相邻几个有效元素挤在同一个 sector 里，搬运量 = n·elem·stride（中间没用的也搬了）
    - stride 大于等于 8：每个有效元素独占一个 sector，搬运量 = n·32 字节，再大的步长也不会更差"""
    elems_per_sector = sector // elem_bytes
    return n * elem_bytes * min(stride, elems_per_sector)
# --8<-- [end:strided_traffic]


# --8<-- [start:bank_conflict]
def bank_conflict_ways(word_addrs: list[int], n_banks: int = N_BANKS) -> int:
    """一个 warp 读 shared memory 时的冲突路数：1 表示无冲突，32 表示完全串行。

    shared memory 按 4 字节一个 word 交错分到 32 个 bank：word w 在 bank w % 32。
    每个 bank 每周期只能出一个 word；同一个 bank 里不同的 word 要排队。
    多个线程读同一个 word 是广播（broadcast），不算冲突。"""
    words_in_bank: dict[int, set[int]] = {}
    for w in word_addrs:
        words_in_bank.setdefault(w % n_banks, set()).add(w)
    # 最挤的那个 bank 里有几个不同的 word，这次访问就要串行几轮
    return max(len(ws) for ws in words_in_bank.values())


def tile_column_words(tile_cols: int, col: int = 0, swizzle: bool = False) -> list[int]:
    """32 个线程读一个 [32, tile_cols] 的 FP32 tile 的第 col 列（线程 i 读第 i 行）时访问的 word 地址。

    tile_cols=32：每行 32 个 word，第 i 行第 0 列在 word 32·i -> 全部落在 bank 0 -> 32 路冲突
    tile_cols=33（padding 一列）：word 33·i -> bank i -> 无冲突，代价是多占 1/32 的 smem
    swizzle：不加 padding，把列号和行号做异或，第 i 行第 c 列存到 (c ^ i) 列 -> 一列散到 32 个 bank"""
    words = []
    for row in range(WARP):
        c = (col ^ row) % tile_cols if swizzle else col
        words.append(row * tile_cols + c)
    return words
# --8<-- [end:bank_conflict]


def _gbps(nbytes: float, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


# --8<-- [start:mac_experiment]
def run_mac_experiment(n: int = 1 << 23, strides=(1, 2, 4, 8, 16, 32, 64)) -> tuple[list, list]:
    """固定"有效数据量" n 个 FP32（32 MB），只改变读的步长：有效工作量不变，变的只是访存模式。"""
    device = get_device()
    max_stride = max(strides)
    # 源数组 n·64 个 FP32 = 2 GB，所有步长共用；步长 s 时只读其中每隔 s 个的一个
    src = torch.randn(n * max_stride, device=device)  # [n·64]
    dst = torch.empty(n, device=device)  # [n]

    rows = []
    for s in strides:
        # src[: n*s : s] 是一个步长为 s 的视图（不复制），copy_ 把它搬进连续的 dst
        view = src[: n * s : s]  # [n]，stride=s
        ms = benchmark(lambda: dst.copy_(view), device, warmup=5, iters=20).median_ms
        # 有效字节：读 n·4 + 写 n·4；模型字节：按 32 字节 sector 计的读 + 写
        useful = 2 * n * 4
        model = strided_traffic(n, s) + n * 4
        # 同一个模型换成 64 字节粒度：用来检验 Apple GPU 的实际搬运粒度是不是更大
        model64 = strided_traffic(n, s, sector=64) + n * 4
        rows.append([s, f"{ms:.3f}", f"{_gbps(useful, ms):.0f}", f"{model / 1e6:.0f}", f"{_gbps(model, ms):.0f}",
                     f"{_gbps(model64, ms):.0f}"])

    # gather：同样读 n 个元素，下标分别是"按步长 64 排好序"和"完全随机"，覆盖同样 2 GB 的范围
    sorted_idx = torch.arange(n, device=device) * max_stride  # [n]
    random_idx = torch.randint(0, n * max_stride, (n,), device=device)  # [n]
    rows_g = []
    for label, idx in [("连续下标 0..n-1", torch.arange(n, device=device)), ("有序、步长 64", sorted_idx), ("随机", random_idx)]:
        # index_select 就是 embedding 查表的本质：按下标从大表里取行（这里每行 1 个数）
        ms = benchmark(lambda: torch.index_select(src, 0, idx, out=dst), device, warmup=5, iters=20).median_ms
        # 有效字节：读下标 8 字节（int64）+ 读值 4 字节 + 写 4 字节
        rows_g.append([label, f"{ms:.3f}", f"{_gbps(n * 16, ms):.0f}"])
    return rows, rows_g
# --8<-- [end:mac_experiment]


def main() -> None:
    device = get_device()
    rows, rows_g = run_mac_experiment()

    patterns = {
        "连续 FP32（线程 i 读第 i 个）": [4 * i for i in range(WARP)],
        "步长 2": [8 * i for i in range(WARP)],
        "步长 8（每线程跨 32 字节）": [32 * i for i in range(WARP)],
        "全部读同一个地址（广播）": [0] * WARP,
        "连续 FP16": [2 * i for i in range(WARP)],
        "每线程读 16 字节向量（float4）": [16 * i + j for i in range(WARP) for j in range(0, 16, 4)],
    }
    rows_w = []
    for name, addrs in patterns.items():
        elem = 2 if "FP16" in name else 4
        rows_w.append([name, len(set(addrs)) * elem, warp_sectors(addrs), warp_sectors(addrs) * SECTOR,
                       f"{coalescing_efficiency(addrs, elem):.1%}"])
    rows_b = [
        ["[32, 32] tile 读一列", bank_conflict_ways(tile_column_words(32))],
        ["[32, 33] tile 读一列（padding）", bank_conflict_ways(tile_column_words(33))],
        ["[32, 32] tile 读一列（XOR swizzle）", bank_conflict_ways(tile_column_words(32, swizzle=True))],
        ["[32, 32] tile 读一行", bank_conflict_ways([5 * 32 + i for i in range(WARP)])],
        ["[32, 64] tile 读一列", bank_conflict_ways(tile_column_words(64))],
    ]

    report = (
        f"# 访存模式实验\n\n设备：{describe(device)}\n\n"
        "## 1. 一个 warp 的一条访存指令触发几个 32 字节 sector（计算）\n\n"
        + tabulate(rows_w, headers=["访问模式", "有效字节", "sector 数", "实际搬运字节", "合并效率"], tablefmt="github")
        + "\n\n## 2. shared memory bank conflict 路数（计算）\n\n"
        + tabulate(rows_b, headers=["访问模式", "冲突路数"], tablefmt="github")
        + "\n\n## 3. Mac 实测：跨步拷贝，有效数据量固定 32 MB（FP32，n = 2^23）\n\n"
        + tabulate(rows, headers=["步长", "ms", "有效带宽 GB/s", "32B sector 模型搬运 MB", "按 32B 模型折算 GB/s",
                                  "按 64B 粒度折算 GB/s"], tablefmt="github")
        + "\n\n## 4. Mac 实测：gather（index_select），n = 2^23 个 FP32，源表 2 GB\n\n"
        + tabulate(rows_g, headers=["下标", "ms", "有效带宽 GB/s"], tablefmt="github")
        + "\n"
    )
    print(report)
    OUT.write_text(report)
    print(f"已保存到 {OUT}")


if __name__ == "__main__":
    main()
