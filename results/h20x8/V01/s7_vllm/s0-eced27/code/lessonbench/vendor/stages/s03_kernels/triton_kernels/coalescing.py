"""合并访存（coalescing）演示：同样搬 n 个数，只改变"相邻线程读的地址相隔多远"。

1. strided_copy：dst[i] = src[i · STRIDE]。STRIDE=1 时一个 warp 读连续 128 字节（4 个 sector），完美合并；
   STRIDE >= 8（FP32）时每个线程独占一个 32 字节 sector，有效带宽降到 1/8。
2. gather_rows：out[r, :] = table[idx[r], :]，就是 embedding 查表。行是随机的，但行内 D 个数连续，
   所以只要 D 够大（每行 >= 128 字节），随机查表也能接近满带宽。

需要 NVIDIA GPU + triton。测试：tests/test_coalescing_gpu.py；benchmark：
    uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.triton_kernels.coalescing --hw H100
"""

import argparse
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
from tabulate import tabulate

from stages.s03_kernels.code.memory_access import strided_traffic
from tools.bench import benchmark
from tools.roofline import HARDWARE

EXP = Path(__file__).resolve().parents[1] / "experiments"


# --8<-- [start:strided_kernel]
@triton.jit
def strided_copy_kernel(src_ptr, dst_ptr, n, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    # 第几个 program；一个 program 负责输出里连续的 BLOCK 个元素
    pid = tl.program_id(axis=0)
    # 本 program 的输出下标 [BLOCK]：pid·BLOCK, pid·BLOCK+1, ...
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # [BLOCK]
    # 最后一个 program 可能越界；mask=False 的位置不读也不写
    mask = offs < n
    # 读地址 = offs · STRIDE：相邻元素在 src 里相隔 STRIDE·4 字节
    # 转 int64 再乘：n·STRIDE 超过 2^31 时 int32 会溢出，指针算错不会报错，只会读到别处
    # Triton 会把 BLOCK 个元素分给 num_warps 个 warp，相邻线程拿相邻的 offs
    # 所以 STRIDE=1：一个 warp 的 32 个地址连续 -> 4 个 sector；STRIDE=8：32 个 sector
    x = tl.load(src_ptr + offs.to(tl.int64) * STRIDE, mask=mask)  # [BLOCK]
    # 写是连续的，永远合并；差别全在读
    tl.store(dst_ptr + offs, x, mask=mask)
# --8<-- [end:strided_kernel]


def strided_copy(src: torch.Tensor, n: int, stride: int, block: int = 1024) -> torch.Tensor:
    """dst[i] = src[i · stride]，i < n。src 必须连续且至少有 (n-1)·stride+1 个元素。"""
    assert src.is_cuda and src.is_contiguous() and src.numel() >= (n - 1) * stride + 1
    dst = torch.empty(n, device=src.device, dtype=src.dtype)
    # cdiv(n, block) 个 program；n = 2^24、block = 1024 时 16384 个，足够填满 132 个 SM
    grid = (triton.cdiv(n, block),)
    strided_copy_kernel[grid](src, dst, n, STRIDE=stride, BLOCK=block)
    return dst


# --8<-- [start:gather_kernel]
@triton.jit
def gather_rows_kernel(table_ptr, idx_ptr, out_ptr, D, BLOCK_D: tl.constexpr):
    # 一个 program 搬一行：第 r 个输出行
    r = tl.program_id(axis=0)
    # 先读这一行要查的表行号（标量）；随机的行号让不同 program 的地址毫无规律
    row = tl.load(idx_ptr + r).to(tl.int64)
    # 行内 D 个数是连续的 [BLOCK_D]，BLOCK_D 是 >= D 的 2 的幂
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < D
    # 行内连续 -> 合并；D=4（16 字节）时一个 sector 只用一半，D>=8（32 字节）起每个 sector 都装满，
    # D>=32（128 字节）时一行正好是整条 cache line，每次请求取回的有用字节也够多
    x = tl.load(table_ptr + row * D + offs_d, mask=mask)  # [BLOCK_D]
    tl.store(out_ptr + r * D + offs_d, x, mask=mask)
# --8<-- [end:gather_kernel]


def gather_rows(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """out[r] = table[idx[r]]，table: [V, D] 连续，idx: [R] int64 -> out: [R, D]"""
    assert table.is_cuda and table.is_contiguous() and table.dim() == 2
    v, d = table.shape
    out = torch.empty(idx.numel(), d, device=table.device, dtype=table.dtype)
    gather_rows_kernel[(idx.numel(),)](table, idx, out, d, BLOCK_D=triton.next_power_of_2(d))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hw", default="H100", choices=sorted(HARDWARE))
    args = p.parse_args()
    hw = HARDWARE[args.hw]
    dev = torch.device("cuda")

    # 1. 跨步拷贝：有效数据量固定 n = 2^24 个 FP32（64 MB），源数组 2 GB
    n, max_stride = 1 << 24, 32
    src = torch.randn(n * max_stride, device=dev)
    rows = []
    for s in [1, 2, 4, 8, 16, 32]:
        ms = benchmark(lambda: strided_copy(src, n, s), dev).median_ms
        model = strided_traffic(n, s) + n * 4  # 按 32 字节 sector 计的读 + 连续写
        rows.append([s, f"{ms:.3f}", f"{2 * n * 4 / ms / 1e6:.0f}", f"{model / 1e6:.0f}",
                     f"{model / ms / 1e6:.0f}", f"{model / (hw.bandwidth_gbs * 1e9) * 1e3:.3f}"])

    # 2. 随机行 gather：R = 2^20 行，表 V = 2^20 行，D 从 4 到 256
    rows_g = []
    r_cnt = 1 << 20
    for d in [4, 8, 32, 128, 256]:
        table = torch.randn(1 << 20, d, device=dev)
        idx = torch.randint(0, table.shape[0], (r_cnt,), device=dev)
        ms = benchmark(lambda: gather_rows(table, idx), dev).median_ms
        # 有效字节：读下标 + 读 R·D·4 + 写 R·D·4；sector 模型：每行读 ceil(D·4/32) 个 sector
        useful = r_cnt * (8 + 2 * d * 4)
        model = r_cnt * (8 + math.ceil(d * 4 / 32) * 32 + d * 4)
        rows_g.append([d, d * 4, f"{ms:.3f}", f"{useful / ms / 1e6:.0f}", f"{model / ms / 1e6:.0f}"])

    report = (
        f"# 合并访存 benchmark（{torch.cuda.get_device_name()}，roofline 按 {hw.name} {hw.bandwidth_gbs} GB/s）\n\n"
        "## 跨步拷贝 n = 2^24 FP32\n\n"
        + tabulate(rows, headers=["步长", "实测 ms", "有效 GB/s", "sector 模型 MB", "按模型折算 GB/s", "roofline ms"],
                   tablefmt="github")
        + "\n\n## 随机行 gather，R = 2^20\n\n"
        + tabulate(rows_g, headers=["D", "每行字节", "实测 ms", "有效 GB/s", "按 sector 模型折算 GB/s"], tablefmt="github")
        + "\n"
    )
    print(report)
    EXP.mkdir(exist_ok=True)
    (EXP / f"coalescing_{args.hw}.md").write_text(report)


if __name__ == "__main__":
    main()
