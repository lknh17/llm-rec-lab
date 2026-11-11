"""融合 softmax：每行只从 HBM 读一次、写一次（naive 写法要 7 次读写，见 code/fusion_bench.py）。

前提：一整行能放进一个 program 的寄存器/SRAM（列数 ≤ 几万）。更长的行要用 online softmax 分块（见 flash_attn.py）。
"""

import torch
import triton
import triton.language as tl


# --8<-- [start:kernel]
# @triton.jit：把下面这个函数编译成 GPU kernel。函数体只能用 Triton 支持的 Python 子集，
# 在 CPU 上不会真的执行这些行，而是在每个 program 里各执行一遍
# 参数里的 out_ptr / in_ptr 是指针：调用时传 torch 张量，Triton 自动换成"首元素地址"
# BLOCK: tl.constexpr 表示编译期常量：每个不同的 BLOCK 值单独编译一个 kernel，
# tl.arange 的长度必须是编译期常量，所以 BLOCK 只能这样传
@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
    # --8<-- [start:softmax_load]
    # tl.program_id(0)：本 program 在 grid 第 0 维的编号。grid=(R,) 时取值 0..R-1，
    # 正好让一个 program 处理一行；R=8192 行就并行启动 8192 个 program
    row = tl.program_id(0)  # 标量，行号
    # tl.arange(0, BLOCK)：[0, 1, ..., BLOCK-1] 的向量，BLOCK 必须是 2 的幂
    # 它不是循环：整个向量由这个 program 里的几十到几百个线程分摊，一次处理完
    cols = tl.arange(0, BLOCK)  # [BLOCK]
    # BLOCK 向上取到 2 的幂，可能比真实列数大（例如 C=3000 时 BLOCK=4096），多出来的位置要屏蔽
    mask = cols < n_cols  # [BLOCK] 布尔
    # 指针运算以"元素"为单位，不是字节：第 row 行首地址 = in_ptr + row * 行步长
    # 再加上 cols 得到 [BLOCK] 个地址；同一行内地址连续，一个 warp 读连续内存 = 合并访存
    # tl.load(地址, mask, other)：mask 为 False 的位置不访问内存，直接填 other
    # other=-inf 是精心选的：-inf 不影响 max，exp(-inf)=0 也不影响 sum
    # .to(tl.float32)：读进寄存器后立刻转 FP32，后面的计算都在 FP32 里做，不经过 HBM
    # 这是整个 kernel 唯一一次读 HBM：C=4096、FP16 时每行 8 KB
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf")).to(tl.float32)  # [BLOCK]
    # --8<-- [end:softmax_load]
    # --8<-- [start:softmax_compute]
    # tl.max(x, axis=0)：把 [BLOCK] 归约成一个标量。归约在 program 内部完成
    # （线程间用 warp shuffle / 共享内存交换），不写 HBM。减去最大值后 exp 最大是 1，不会溢出
    x = x - tl.max(x, axis=0)  # [BLOCK]
    # 逐元素 exp，结果留在寄存器；被屏蔽的位置是 exp(-inf)=0
    num = tl.exp(x)  # [BLOCK]
    # 行内求和，同样是 program 内归约
    den = tl.sum(num, axis=0)  # 标量
    # 标量广播到 [BLOCK]
    y = num / den  # [BLOCK] FP32
    # out_ptr.dtype.element_ty：输出指针指向的元素类型（例如 fp16），写回前转回去
    # mask 保证越界的那些位置不写；这是唯一一次写 HBM。读 1 次 + 写 1 次 = 2 次
    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)
    # --8<-- [end:softmax_compute]
# --8<-- [end:kernel]


# --8<-- [start:launch]
def softmax(x: torch.Tensor) -> torch.Tensor:
    """x: [R, C]，对最后一维做 softmax。"""
    # 只支持 CUDA 上的 2 维张量，且每行内部连续（stride(1)==1），否则 cols 的地址不连续
    assert x.is_cuda and x.ndim == 2 and x.stride(1) == 1
    r, c = x.shape
    # empty_like：分配同形状同 dtype 的显存，不初始化（kernel 会写满每个位置）
    out = torch.empty_like(x)  # [R, C]
    # Triton 的 tl.arange 要求 2 的幂：C=4096 时 BLOCK=4096，C=3000 时也是 4096
    block = triton.next_power_of_2(c)  # 标量
    # num_warps：每个 program 用几个 warp（1 warp = 32 线程）。BLOCK=4096 时下面选 8 warp，
    # 每个线程在寄存器里持有 4096/256 = 16 个 FP32 数（若只用 4 warp 就是 32 个）
    num_warps = 4 if block <= 2048 else (8 if block <= 8192 else 16)  # 行越长，用越多线程并行归约
    # kernel[grid](参数...) 是 Triton 的启动语法：grid=(r,) 启动 r 个 program
    # 行步长 x.stride(0) 显式传进去，这样带 padding 的行（stride > C）也能正确寻址
    softmax_kernel[(r,)](out, x, x.stride(0), out.stride(0), c, BLOCK=block, num_warps=num_warps)
    return out
# --8<-- [end:launch]
