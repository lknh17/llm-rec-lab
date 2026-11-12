"""Triton 入门：向量加法。理解 Triton 的编程模型——"一个 program 处理一块数据"。

CUDA 是"每个线程处理一个元素"，Triton 是"每个 program（≈ 一个线程块）处理一个 BLOCK 的向量"，
块内的向量化、合并访存（coalescing）由编译器负责。
需要 NVIDIA GPU + triton（Linux）。
"""

import torch
import triton
import triton.language as tl


# --8<-- [start:kernel]
# --8<-- [start:add_kernel]
# @triton.jit：把函数编译成 GPU kernel；调用时 GPU 上会同时跑很多个 program，每个都执行一遍函数体
# x_ptr / y_ptr / out_ptr：调用时传 torch 张量，Triton 自动换成指向首元素的指针
# BLOCK: tl.constexpr：编译期常量，每个不同的值编译一份 kernel（tl.arange 的长度必须是常量）
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # tl.program_id(axis=0)：本 program 在 grid 第 0 维的编号，0..grid[0]-1
    # 它是区分各个 program 的唯一依据，相当于 CUDA 的 blockIdx.x
    pid = tl.program_id(axis=0)  # 第几个 program
    # 第 pid 个 program 负责 [pid·BLOCK, pid·BLOCK + BLOCK) 这一段
    # tl.arange(0, BLOCK) 是 [0..BLOCK-1] 的向量，加上起点得到 BLOCK 个全局下标
    # 例：BLOCK=1024、pid=3 时 offsets = [3072, 3073, ..., 4095]
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # [BLOCK]
    # n 不一定是 BLOCK 的整数倍：n=5000 时最后一个 program 的 offsets 到 5119，后 120 个越界
    mask = offsets < n_elements  # 最后一块可能越界
    # 指针 + 下标向量 = BLOCK 个地址（单位是元素，不是字节）
    # tl.load 带 mask：越界位置不访问内存（不加 mask 会读到别人的显存甚至非法地址）
    # 没给 other 时，越界位置的值未定义，但后面 store 也带同一个 mask，不会写出去
    # HBM -> 寄存器；地址连续，一个 warp 的 32 个线程读相邻地址 = 合并访存
    x = tl.load(x_ptr + offsets, mask=mask)  # [BLOCK]
    y = tl.load(y_ptr + offsets, mask=mask)  # [BLOCK]
    # x + y 是 [BLOCK] 向量的逐元素加法，在寄存器里完成；结果写回 HBM
    # 每个元素读 2 次写 1 次（FP16 共 6 字节），只做 1 次加法：强度 1/6 ≈ 0.17 FLOP/B
    tl.store(out_ptr + offsets, x + y, mask=mask)  # 寄存器 -> HBM
# --8<-- [end:add_kernel]


# --8<-- [start:add_launch]
def add(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    # 要求连续内存：kernel 把张量当成一维数组按 offsets 寻址，不连续就会读错位置
    assert x.is_cuda and x.shape == y.shape and x.is_contiguous() and y.is_contiguous()
    # 分配输出，不初始化
    out = torch.empty_like(x)
    n = x.numel()
    # cdiv = 向上取整的除法：n=5000、block=1024 时启动 5 个 program
    grid = (triton.cdiv(n, block),)  # 启动多少个 program
    # kernel[grid](...)：Triton 的启动语法，grid 是元组，几维就是几维的 program 网格
    add_kernel[grid](x, y, out, n, BLOCK=block)
    return out
# --8<-- [end:add_launch]
# --8<-- [end:kernel]
