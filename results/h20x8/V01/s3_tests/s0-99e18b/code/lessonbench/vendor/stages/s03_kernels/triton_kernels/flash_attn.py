"""FlashAttention 前向（Triton）：与 code/flash_attention_ref.py 同一个算法，但每块的中间结果都在片上。

一个 program 负责一个 (batch, head) 的一个 Q 块 [BLOCK_M, D]：
  Q 块常驻寄存器 -> 循环读 K/V 块 -> s = Q·K^T（SRAM 里的小块）-> online softmax 更新 m, l, acc -> 最后写 O。
HBM 上永远不出现 [T, T] 矩阵。限制：D ∈ {16, 32, 64, 128}，q/k/v 形状 [B, H, T, D] 且最后一维连续。
"""

import torch
import triton
import triton.language as tl


# --8<-- [start:flash_signature]
# 参数分四组：
#   Q, K, V, O：四个 [B, H, T, D] 张量的首元素指针
#   stride_xb / stride_xh / stride_xt：张量 x 沿 B、H、T 维走一步跨多少个元素
#     （最后一维 D 要求连续，步长恒为 1，所以不传）。连续的 [B, H, T, D] 时 stride_xt = D
#   H, T, sm_scale：运行期标量，sm_scale = 1/sqrt(D)
#   D / BLOCK_M / BLOCK_N / CAUSAL：tl.constexpr 编译期常量。块形状必须编译期已知；
#     CAUSAL 设成常量后，下面的 if CAUSAL 在编译期就被裁掉，不产生运行期分支
@triton.jit
def flash_fwd_kernel(Q, K, V, O,
                     stride_qb, stride_qh, stride_qt,
                     stride_kb, stride_kh, stride_kt,
                     stride_vb, stride_vh, stride_vt,
                     stride_ob, stride_oh, stride_ot,
                     H, T, sm_scale,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    # --8<-- [end:flash_signature]
    # --8<-- [start:kernel_body]
    # --8<-- [start:flash_load_q]
    # 二维 grid = (cdiv(T, BLOCK_M), B·H)：第 0 维是"第几个 Q 块"，第 1 维是"哪个 (batch, head)"
    # T=4096、BLOCK_M=64、B=1、H=32 时共 64 × 32 = 2048 个 program，足够填满 132 个 SM
    pid_m = tl.program_id(0)  # Q 块编号
    bh = tl.program_id(1)  # (b, h) 拍平后的编号
    # 把拍平的编号拆回 batch 和 head
    b = bh // H
    h = bh % H
    # 本 program 负责的 BLOCK_M 个 query 行号（全局行号，可能超过 T，靠 mask 处理）
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    # D 方向的偏移 0..D-1；D 是 constexpr 且必须是 2 的幂（16/32/64/128）
    offs_d = tl.arange(0, D)  # [D]

    # 定位到 Q[b, h] 这一片的起点，再加上二维偏移：
    # offs_m[:, None] 是 [BLOCK_M, 1]，offs_d[None, :] 是 [1, D]，广播成 [BLOCK_M, D] 的地址块
    # D 维步长是 1，所以 offs_d 直接加，不乘步长
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :]  # [BLOCK_M, D]
    # 行号 ≥ T 的行不读，填 0（这些行的结果最后也不会写出去）
    # 这块 Q 整个循环期间都留在寄存器里：64×128 FP16 = 16 KB，只从 HBM 读这一次
    q = tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0)  # [BLOCK_M, D]
    # --8<-- [end:flash_load_q]

    # --8<-- [start:flash_init]
    # online softmax 的三个统计量，全部 FP32 放寄存器：每行最大值、每行 exp 和、未归一化输出
    # tl.full(形状, 值, dtype) 填常数；m_i 初值 -inf 表示"还没看到任何分数"
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)  # [BLOCK_M]
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)  # [BLOCK_M]
    # acc 64×128 FP32 = 32 KB 寄存器；块再大就可能溢出到 local memory（register spill）
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)  # [BLOCK_M, D]

    # 因果：第 pid_m 块的最后一行是 (pid_m+1)·BLOCK_M - 1，它最多只看到这一列
    # 所以 K/V 只需扫到 (pid_m+1)·BLOCK_M，右侧的块整块不读、不算
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, T)  # 跳过对角线右侧的整块
    else:
        hi = T
    # --8<-- [end:flash_init]
    # --8<-- [start:flash_scores]
    # 内层循环：每轮处理 BLOCK_N 个 key。这是 kernel 内部的 for 循环，不是多次 kernel 启动
    for start_n in range(0, hi, BLOCK_N):
        # 这一轮 key 的全局行号
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        # K、V 块的地址，构造方式和 Q 相同
        k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :]  # [BLOCK_N, D]
        v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :]  # [BLOCK_N, D]
        # 读 K 块：超过 T 的行填 0。每轮 64×128 FP16 = 16 KB
        k = tl.load(k_ptrs, mask=offs_n[:, None] < T, other=0.0)  # [BLOCK_N, D]
        # tl.trans(k)：转置成 [D, BLOCK_N]（寄存器里换布局，不碰 HBM）
        # tl.dot：[BLOCK_M, D] @ [D, BLOCK_N] 走 Tensor Core，FP16 输入、FP32 结果
        # 每轮 2·64·64·128 ≈ 1 MFLOP；这个分数块从生到灭都在片上
        s = tl.dot(q, tl.trans(k)) * sm_scale  # [BLOCK_M, BLOCK_N]，FP32
        # 合法位置：key 行号没越界。[1, BLOCK_N] 的布尔，会广播
        valid = offs_n[None, :] < T  # [1, BLOCK_N]
        if CAUSAL:
            # 再加上因果条件 query 行号 ≥ key 行号；[BLOCK_M, 1] 和 [1, BLOCK_N] 比较 -> [BLOCK_M, BLOCK_N]
            valid = valid & (offs_m[:, None] >= offs_n[None, :])  # [BLOCK_M, BLOCK_N]
        # tl.where(条件, a, b)：逐元素三选一。不合法的位置置 -inf，exp 后为 0
        s = tl.where(valid, s, float("-inf"))  # [BLOCK_M, BLOCK_N]
        # --8<-- [end:flash_scores]

        # --8<-- [start:flash_update]
        # tl.max(s, 1)：沿第 1 维（key 维）取行最大值 -> [BLOCK_M]
        # 新最大值 = max(旧最大值, 本块最大值)
        m_new = tl.maximum(m_i, tl.max(s, 1))  # [BLOCK_M]
        # 按新最大值做 exp，每个元素 ≤ 1。m_new[:, None] 变成 [BLOCK_M, 1] 便于广播
        p = tl.exp(s - m_new[:, None])  # [BLOCK_M, BLOCK_N]
        # 旧统计量是按旧最大值算的，乘 exp(m_i - m_new) 换算到新最大值；第一轮 m_i=-inf，alpha=0
        alpha = tl.exp(m_i - m_new)  # [BLOCK_M]
        # exp 和：旧的缩放后加上本块的行和
        l_i = l_i * alpha + tl.sum(p, 1)  # [BLOCK_M]
        # V 块到用的时候才读，缩短它在寄存器里的存活时间
        v = tl.load(v_ptrs, mask=offs_n[:, None] < T, other=0.0)  # [BLOCK_N, D]
        # p 转成 V 的 dtype（FP16）才能走 Tensor Core：[BLOCK_M, BLOCK_N] @ [BLOCK_N, D]
        # 结果是 FP32，加到同样缩放过的 acc 上。矩阵乘用低精度、统计量用 FP32，是 FA 系列的标准做法
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # [BLOCK_M, D]
        m_i = m_new
        # --8<-- [end:flash_update]

    # --8<-- [start:flash_store]
    # 扫完所有 K/V 块后统一归一化，等价于完整 softmax 后再乘 V
    acc = acc / l_i[:, None]  # [BLOCK_M, D]
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :]  # [BLOCK_M, D]
    # 转回输出 dtype，越界行不写。整个 kernel 对 HBM 只有：读 Q 一次、读 K/V 若干次、写 O 一次
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < T)
    # --8<-- [end:flash_store]
    # --8<-- [end:kernel_body]


# --8<-- [start:flash_launch]
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                    block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    # stride(-1) == 1：最后一维 D 必须连续，kernel 里 offs_d 直接加、不乘步长
    assert q.is_cuda and q.shape == k.shape == v.shape and q.stride(-1) == 1
    b, h, t, d = q.shape
    # tl.arange(0, D) 要求 2 的幂；tl.dot 要求维度至少 16
    assert d in (16, 32, 64, 128)
    o = torch.empty_like(q)  # [B, H, T, D]
    # 二维 grid：(Q 块数, B·H)。和 kernel 里的 program_id(0)、program_id(1) 一一对应
    grid = (triton.cdiv(t, block_m), b * h)
    flash_fwd_kernel[grid](
        q, k, v, o,
        # 每个张量传 B、H、T 三个维度的步长；不要求整个张量连续（比如从 [B, T, H, D] 转置来的也行）
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        h, t, d ** -0.5,
        # constexpr 参数用关键字传；每种 (D, 块大小, 因果) 组合编译一份 kernel
        D=d, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
        # 4 warp = 128 线程；num_stages=2 让编译器把下一轮的 K/V 加载和本轮计算重叠
        num_warps=4, num_stages=2,
    )
    return o
# --8<-- [end:flash_launch]
