"""W4A16 GEMM（Triton，需要 NVIDIA GPU）：y = x @ W^T，x FP16 [M, K]，W 以 INT4 打包存放。

和第 2 课 matmul.py 的结构相同（一个 program 算 y 的一个 [BM, BN] 块，沿 K 循环），只在主循环里多三步：
    读打包权重 [BK, BN] uint8 -> 寄存器里移位取半字节、减 8 -> 乘本组的 scale -> tl.dot 喂 Tensor Core。
FP16 权重只在寄存器里存在，显存里只有 0.5 字节/权重 + 每 g 个权重一个 FP16 scale。

打包布局见 code/w4a16_ref.py 的 pack_int4：packed [K/2, N] uint8，packed[j, n] 的低 4 位是 W^T[2j, n]，高 4 位是 W^T[2j+1, n]。
scales [K/g, N] FP16。限制：K 是 128 的倍数，group_size 是 2 的幂。

这是 GPTQ-Triton 风格的写法（每行按 k//2 取字节、按 k%2 选半字节），容易读懂；
Marlin 这类生产 kernel 会把权重离线重排成 mma 指令需要的寄存器布局，再配合异步流水，见讲义第 2 步。
"""

import torch
import triton
import triton.language as tl


# --8<-- [start:autotune]
@triton.autotune(
    configs=[
        # decode（M ≤ 16）：BM 取最小的 16，program 数只能靠 N/BN 凑（本 kernel 没做 split-K）；BK 大一点减少循环次数
        triton.Config({"BM": 16, "BN": 64, "BK": 128}, num_stages=4, num_warps=4),
        triton.Config({"BM": 16, "BN": 128, "BK": 128}, num_stages=4, num_warps=4),
        triton.Config({"BM": 32, "BN": 64, "BK": 128}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_stages=4, num_warps=4),
        # prefill（M 大）：和 FP16 matmul 一样用大块，摊薄反量化（每个 BM 行块解包一次权重）
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 64, "BK": 64}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def w4a16_kernel(a_ptr, b_ptr, s_ptr, c_ptr, M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_sg, stride_sn, stride_cm, stride_cn,
                 GROUP_SIZE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # --8<-- [end:autotune]
    # --8<-- [start:setup]
    # 二维 grid：(M 方向的块号, N 方向的块号)。decode 时 M 方向只有 1 块，并行度全靠 N/BN
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)  # [BM]
    offs_n = pid_n * BN + tl.arange(0, BN)  # [BN]
    offs_k = tl.arange(0, BK)  # [BK]

    # x 的 [BM, BK] 块：普通 FP16，和 matmul.py 一样
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak  # [BM, BK]
    # 打包权重：第 k 行在字节行 k // 2，所以相邻两行 k=2j、2j+1 指向同一个字节（重复读的那次命中 L1）
    b_ptrs = b_ptr + (offs_k[:, None] // 2) * stride_bk + offs_n[None, :] * stride_bn  # [BK, BN]
    # 偶数行取低 4 位（右移 0），奇数行取高 4 位（右移 4）
    shifts = (offs_k % 2) * 4  # [BK]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # --8<-- [end:setup]

    # --8<-- [start:mainloop]
    for k0 in range(0, K, BK):
        # x 块：M 不满 BM（decode 时 M=1）的行填 0；K 是 BK 的倍数，不用 K 方向的 mask
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)  # [BM, BK] fp16
        # 读 int4：BK×BN 个权重只占 BK·BN/2 字节，这正是 W4A16 省下的带宽
        packed = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0)  # [BK, BN] uint8
        # 解包：先转 int32 再移位（避免 uint8 移位的类型提升歧义），& 0xF 得到 [0, 15]，
        # 转 FP16 再减 8 回到 [-8, 7]。全部在寄存器里，由 CUDA Core 执行
        q = ((packed.to(tl.int32) >> shifts[:, None]) & 0xF).to(tl.float16) - 8.0  # [BK, BN]
        # 本块每一行所在的组：(k0 + r) // g。BK ≤ g 时整块共用一行 scale，这些地址相同、只读一次
        s_ptrs = s_ptr + ((k0 + offs_k[:, None]) // GROUP_SIZE) * stride_sg + offs_n[None, :] * stride_sn
        scale = tl.load(s_ptrs, mask=offs_n[None, :] < N, other=0.0)  # [BK, BN] fp16
        # 反量化：显式转回 FP16，保证 tl.dot 两个操作数类型一致
        w = (q * scale).to(tl.float16)  # [BK, BN]
        # Tensor Core：[BM, BK] @ [BK, BN] -> FP32 累加
        acc += tl.dot(a, w)
        a_ptrs += BK * stride_ak
        # 打包权重每 BK 行只前进 BK/2 个字节行
        b_ptrs += (BK // 2) * stride_bk
    # --8<-- [end:mainloop]

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


# --8<-- [start:launch]
def w4a16_matmul(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """x: [M, K] fp16，packed: [K/2, N] uint8，scales: [K/g, N] fp16 -> [M, N] fp16。"""
    assert x.is_cuda and x.dtype == torch.float16 and packed.dtype == torch.uint8
    m, k = x.shape
    n = packed.shape[1]
    assert packed.shape[0] * 2 == k and k % 128 == 0 and group_size & (group_size - 1) == 0
    c = torch.empty((m, n), device=x.device, dtype=x.dtype)
    # grid 是函数：autotune 选好 BM、BN 之后才知道要几个 program
    grid = lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))
    w4a16_kernel[grid](x, packed, scales, c, m, n, k,
                       x.stride(0), x.stride(1), packed.stride(0), packed.stride(1),
                       scales.stride(0), scales.stride(1), c.stride(0), c.stride(1),
                       GROUP_SIZE=group_size)
    return c
# --8<-- [end:launch]
