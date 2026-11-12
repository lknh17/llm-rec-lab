"""分块矩阵乘（tiling）：让每个从 HBM 读进来的元素在片上被复用 BM 或 BN 次。

一个 program 计算 C 的一个 [BM, BN] 块：沿 K 维每次读 A 的 [BM, BK] 和 B 的 [BK, BN] 进 SRAM，
用 Tensor Core 做 tl.dot 累加到 FP32 寄存器里。
块内算术强度 = 2·BM·BN·BK / (2·(BM·BK + BK·BN)) = BM·BN / (BM + BN)：128×128 的块 → 64 FLOP/B（相对 HBM），
再加上 L2 复用（GROUP_M 分组调度），足以越过 H100 的拐点。
"""

import torch
import triton
import triton.language as tl


# --8<-- [start:autotune]
# --8<-- [start:matmul_configs]
# @triton.autotune：给出几组候选的编译期参数，每种新形状第一次调用时逐个实测，缓存最快的那组
# triton.Config({...}, num_stages, num_warps)：
#   BM/BN/BK = C 块的行数/列数、每次沿 K 读多少；GROUP_M = 分组调度的组高（见下）
#   num_stages = 软件流水线级数：同时在飞的 A/B 小块份数，越多越能藏住访存延迟，越占共享内存
#   num_warps = 每个 program 的 warp 数（1 warp = 32 线程）；8 warp = 256 线程
# 第一组 128×128×64、3 级流水：共享内存 3·(128·64 + 64·128)·2 B = 96 KB，累加器 64 KB 寄存器
@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 64, "BK": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
    ],
    # key 里的参数值一变就重新 autotune；同一形状只测一次（首次调用会慢几百 ms）
    key=["M", "N", "K"],  # 不同形状自动选最快的块配置
)
# stride_xy = 张量 x 沿 y 维走一步跨多少个元素。行优先的 A[M, K]：stride_am=K，stride_ak=1
# 把步长作为参数传进来，kernel 就能处理转置过的、不连续的输入，不必先 .contiguous()
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    # --8<-- [end:matmul_configs]
    # --8<-- [start:matmul_group]
    # ---- 分组调度：相邻的 program 计算相邻的 C 块，共享 A 的行块/B 的列块，提高 L2 命中率
    # grid 是一维的，pid 从 0 到 (M/BM)·(N/BN)-1；下面把它翻译成 C 块的二维坐标 (pid_m, pid_n)
    pid = tl.program_id(0)  # 标量
    # C 在行、列方向各有几个块。M=N=4096、BM=BN=128 时都是 32，共 1024 个块
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    # 一组 = GROUP_M 行块 × 全部列块。例子里 8 × 32 = 256 个 program 一组
    num_pid_in_group = GROUP_M * num_pid_n
    # 本 program 属于第几组，这一组从第几个行块开始
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    # 最后一组可能不满 GROUP_M 行（num_pid_m 不是 GROUP_M 的倍数时）
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    # 组内编号先沿行走、再换列（列优先）：组内 pid 0..7 -> (行 0..7, 列 0)，8..15 -> (行 0..7, 列 1)
    # 这样同时运行的几百个 program 只用到 8 个 A 行块和少数几个 B 列块，都能留在 L2 里
    # 对比朴素的 pid_m = pid // num_pid_n：同时运行的 program 会扫过全部 32 个 B 列块
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # --8<-- [end:matmul_group]
    # --8<-- [end:autotune]

    # --8<-- [start:kloop]
    # --8<-- [start:matmul_ptrs]
    # ---- 本 program 的 A 行块与 B 列块的指针
    # 本块负责的 BM 个行号、BN 个列号。% M 把越界行号绕回到合法行：
    # 这样读 A 时不用 mask 也不会越界，绕回的那些行算出的结果在最后 store 时被 mask 丢掉
    offs_m = (pid_m * BM + tl.arange(0, BM)) % M  # [BM]
    offs_n = (pid_n * BN + tl.arange(0, BN)) % N  # [BN]
    # K 方向的块内偏移 0..BK-1，每轮循环都一样，靠移动指针前进
    offs_k = tl.arange(0, BK)  # [BK]
    # 二维指针块：offs_m[:, None] 是 [BM, 1]，offs_k[None, :] 是 [1, BK]，相加广播成 [BM, BK]
    # 第 (i, j) 个元素 = A[offs_m[i], offs_k[j]] 的地址 = 行号·行步长 + 列号·列步长
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak  # [BM, BK]
    # B 同理：行是 K 方向，列是本块的 BN 个列号
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # [BK, BN]
    # --8<-- [end:matmul_ptrs]

    # --8<-- [start:matmul_loop]
    # 累加器放在寄存器里，整个 K 循环期间不离开片上。128×128 FP32 = 64 KB，8 warp 时每线程 64 个
    # FP32 累加，避免 BF16 累加的精度损失：FP16/BF16 尾数只有 10/7 位，
    # 累加 K=4096 个乘积后误差会大到不可接受。tl.zeros(形状, dtype) 创建全 0 的块
    acc = tl.zeros((BM, BN), dtype=tl.float32)  # [BM, BN]
    # 沿 K 走 cdiv(K, BK) 轮：K=4096、BK=64 时 64 轮
    for k in range(0, tl.cdiv(K, BK)):
        # K - k·BK 是这一轮还剩多少个合法的 K 下标；最后一轮可能不足 BK，超出部分填 0
        # 填 0 而不是别的值：0 乘任何数是 0，不影响点积
        # 每轮读 A 块 128×64×2 B = 16 KB、B 块 16 KB，进共享内存
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)  # [BM, BK]
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)  # [BK, BN]
        # tl.dot：块矩阵乘 [BM, BK] @ [BK, BN] -> [BM, BN]，编译成 Tensor Core 的 MMA 指令
        # FP16 输入、FP32 输出。每轮 2·128·128·64 ≈ 2.1 MFLOP，对 32 KB 读取：强度 64
        acc += tl.dot(a, b)  # Tensor Core
        # 指针沿 K 前进一个块：A 往右移 BK 列，B 往下移 BK 行
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    # --8<-- [end:matmul_loop]

    # --8<-- [start:matmul_store]
    # 写 C 时重新计算行列号，这次不取模，越界的就是越界的，交给 mask 处理
    offs_cm = pid_m * BM + tl.arange(0, BM)  # [BM]
    offs_cn = pid_n * BN + tl.arange(0, BN)  # [BN]
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn  # [BM, BN]
    # 二维 mask：行和列都合法才写。acc 转回 C 的 dtype（如 fp16）后只写这一次
    # 如果要做 GEMM + bias + 激活（epilogue 融合），就插在这一行之前对 acc 操作
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
    # --8<-- [end:matmul_store]
    # --8<-- [end:kloop]


# --8<-- [start:matmul_launch]
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [M, K] @ b: [K, N] -> [M, N]"""
    assert a.is_cuda and a.shape[1] == b.shape[0]
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)  # [M, N]
    # grid 写成函数：BM、BN 由 autotune 在启动时才选定，meta 就是被选中的那组配置
    # 一维 grid，program 总数 = C 的块数；4096² 用 128×128 块时是 1024 个
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    # 把三个张量的行、列步长都传进去；BM/BN/BK/GROUP_M 不用传，autotune 会填
    matmul_kernel[grid](a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
# --8<-- [end:matmul_launch]
