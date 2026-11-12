"""MoE grouped GEMM（Triton，需要 NVIDIA GPU），按 vLLM fused_moe 的思路写。

预处理（code/moe_ref.py 的 moe_align_block_size_ref）：把 T·k 个 (token, 槽位) 按专家排序，
每个专家的段补齐到 BLOCK_M 的倍数，得到 sorted_token_ids [P]、expert_ids [P/BLOCK_M]。

kernel：一个 program 算输出的一个 [BLOCK_M, BLOCK_N] 块。BLOCK_M 行全部属于同一个专家 expert_ids[pid_m]，
所以只需要把 B 指针偏移到这个专家的权重，其余就是第 2 课的分块 matmul：
    A 行 = sorted_token_ids 指向的 token（间接寻址，gather），B = w[expert]，C 行 = 写回同一个 flat 下标（scatter）。
一次 kernel 启动算完所有专家：不用 E 次启动，也不用按最大负载 padding。
"""

import torch
import triton
import triton.language as tl

from stages.s03_kernels.code.moe_ref import moe_align_block_size_ref


@triton.jit
def fused_moe_kernel(
    a_ptr, b_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, TOP_K: tl.constexpr,
):
    # --8<-- [start:pid]
    # 一维 grid，按第 2 课 matmul.py 的 GROUP_M 分组调度换算成 (pid_m, pid_n)，提高 L2 命中
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # --8<-- [end:pid]

    # --8<-- [start:gather]
    # 本块的 BLOCK_M 个位置 -> 它们各自的 flat 下标（token·k + 槽位）；补位行是哨兵 num_valid_tokens
    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)  # [BLOCK_M] int32
    token_mask = offs_token < num_valid_tokens
    # 整块只属于一个专家：读一个标量就知道该用哪份权重。int64 防止 expert·stride_be 溢出
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # % N：最后一个 N 块越界的列绕回到合法列，B 的加载就不用 N 方向的 mask；这些列的结果在写回时被屏蔽
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    # A 是 gather：第一层 A=hidden [T, C]，flat 下标 // TOP_K 才是 token 行号；第二层 TOP_K=1，行号就是 flat 下标
    # 同一块里的行在 A 中不连续，每行内部 K 维连续，所以每行仍能合并访存
    a_ptrs = a_ptr + (offs_token[:, None] // TOP_K) * stride_am + offs_k[None, :] * stride_ak  # [BLOCK_M, BLOCK_K]
    # B 是这个专家的权重 w[expert]：[N, K] 按 (k, n) 取，当成 [BLOCK_K, BLOCK_N] 的 B 操作数
    b_ptrs = b_ptr + expert * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]
    # --8<-- [end:gather]

    # --8<-- [start:kloop]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        # 补位行不读（mask），填 0；它们算出来的结果最后也不写回
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(a, b)  # Tensor Core，FP32 累加
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # --8<-- [end:kloop]

    # --8<-- [start:epilogue]
    if MUL_ROUTED_WEIGHT:
        # 第二层顺手乘 gate 权重（epilogue 融合）：topk_weights 按 flat 下标存放 [T·k]
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)  # [BLOCK_M]
        acc = acc * w[:, None]
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # C 是 scatter：写回到 flat 下标那一行，等于在 kernel 里顺便做完了 unpermute
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=token_mask[:, None] & (offs_cn[None, :] < N))
    # --8<-- [end:epilogue]


# --8<-- [start:launch]
def grouped_gemm(a, w, out, sorted_ids, expert_ids, top_k, topk_weights=None, block_m=64, block_n=64, block_k=64):
    """out[flat] = a[flat // top_k] @ w[expert(flat)]^T（可选乘 topk_weights[flat]）。w: [E, N, K]，out: [T·k, N]。"""
    em = sorted_ids.numel()
    n, k = w.shape[1], w.shape[2]
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, block_n),)
    # 第一层没有 gate 权重：随便传一个合法指针（out）占位，MUL_ROUTED_WEIGHT=False 时 kernel 不会读它
    # num_valid_tokens = out.shape[0] = T·k，也就是哨兵值：sorted_ids 里 ≥ 它的都是补位行
    fused_moe_kernel[grid](
        a, w, out, topk_weights if topk_weights is not None else out, sorted_ids, expert_ids,
        n, k, em, out.shape[0],
        a.stride(0), a.stride(1), w.stride(0), w.stride(1), w.stride(2), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8,
        MUL_ROUTED_WEIGHT=topk_weights is not None, TOP_K=top_k,
        num_warps=4, num_stages=3,
    )
    return out


def fused_moe(x, w1, w2, topk_weights, topk_ids, block_m: int | None = None):
    """x: [T, C] fp16，w1: [E, 2I, C]，w2: [E, C, I]，topk_weights/ids: [T, k] -> [T, C]。"""
    t, top_k = topk_ids.shape
    e_num, two_i, _ = w1.shape
    if block_m is None:
        # 平均每个专家分到的行数少（decode）时用小块，减少补位浪费；多（prefill）时用大块，提高复用
        block_m = 16 if t * top_k / e_num <= 32 else 64
    sorted_ids, expert_ids, _ = moe_align_block_size_ref(topk_ids, e_num, block_m)
    # 第一层：hidden -> [T·k, 2I]，每个 (token, 槽位) 一行
    h = torch.empty(t * top_k, two_i, device=x.device, dtype=x.dtype)
    grouped_gemm(x, w1, h, sorted_ids, expert_ids, top_k, block_m=block_m)
    # SiLU(gate)·up：vLLM 里是一个单独的融合 kernel（silu_and_mul），这里用 PyTorch
    gate, up = h.chunk(2, dim=-1)
    act = (torch.nn.functional.silu(gate) * up).contiguous()  # [T·k, I]
    # 第二层：A 已经按 flat 下标排列，所以 top_k=1；epilogue 乘 gate 权重
    y = torch.empty(t * top_k, w2.shape[1], device=x.device, dtype=x.dtype)
    grouped_gemm(act, w2, y, sorted_ids, expert_ids, 1, topk_weights.flatten().to(x.dtype).contiguous(),
                 block_m=block_m)
    # 沿 k 求和合并：[T, k, C] -> [T, C]（vLLM 用 moe_sum kernel）
    return y.view(t, top_k, -1).sum(dim=1)
# --8<-- [end:launch]
