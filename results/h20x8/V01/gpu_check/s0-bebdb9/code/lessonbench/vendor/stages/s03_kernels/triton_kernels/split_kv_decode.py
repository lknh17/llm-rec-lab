"""paged split-KV decode attention（FlashDecoding 的两阶段写法，Triton，需要 NVIDIA GPU）。

阶段 1（partial kernel）：grid = (B, Hkv, S)。每个 program 负责一条序列、一个 KV 头、KV 的第 s 段：
    按 block table 逐页读 K/V（每页 PAGE 个 token），组内 G 个 query 头一起做 online softmax，
    写出局部输出 o_s [G, D] 和 lse_s [G]（FP32）。
阶段 2（reduce kernel）：grid = (B, H)。读回 S 份 (o_s, lse_s)，用 log-sum-exp 加权合并成最终输出。

和参考实现 code/split_kv_decode.py 的 paged_split_kv_decode 一一对应，数值应一致。
限制：PAGE 和 D 是 16 的倍数且为 2 的幂（tl.dot 的要求）；q [B, H, D] 最后一维连续；
k_pool/v_pool [num_blocks, PAGE, Hkv, D]（vLLM 的块布局）；block_table [B, max_pages] int32；seq_lens [B] int32。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def split_kv_partial_kernel(
    Q, K_pool, V_pool, block_table, seq_lens, O_part, LSE_part,
    stride_qb, stride_qh,
    stride_kn, stride_kp, stride_kh,
    stride_vn, stride_vp, stride_vh,
    stride_btb,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh,
    sm_scale, G,
    D: tl.constexpr, PAGE: tl.constexpr, BLOCK_H: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    # --8<-- [start:partial_setup]
    # tl.program_id(axis) 是本 program 在 grid 第 axis 维上的编号；grid = (B, Hkv, S)
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)

    # 每条序列长度不同，从 seq_lens 读出本序列的真实长度（标量）
    seq_len = tl.load(seq_lens + b)
    # 按页切段：本序列共 n_pages 页，每段 pages_per_split 页，本段负责 [p_start, p_end)
    n_pages = tl.cdiv(seq_len, PAGE)
    pages_per_split = tl.cdiv(n_pages, NUM_SPLITS)
    p_start = split * pages_per_split
    # 最后一段可能不满；S 比页数还多时 p_start >= p_end，循环一次都不跑（空段）
    p_end = tl.minimum(p_start + pages_per_split, n_pages)

    offs_h = tl.arange(0, BLOCK_H)  # [BLOCK_H]，组内 q 头
    offs_d = tl.arange(0, D)  # [D]
    offs_p = tl.arange(0, PAGE)  # [PAGE]，页内槽位
    # tl.dot 要求 M ≥ 16：G 只有 7（Qwen2.5-7B）时把组补到 16 行，多出的行用 mask 屏蔽
    h_mask = offs_h < G
    # 本 KV 头负责的 query 头的全局编号：kvh·G + [0, G)
    heads = kvh * G + offs_h  # [BLOCK_H]

    # q 指针：基址 + b 行 + 每个头一行 + D 维连续；[BLOCK_H, 1] 和 [1, D] 广播成 [BLOCK_H, D]
    q_ptrs = Q + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :]
    # mask 为 False 的位置不访存，直接填 other=0；补出来的行算出的结果最后不写回
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)  # [BLOCK_H, D]

    # online softmax 的三个统计量，全部 FP32、常驻寄存器
    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)  # 行最大值
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)  # 指数和
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)  # 未归一化输出
    # --8<-- [end:partial_setup]

    # --8<-- [start:partial_loop]
    # 循环上下界是运行时的值（每条序列不同），Triton 会编译成普通的 for 循环
    for page in range(p_start, p_end):
        # 间接寻址：逻辑页号 -> 物理块号。转 int64 防止大块池时 phys·stride 溢出 int32
        phys = tl.load(block_table + b * stride_btb + page).to(tl.int64)
        # 这一页里的 token 全局下标，最后一页可能只用了一部分槽位
        tok = page * PAGE + offs_p  # [PAGE]
        tok_mask = tok < seq_len
        # 物理块内的地址：块基址 + 槽位 + KV 头 + D。一页是 [PAGE, D] 的一块，D 维连续，访存可以合并
        k_ptrs = K_pool + phys * stride_kn + offs_p[:, None] * stride_kp + kvh * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=tok_mask[:, None], other=0.0)  # [PAGE, D]
        # G 个 q 头同时和这一页的 PAGE 个 key 打分：一次 [BLOCK_H, D] @ [D, PAGE]，FP32 结果
        s = tl.dot(q, tl.trans(k)) * sm_scale  # [BLOCK_H, PAGE]
        # 槽位没用到的列填 -inf，softmax 后权重为 0
        s = tl.where(tok_mask[None, :], s, float("-inf"))

        # online softmax 更新，和第 3 课 flash_attn.py 完全相同
        m_new = tl.maximum(m_i, tl.max(s, 1))  # [BLOCK_H]
        p = tl.exp(s - m_new[:, None])  # [BLOCK_H, PAGE]
        # 第一页时 m_i = -inf，alpha = 0，旧的（空）统计量被清掉
        alpha = tl.exp(m_i - m_new)  # [BLOCK_H]
        l_i = l_i * alpha + tl.sum(p, 1)
        v_ptrs = V_pool + phys * stride_vn + offs_p[:, None] * stride_vp + kvh * stride_vh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=tok_mask[:, None], other=0.0)  # [PAGE, D]
        # P 转回 FP16 再上 Tensor Core，累加器保持 FP32
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # [BLOCK_H, D]
        m_i = m_new
    # --8<-- [end:partial_loop]

    # --8<-- [start:partial_store]
    # 局部输出按本段自己的分母归一化；空段 l_i = 0，用 1 代替防止 0/0，lse 记 -inf 让合并时权重为 0
    has = l_i > 0
    o = acc / tl.where(has, l_i, 1.0)[:, None]  # [BLOCK_H, D]
    lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))  # [BLOCK_H]
    # O_part [B, H, S, D]、LSE_part [B, H, S]，都是 FP32：S 份局部结果要保留足够精度再合并
    o_ptrs = O_part + b * stride_ob + heads[:, None] * stride_oh + split * stride_os + offs_d[None, :]
    tl.store(o_ptrs, o, mask=h_mask[:, None])
    tl.store(LSE_part + b * stride_lb + heads * stride_lh + split, lse, mask=h_mask)
    # --8<-- [end:partial_store]


@triton.jit
def split_kv_reduce_kernel(
    O_part, LSE_part, Out,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh,
    stride_outb, stride_outh,
    NUM_SPLITS,
    D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # --8<-- [start:reduce]
    # grid = (B, H)：每个 program 合并一个 (序列, q 头) 的 S 份局部结果
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, BLOCK_S)  # [BLOCK_S]，BLOCK_S 是 ≥ S 的 2 的幂
    offs_d = tl.arange(0, D)
    s_mask = offs_s < NUM_SPLITS
    # 补出来的段 lse 填 -inf，权重自然是 0
    lse = tl.load(LSE_part + b * stride_lb + h * stride_lh + offs_s, mask=s_mask, other=float("-inf"))  # [BLOCK_S]
    # 减最大值防上溢；至少有一段非空（seq_len ≥ 1），所以 lse_max 有限
    lse_max = tl.max(lse, 0)
    w = tl.exp(lse - lse_max)  # [BLOCK_S]
    o_ptrs = O_part + b * stride_ob + h * stride_oh + offs_s[:, None] * stride_os + offs_d[None, :]
    o = tl.load(o_ptrs, mask=s_mask[:, None], other=0.0)  # [BLOCK_S, D]
    # Σ w_s·o_s / Σ w_s：和 S7 cascade attention 的合并公式相同
    out = tl.sum(o * w[:, None], 0) / tl.sum(w, 0)  # [D]
    tl.store(Out + b * stride_outb + h * stride_outh + offs_d, out.to(Out.dtype.element_ty))
    # --8<-- [end:reduce]


# --8<-- [start:launch]
def default_num_splits(batch: int, n_kv_heads: int, max_pages: int, n_sm: int) -> int:
    """最小的 2 的幂 S，使 B·Hkv·S ≥ SM 数（每个 SM 至少一个 program），且每段至少 1 页、最多 64 段。"""
    s = 1
    while batch * n_kv_heads * s < n_sm and s * 2 <= min(64, max_pages):
        s *= 2
    return s


def split_kv_decode(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor, block_table: torch.Tensor,
                    seq_lens: torch.Tensor, num_splits: int | None = None) -> torch.Tensor:
    """q: [B, H, D] -> [B, H, D]。seq_lens: [B] int32（cuda）。num_splits=None 时按 SM 数自动选。"""
    assert q.is_cuda and q.stride(-1) == 1 and k_pool.stride(-1) == 1 and v_pool.stride(-1) == 1
    b, h, d = q.shape
    page, hkv = k_pool.shape[1], k_pool.shape[2]
    g = h // hkv
    assert h % hkv == 0 and d in (16, 32, 64, 128, 256) and page >= 16 and page & (page - 1) == 0
    if num_splits is None:
        n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = default_num_splits(b, hkv, block_table.shape[1], n_sm)
    # 中间结果：S 份 FP32 的 (o, lse)。B=1、H=28、S=32、D=128 时约 0.46 MB，相比 20 MB 的 KV 很小
    o_part = torch.empty(b, h, num_splits, d, device=q.device, dtype=torch.float32)
    lse_part = torch.empty(b, h, num_splits, device=q.device, dtype=torch.float32)
    # 组内 q 头补到 ≥ 16 的 2 的幂，满足 tl.dot 的最小尺寸
    block_h = max(16, triton.next_power_of_2(g))
    split_kv_partial_kernel[(b, hkv, num_splits)](
        q, k_pool, v_pool, block_table, seq_lens, o_part, lse_part,
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2),
        block_table.stride(0),
        o_part.stride(0), o_part.stride(1), o_part.stride(2),
        lse_part.stride(0), lse_part.stride(1),
        d ** -0.5, g,
        D=d, PAGE=page, BLOCK_H=block_h, NUM_SPLITS=num_splits,
        num_warps=4, num_stages=2,
    )
    out = torch.empty_like(q)
    # 第二个 kernel 启动约 2-4 μs，这是 split-KV 的固定代价；S=1 时也照跑，保持代码简单
    split_kv_reduce_kernel[(b, h)](
        o_part, lse_part, out,
        o_part.stride(0), o_part.stride(1), o_part.stride(2),
        lse_part.stride(0), lse_part.stride(1),
        out.stride(0), out.stride(1),
        num_splits,
        D=d, BLOCK_S=max(2, triton.next_power_of_2(num_splits)),
        num_warps=4,
    )
    return out
# --8<-- [end:launch]


def contiguous_to_paged(k: torch.Tensor, v: torch.Tensor, page: int, seed: int = 0):
    """等长连续 KV [B, Hkv, T, D] -> 打乱的块池 [B·T/P, P, Hkv, D] + block_table [B, T/P]（T 须是 P 的倍数）。"""
    b, hkv, t, d = k.shape
    assert t % page == 0
    n = t // page

    def to_pool(x):
        # [B, Hkv, n, P, D] -> [B, n, P, Hkv, D] -> [B·n, P, Hkv, D]
        return x.view(b, hkv, n, page, d).permute(0, 2, 3, 1, 4).reshape(b * n, page, hkv, d)

    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(b * n, generator=g).to(k.device)
    # pool_new[perm[i]] = pool_old[i]：逻辑块 i 被放到物理块 perm[i]
    k_pool, v_pool = torch.empty_like(to_pool(k)), torch.empty_like(to_pool(v))
    k_pool[perm], v_pool[perm] = to_pool(k), to_pool(v)
    block_table = perm.view(b, n).to(torch.int32)
    return k_pool, v_pool, block_table
