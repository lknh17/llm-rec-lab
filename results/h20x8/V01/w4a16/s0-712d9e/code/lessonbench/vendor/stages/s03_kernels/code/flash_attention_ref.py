"""S3 第 3 课：用纯 PyTorch 写出 FlashAttention 的算法（分块 + online softmax）。

它在 Mac 上并不会更快（Python 循环、每块仍经过 HBM），目的是把算法本身讲清楚、用测试证明正确。
真正快的版本是 triton_kernels/flash_attn.py：同样的循环，但每块的中间结果都留在片上 SRAM/寄存器。

核心思想：
- 不物化 [T, T] 分数矩阵。把 K/V 按 block_n 切块，Q 按 block_m 切块。
- softmax 需要整行的 max 和 sum —— 用 online softmax 边扫边更新：
    m_new = max(m_old, max(s_block))
    l_new = l_old · exp(m_old - m_new) + sum(exp(s_block - m_new))
    acc   = acc   · exp(m_old - m_new) + exp(s_block - m_new) @ V_block
  最后 O = acc / l。数学上与普通 softmax 完全等价。
"""

import torch


# --8<-- [start:ref]
# --8<-- [start:ref_outer]
def flash_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                        block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """q, k, v: [B, H, T, D] -> [B, H, T, D]"""
    b, h, t, d = q.shape
    # 1/sqrt(D)，D=64 时是 0.125
    scale = d ** -0.5
    # 输出先占好位置，每个 Q 块算完写进自己那一段
    out = torch.empty_like(q)  # [B, H, T, D]
    # 外层循环：Q 按 block_m 行一块。ms/me = 这一块的起止行（me 是开区间）
    # T=2048、block_m=64 时 32 块。Triton 版里每一块是一个独立的 program，完全并行
    for ms in range(0, t, block_m):
        # 最后一块可能不足 block_m 行（例如 T=200 时最后一块只有 8 行）
        me = min(ms + block_m, t)
        # 取出这一块 Q，转 FP32 并提前乘 scale（乘在 [bm, D] 上比乘在 [bm, bn] 分数上便宜）
        # 这一块 Q 在整个内层循环里反复使用，对应真实 kernel 里"常驻片上"
        qb = q[:, :, ms:me].float() * scale  # [B, H, bm, D]
        # 三个 online softmax 统计量，每个 query 行一份，全部 FP32
        # m 初值 -inf：还没看到任何分数；torch.full 用给定值填满新张量
        m = torch.full((b, h, me - ms), float("-inf"), device=q.device)  # 每行当前的最大值
        l = torch.zeros(b, h, me - ms, device=q.device)  # 每行当前的 exp 和
        # acc 是"按当前最大值缩放、还没除以 l"的加权和，[bm, D] 而不是 [bm, T]
        acc = torch.zeros(b, h, me - ms, d, device=q.device)  # 未归一化的输出
        # 因果：这一块最后一行是 me-1，只能看到列 < me，所以只需扫到对角线所在的块，
        # 右上角整块跳过（省一半计算）；非因果要扫满全部 T 列
        hi = me if causal else t  # K/V 扫描上界
        # --8<-- [end:ref_outer]
        # --8<-- [start:ref_scores]
        # 内层循环：K/V 按 block_n 行一块，从 0 扫到 hi
        for ns in range(0, hi, block_n):
            ne = min(ns + block_n, hi)
            # 取一块 K、V 并转 FP32；真实 kernel 里这一步是 HBM -> SRAM 的加载
            kb, vb = k[:, :, ns:ne].float(), v[:, :, ns:ne].float()  # [B, H, bn, D]
            # [B, H, bm, D] @ [B, H, D, bn]：只是一个小块，不是 [T, T]
            # 每个 (b, h) 的 64×64 FP32 只有 16 KB，能放进片上
            s = qb @ kb.transpose(-2, -1)  # [B, H, bm, bn]
            # 块的列范围 [ns, ne) 和行范围 [ms, me) 有交叉时才可能出现"列 > 行"
            # 完全在对角线左下方的块（ne <= ms）所有位置都合法，省掉构造 mask
            if causal and ne > ms:  # 只有跨对角线的块需要 mask
                # 用全局行号、列号构造 mask：rows 是 [bm, 1]，cols 是 [1, bn]，比较后广播成 [bm, bn]
                rows = torch.arange(ms, me, device=q.device)[:, None]  # [bm, 1]
                cols = torch.arange(ns, ne, device=q.device)[None, :]  # [1, bn]
                # 列号 > 行号 = 未来位置，填 -inf，exp 后为 0
                s = s.masked_fill(cols > rows, float("-inf"))  # [B, H, bm, bn]
            # --8<-- [end:ref_scores]
            # --8<-- [start:ref_update]
            # online softmax 第 1 式：新最大值 = max(旧最大值, 这一块的行最大值)
            m_new = torch.maximum(m, s.amax(-1))  # [B, H, bm]
            # 用新最大值做 exp：每个元素 ≤ 1，不会溢出。m_new[..., None] 补一维 -> [B, H, bm, 1] 便于广播
            p = torch.exp(s - m_new[..., None])  # [B, H, bm, bn]
            # 旧的 l、acc 是按旧最大值 m 算的：exp(s - m)。换成新最大值要乘 exp(m - m_new) ≤ 1
            # 因为 exp(s - m) · exp(m - m_new) = exp(s - m_new)。第一块时 m=-inf，alpha=0，旧值清零
            alpha = torch.exp(m - m_new)  # 旧统计量的缩放因子
            # 第 2 式：旧的 exp 和先缩放，再加上这一块的 exp 和
            l = l * alpha + p.sum(-1)  # [B, H, bm]
            # 第 3 式：旧的加权和先缩放，再加上这一块的 P·V
            # p @ vb：[B, H, bm, bn] @ [B, H, bn, D] -> [B, H, bm, D]
            acc = acc * alpha[..., None] + p @ vb  # [B, H, bm, D]
            m = m_new
        # 扫完所有 K/V 块后统一除以 l，等价于对整行做了一次标准 softmax 再乘 V
        # 转回输入 dtype 后写进输出的对应行：每个 Q 块只写一次
        out[:, :, ms:me] = (acc / l[..., None]).to(q.dtype)
    return out
    # --8<-- [end:ref_update]
# --8<-- [end:ref]


# --8<-- [start:hbm_model]
def hbm_bytes_naive(b: int, h: int, t: int, d: int, dtype_bytes: int = 2, score_passes: int = 6) -> float:
    """naive：读 Q/K/V、写 O，加上 [T,T] 分数矩阵被读写 score_passes 次（matmul 写、scale 读写、mask 读写、softmax 读写、AV 读）。"""
    # 4·T·D：Q、K、V、O 四个 [T, D] 各一遍；score_passes·T²：分数矩阵的反复读写
    # T=8K、H=32、D=128 时前者 0.27 GB、后者 25.8 GB：分数矩阵占了 99%
    return dtype_bytes * b * h * (4 * t * d + score_passes * t * t)


def hbm_bytes_flash(b: int, h: int, t: int, d: int, block_m: int = 128, dtype_bytes: int = 2) -> float:
    """flash：每个 Q 块读一次、写一次 O；每个 Q 块要把它之前的 K/V 扫一遍（因果平均扫一半）。"""
    # Q 被切成几块；每一块是一个 program，各自要把 K/V 扫一遍
    n_q_blocks = max(1, t // block_m)
    # K、V 各 [T, D]，每个 Q 块读一遍，因果再除以 2。T 越大、block_m 越小，重复读越多
    # T=8K、block_m=128 时 64 个 Q 块，K/V 平均被读 32 遍
    # 因果：平均每个 Q 块扫一半的 K/V
    kv_reads = 2 * t * d * n_q_blocks / 2  # 元素数
    # 2·T·D：读一遍 Q、写一遍 O。没有任何 T² 项落在 HBM 上
    return dtype_bytes * b * h * (2 * t * d + kv_reads)
# --8<-- [end:hbm_model]
