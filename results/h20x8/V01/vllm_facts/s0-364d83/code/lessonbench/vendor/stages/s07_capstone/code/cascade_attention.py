"""共享前缀的 cascade attention（FlashInfer 的 cascade / Hydragen 的思路）。

beam search 的 W 个 beam 共享同一段用户历史（P 个 token），各自只多出几个私有 token（S 个）。
普通的 batch decode 让每个 beam 各读一遍历史 KV：读取量 W·(P+S)。
cascade 把 attention 拆成两段，再用 log-sum-exp 合并（与 FlashDecoding 合并分块结果的公式相同）：

    第 1 段：W 个 beam 的 query 拼在一起，对共享前缀做一次 attention   读 KV：P（只读一遍）
             —— 等价于把 W 个 query 当成 prefill 的 W 个 token，变成矩阵乘（算术强度 ×W）
    第 2 段：每个 beam 对自己的 S 个私有 token 做 attention               读 KV：W·S
    合并：  o = (o1·e^{lse1} + o2·e^{lse2}) / (e^{lse1} + e^{lse2})

KV 读取量从 W·(P+S) 降到 P + W·S。P=10K、W=20、S≈16 时降到约 1/19。

GQA：H 个 query 头分成 Hkv 组，每组 G = H/Hkv 个头共享一个 KV 头。
把 [W, H, D] 的 query 重排成 [Hkv, W·G, D]，每个 KV 头只读一次，不用 repeat_interleave 复制 KV。
"""

import torch


# --8<-- [start:cascade]
# --8<-- [start:cascade_attend]
def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q [..., M, D]，k/v [..., N, D] -> (o [..., M, D]，lse [..., M])，无 mask。"""
    # k.transpose(-1, -2)：[..., N, D] -> [..., D, N]，只改步长不搬数据；"..." 的前几维当作批
    # q.size(-1) ** -0.5 就是 1/√D。不需要因果 mask：两段 KV 都在当前 token 之前（或就是它自己）
    # 这个分数矩阵会真实落地：genrec_1b 共享段每层 [16, 20, 10K] 个 fp32 ≈ 12.8 MB
    scores = q @ k.transpose(-1, -2) * q.size(-1) ** -0.5  # [..., M, N]
    # lse = log Σ_j exp(s_j)：这一段 softmax 的对数分母，合并两段时要用它加权
    # torch.logsumexp 内部先减行最大值再 exp，不会上溢；dim=-1 表示沿 key 维归约
    lse = torch.logsumexp(scores, dim=-1)  # [..., M]
    # softmax 里又做一遍"行最大值 + 指数和"，和上一行重复：教学版为了可读多读了一遍 scores
    # 融合 kernel（FlashAttention 的 softmax_lse 输出）在一次扫描里同时得到 o 和 lse
    # [..., M, N] @ [..., N, D] -> o [..., M, D]
    return torch.softmax(scores, dim=-1) @ v, lse
# --8<-- [end:cascade_attend]


# --8<-- [start:cascade_private]
def cascade_attention(q: torch.Tensor, k_shared: torch.Tensor | None, v_shared: torch.Tensor | None,
                      k_priv: torch.Tensor, v_priv: torch.Tensor) -> torch.Tensor:
    """
    q:        [W, H, D]       W 个 beam 当前 token 的 query
    k_shared: [Hkv, P, D]     所有 beam 共享的前缀 KV（只读一次）；P=0 时传 None
    k_priv:   [W, Hkv, S, D]  每个 beam 私有的 KV（含当前 token，所以 S ≥ 1）
    返回      [W, H, D]
    """
    # W = beam 数，H = query 头数，D = 头维度；KV 头数 Hkv 从私有 KV 的第 1 维读出
    w, h, d = q.shape
    hkv = k_priv.size(1)
    # G = 每个 KV 头服务几个 query 头：MHA 时 G=1，GQA 8/2 时 G=4
    g = h // hkv
    # attention 和合并都在 fp32 里做，exp 和除法的误差更小
    # .float() 对 fp32 输入不复制；对 BF16 输入会复制一份 fp32，生产 kernel 是在寄存器里转
    qf = q.float()  # [W, H, D]

    # 第 2 段：私有部分，按 beam 批量计算
    # view 把 H 维切成 [Hkv, G]：第 j 个 KV 头对应第 j·G 到 j·G+G-1 个 query 头
    # 这和 S1 的 repeat_interleave 分组约定（[k0, k0, k1, k1]）一致，换个切法就会配错头
    q2 = qf.view(w, hkv, g, d)  # [W, Hkv, G, D]
    # [W, Hkv, G, D] @ [W, Hkv, D, S] -> [W, Hkv, G, S]：前两维当批，每个 beam 只读自己的 S 个 token
    # 读取量 W·S 个 token，S ≤ block_size + 生成步数（十几个），和 P=10K 比可以忽略
    # 返回 o2 [W, Hkv, G, D]，lse2 [W, Hkv, G]
    o2, lse2 = _attend(q2, k_priv.float(), v_priv.float())
    # 历史短于一个块时 shared_len=0、没有共享段：私有段就是全部 KV，它的结果就是答案
    # reshape 把 [Hkv, G] 合回 H，.to(q.dtype) 转回调用方的精度
    if k_shared is None:
        return o2.reshape(w, h, d).to(q.dtype)  # [W, H, D]
# --8<-- [end:cascade_private]

# --8<-- [start:cascade_shared]
    # 第 1 段：共享前缀，W·G 个 query 一起读同一份 KV
    # permute(1, 0, 2, 3)：[W, Hkv, G, D] -> [Hkv, W, G, D]，把 KV 头挪到最前面当批
    # reshape 再把 W、G 合成一维：同一个 KV 头下的 W·G 个 query 排成一个矩阵
    # permute 之后内存不连续，view 会报错；reshape 在需要时自动复制，这里只有 W·H·D 个数
    q1 = q2.permute(1, 0, 2, 3).reshape(hkv, w * g, d)  # [Hkv, W·G, D]
    # [Hkv, W·G, D] @ [Hkv, D, P] -> [Hkv, W·G, P]：每个 KV 头的 P 个 token 只读一次
    # 每读一个 KV 元素做 W·G 次乘加，算术强度从普通 decode 的 G 提到 W·G
    # genrec_1b（MHA）、P=10K：每层读 K、V 共 2·16·10K·128·2 B ≈ 82 MB，24 层合计 1.97 GB
    # 返回 o1 [Hkv, W·G, D]，lse1 [Hkv, W·G]
    o1, lse1 = _attend(q1, k_shared.float(), v_shared.float())
    # 拆回 beam 维、换回 [W, Hkv, G, ...] 的顺序，和第 2 段的结果逐元素对齐
    o1 = o1.view(hkv, w, g, d).permute(1, 0, 2, 3)  # [W, Hkv, G, D]
    lse1 = lse1.view(hkv, w, g).permute(1, 0, 2)  # [W, Hkv, G]
# --8<-- [end:cascade_shared]

# --8<-- [start:cascade_merge]
    # 合并：按各段的 softmax 分母（e^lse）加权
    # 推理：o1 = Σ_{j∈段1} e^{s_j}·v_j / e^{lse1}，所以 e^{lse1}·o1 就是段 1 的"未归一化分子"
    # 两段分子相加、再除以分母之和 e^{lse1} + e^{lse2}，正好是对全部 P+S 个 key 做 softmax
    # 先减两者的较大值 m：直接 exp(lse) 可能上溢；分子分母同除以 e^m，比值不变
    m = torch.maximum(lse1, lse2)  # [W, Hkv, G]
    # [..., None] 在末尾补一维 [W, Hkv, G, 1]，才能广播到 [W, Hkv, G, D]
    w1, w2 = torch.exp(lse1 - m)[..., None], torch.exp(lse2 - m)[..., None]
    # 逐元素运算，量级只有 W·H·D；FlashInfer 里对应单独的 merge_state kernel
    o = (o1 * w1 + o2 * w2) / (w1 + w2)  # [W, Hkv, G, D]
    # [Hkv, G] 合回 H；内存不连续时 reshape 会自动复制一份
    return o.reshape(w, h, d).to(q.dtype)  # [W, H, D]
# --8<-- [end:cascade_merge]
# --8<-- [end:cascade]
