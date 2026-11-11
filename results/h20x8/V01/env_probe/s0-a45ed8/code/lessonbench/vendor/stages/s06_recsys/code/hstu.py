"""HSTU（Hierarchical Sequential Transduction Unit，Meta 2024，"Actions Speak Louder than Words"）。

和标准 Transformer block 的区别：

    标准：  Y = X + Attn(LN(X));  Z = Y + FFN(LN(Y))           每层 12·d² 参数（MHA + 4d FFN）
    HSTU：  U, V, Q, K = split(SiLU(f1(LN(X))))                 f1: d -> 4·H·dh
            A = SiLU(Q Kᵀ + rab) / N   （逐点激活，不做 softmax）
            Y = X + f2(LN(A V) ⊙ U)                             没有 FFN，U 门控代替 FFN 的作用

为什么不用 softmax：
- softmax 把注意力权重归一化为和为 1，"看了 100 个相关物品"和"看了 1 个"被抹平；推荐里交互的强度/次数本身是信号。
- 去掉 softmax 后不需要 online softmax 的 max/sum 规约，kernel 更简单；配合 jagged（变长不补齐）实现，长序列上比 FlashAttention2 快 5-15 倍（论文数据）。

rab 是相对位置偏置（原文还有相对时间偏置：两次交互的时间间隔分桶）。
"""

import torch
import torch.nn.functional as F
from torch import nn


# --8<-- [start:hstu_layer]
# --8<-- [start:hstu_init]
class HSTULayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, head_dim: int, max_len: int):
        super().__init__()
        self.h, self.dh = n_head, head_dim
        # 输入前的 LayerNorm（pre-norm，和 GPT block 一样）
        self.norm_in = nn.LayerNorm(d_model)
        # 一个大投影同时产出 U、V、Q、K：[C] -> [4·H·dh]。H·dh = C 时参数 4C²
        # 对应 GPT 里的 wq、wk、wv 三个投影，外加一份门控 U
        self.f1 = nn.Linear(d_model, 4 * n_head * head_dim)
        # 对 attention 输出再做一次 LayerNorm：没有 softmax 归一化，输出量级要靠它稳住
        self.norm_attn = nn.LayerNorm(n_head * head_dim)
        # 输出投影 [H·dh] -> [C]，参数 C²。f1 + f2 合计 5C²，这一层就这么多参数
        self.f2 = nn.Linear(n_head * head_dim, d_model)
        # nn.Parameter：把张量登记为可训练参数（会出现在 model.parameters() 里）
        # 每个头一张长度 max_len 的表：相对距离 d 查第 d 个偏置。只有 H × 50 = 100 个数
        self.rel_pos = nn.Parameter(torch.zeros(n_head, max_len))  # 相对距离 0..max_len-1 的偏置
# --8<-- [end:hstu_init]

# --8<-- [start:hstu_proj]
    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]；valid: [B, T] bool（padding 为 False）"""
        b, t, _ = x.shape
        # 先 SiLU 再拆：U、V、Q、K 四路都过了非线性。GPT 的 Q/K/V 是纯线性投影
        h = F.silu(self.f1(self.norm_in(x)))  # [B, T, 4·H·dh]
        # view 把最后一维切成 [4, H, dh]：[B, T, 4, H, dh]
        # permute(2, 0, 3, 1, 4) 把维度重排成 [4, B, H, T, dh]，只改步长不搬数据
        # 再按第 0 维解包成 4 个张量。顺序必须和训练时一致，否则加载权重会静默出错
        u, v, q, k = h.view(b, t, 4, self.h, self.dh).permute(2, 0, 3, 1, 4)  # 每个 [B, H, T, dh]
# --8<-- [end:hstu_proj]

# --8<-- [start:hstu_rab]
        # [T, 1] - [1, T] 广播成 [T, T]：dist[i, j] = i - j，query i 和 key j 隔了几步
        # j > i 的未来位置是负数，clamp 成 0 只是为了下标合法，后面会被因果 mask 掉
        dist = (torch.arange(t, device=x.device)[:, None] - torch.arange(t, device=x.device)[None]).clamp(min=0)
        # 用 [T, T] 的整数矩阵做下标查表：[H, max_len] -> [H, T, T]
        rab = self.rel_pos[:, dist]  # [H, T, T]
        # 和标准 attention 一样的点积打分，再加偏置；[H, T, T] 广播到 [B, H, T, T]
        # 显式构造 [B, H, T, T]：T=50 时很小；T=10K 时单层单头 1e8 个元素、fp32 400 MB
        scores = q @ k.transpose(-1, -2) / self.dh**0.5 + rab  # [B, H, T, T]
# --8<-- [end:hstu_rab]

# --8<-- [start:hstu_attn]
        # 下三角：query i 只能看 j ≤ i
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()  # [T, T]
        # 三个条件取与：因果、key 是真实 token、query 是真实 token
        # valid[:, None, None, :] 是 [B, 1, 1, T]（按 key 广播），[:, None, :, None] 是 [B, 1, T, 1]（按 query）
        mask = causal & valid[:, None, None, :] & valid[:, None, :, None]  # [B, 1, T, T]
        # 每个序列的真实长度 N（不是补齐后的 T），[B] -> [B, 1, 1, 1] 以便广播
        # clamp(min=1) 防止全 padding 的序列除以 0
        n = valid.sum(dim=1).clamp(min=1).view(b, 1, 1, 1).float()  # 每个序列的真实长度
        # 逐点 SiLU 代替 softmax：每个 (i, j) 的权重独立，不按行归一化
        # mask 必须乘在 SiLU 之后：先填 -inf 的话 SiLU(-inf) = -inf·0 = NaN
        # bool 乘浮点时 True 当 1、False 当 0；再除以 N 让长短不同的用户量级可比
        attn = F.silu(scores) * mask / n  # 逐点激活，除以序列长度（不是 softmax）
# --8<-- [end:hstu_attn]

# --8<-- [start:hstu_out]
        # 加权求和 [B, H, T, T] @ [B, H, T, dh] -> [B, H, T, dh]
        # 合并多头 -> [B, T, H, dh] -> [B, T, H·dh]；transpose 后不连续，reshape 会自动复制
        o = (attn @ v).transpose(1, 2).reshape(b, t, -1)  # [B, T, H·dh]
        # U 门控：LayerNorm 后的 attention 输出和 U 逐元素相乘，代替 FFN 提供非线性
        # U 也要从 [B, H, T, dh] 合并成 [B, T, H·dh] 才能逐元素乘
        # f2 投影回 [B, T, C]，加残差。整层没有 FFN
        return x + self.f2(self.norm_attn(o) * u.transpose(1, 2).reshape(b, t, -1))  # [B, T, C]
# --8<-- [end:hstu_out]
# --8<-- [end:hstu_layer]


# --8<-- [start:block_params]
# --8<-- [start:hstu_rec]
class HSTURec(nn.Module):
    """物品 id 序列 -> 每个位置对下一个物品的打分 [B, T, n_items+1]（输出层与物品 embedding 共享）。"""

    def __init__(self, n_items: int, d_model: int = 64, n_layer: int = 2, n_head: int = 2, max_len: int = 50):
        super().__init__()
        # 物品 embedding [3707, 64] = 24 万参数，是整个模型的大头
        # padding_idx=0：第 0 行不参与梯度更新
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        # ModuleList：让 PyTorch 能找到列表里每一层的参数；H·dh = d
        self.layers = nn.ModuleList(HSTULayer(d_model, n_head, d_model // n_head, max_len) for _ in range(n_layer))
        self.norm = nn.LayerNorm(d_model)
        # 小方差初始化，和 GPT 的 embedding 初始化一致
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # 物品 id 0 是 padding
        valid = idx != 0  # [B, T]
        # 查表 [B, T] -> [B, T, C]；valid[..., None] 是 [B, T, 1]，把 padding 位置的输入清零
        x = self.item_emb(idx) * valid[..., None]  # [B, T, C]
        for layer in self.layers:
            x = layer(x, valid)  # [B, T, C]
        # 输出层和输入 embedding 共享权重：[B, T, C] @ [C, V] -> [B, T, V]
        # 打分 = 用户表示和物品 embedding 的点积，就是双塔召回的打分方式
        return self.norm(x) @ self.item_emb.weight.T  # [B, T, n_items+1]
# --8<-- [end:hstu_rec]


# --8<-- [start:block_params_fn]
def block_params(d: int, kind: str) -> int:
    """每层参数量（不含 norm 和偏置）。"""
    # d=64 时 4.9 万；d=2048 时 5000 万
    if kind == "transformer":  # MHA 4d² + FFN 8d²
        return 12 * d * d
    # d=64 时 2.0 万；d=2048 时 2100 万。同预算可以堆 12/5 = 2.4 倍的层
    if kind == "hstu":  # f1 4d²（H·dh = d）+ f2 d²
        return 5 * d * d
    raise ValueError(kind)
# --8<-- [end:block_params_fn]
# --8<-- [end:block_params]
