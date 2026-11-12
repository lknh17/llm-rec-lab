"""S1 第 1 课：Attention 的三种写法。

1. naive_attention：把公式一步一步写出来，完整的 [T, T] 分数矩阵会真实地落在显存里。最适合理解原理。
2. sdpa_attention：调用 PyTorch 官方的 scaled_dot_product_attention。在 CUDA 上会走 FlashAttention 这类融合 kernel。
3. CausalSelfAttention：完整的模块，包括 Q/K/V 投影、多头、GQA、输出投影。这是 LLM 里真正使用的形态。

记号：B=batch，T=序列长度，C=模型维度 (d_model)，H=query 头数，Hkv=KV 头数，D=每个头的维度 (C // H)。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


# --8<-- [start:naive]
# --8<-- [start:naive_scores]
def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """q, k, v: [B, H, T, D] -> 输出 [B, H, T, D]"""
    # 每个头的维度 D，比如 64；size(-1) 表示"最后一维的长度"
    d = q.size(-1)

    # 第 1 步：每个 query 和所有 key 做点积，得到相关性分数
    # k.transpose(-2, -1) 交换最后两维：[B, H, T, D] -> [B, H, D, T]，只改步长、不搬数据
    # @ 是批量矩阵乘：前两维 [B, H] 当作"批"，后两维做 [T, D] @ [D, T] -> [T, T]
    # 计算量 2·B·H·T²·D；这个 [B, H, T, T] 矩阵是显存随 T² 增长的根源
    # 例：B=4, H=8, T=4096, FP16 时它有 4·8·4096²·2 字节 ≈ 1.07 GB
    scores = q @ k.transpose(-2, -1)  # [B, H, T, T]

    # 第 2 步：除以 sqrt(D)。D 维点积的方差约等于 D，不缩放的话 softmax 会过于尖锐、梯度消失
    # 代价：这是一个独立的逐元素 kernel，要把 [B, H, T, T] 完整读一遍、写一遍
    scores = scores / math.sqrt(d)  # [B, H, T, T]
# --8<-- [end:naive_scores]

# --8<-- [start:naive_mask]
    # 第 3 步：因果 mask。第 i 个 token 只能看到 ≤ i 的位置，这是自回归生成的前提
    if causal:
        # query 个数和 key 个数；训练时相等，decode 时 t_q 很小
        t_q, t_k = q.size(-2), k.size(-2)
        # torch.ones(...).tril(k) 生成下三角为 True 的 [t_q, t_k] 布尔矩阵
        # diagonal=t_k - t_q 把对角线右移：decode 时最后一个 query 应该能看到全部 t_k 个 key
        mask = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril(diagonal=t_k - t_q)  # [t_q, t_k]
        # ~mask 取反得到"未来位置"；masked_fill 把这些位置填 -inf，softmax 后权重就是 0
        # [t_q, t_k] 的 mask 会自动广播到 [B, H, t_q, t_k]，不用真的复制 B·H 份
        # 代价：又一次完整读写 [B, H, T, T]
        scores = scores.masked_fill(~mask, float("-inf"))  # [B, H, T, T]
# --8<-- [end:naive_mask]

# --8<-- [start:naive_softmax]
    # 第 4 步：softmax 归一化成权重，每一行的和为 1
    # dim=-1 表示沿最后一维（key 维）归一化：每个 query 对所有 key 的权重和为 1
    # 内部要先求行最大值、再求指数和、再相除，至少两遍读 [B, H, T, T]
    weights = scores.softmax(dim=-1)  # [B, H, T, T]

    # 第 5 步：按权重对 V 加权求和
    # [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]；计算量同样是 2·B·H·T²·D
    # 输出回到和输入同样的形状，T² 的中间矩阵到这里才被"消化掉"
    return weights @ v  # [B, H, T, D]
# --8<-- [end:naive_softmax]
# --8<-- [end:naive]


# --8<-- [start:sdpa]
def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """官方融合实现，数学上与 naive_attention 等价。"""
    # 一个函数调用完成上面五步。PyTorch 按设备和形状自动选后端：
    # CUDA 上优先 FlashAttention-2，其次 memory-efficient attention，都不满足才退回 math（等价于 naive）
    # is_causal=True 时融合 kernel 直接跳过上三角的分块，计算量约减半；naive 做不到这一点
    # F 是文件开头 import 的 torch.nn.functional，和形状记号无关
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)  # [B, H, T, D]
# --8<-- [end:sdpa]


# --8<-- [start:module]
# --8<-- [start:module_init]
class CausalSelfAttention(nn.Module):
    """LLM 里的 attention 层：投影 -> 拆多头 -> attention -> 合并 -> 输出投影。

    n_kv_heads < n_heads 时就是 GQA（Grouped-Query Attention）：多个 query 头共享同一组 K/V。
    这样能把 KV cache 缩小为原来的 n_kv_heads / n_heads。Llama-3、Qwen 等主流模型都在用。
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, use_sdpa: bool = True):
        # nn.Module 的固定写法：注册参数、子模块之前必须先调用
        super().__init__()
        # 不传就等于 n_heads，即普通多头（MHA）
        n_kv_heads = n_kv_heads or n_heads
        # C 必须能被 H 整除（每个头分到整数维）；H 必须能被 Hkv 整除（每个 KV 头带整数个 Q 头）
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        # D = C / H，比如 4096 / 32 = 128
        self.head_dim = d_model // n_heads
        # True 走融合 kernel，False 走 naive，便于对照
        self.use_sdpa = use_sdpa

        # 参数量：Q 是 C·C；K 和 V 各是 C·(Hkv·D)；O 是 C·C。没有 GQA 时合计 4C²
        # nn.Linear(in, out) 内部存一个 [out, in] 的权重矩阵，前向就是 x @ W^T
        # bias=False：现代 LLM（Llama、Qwen 的 O 投影）基本不用偏置，省参数且不影响效果
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)     # [C] -> [H·D] = [C]
        # [C] -> [Hkv·D]，GQA 时更小
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)  # [C] -> [Hkv·D]
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)     # [H·D] -> [C]
# --8<-- [end:module_init]

# --8<-- [start:module_split]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # "_" 表示这一维的值后面用不到
        b, t, _ = x.shape  # [B, T, C]

        # 投影并拆成多头：[B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        # view 只重新解释形状、不搬数据：最后一维 C 被切成 H 段，每段 D 个数
        # transpose(1, 2) 把"头"挪到"序列"前面，这样每个头的 [T, D] 是一块，便于按头做矩阵乘
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)     # [B, H, T, D]
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, T, D]
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, T, D]
# --8<-- [end:module_split]

# --8<-- [start:module_gqa]
        # GQA：把每个 KV 头复制给它负责的那一组 query 头。[B, Hkv, T, D] -> [B, H, T, D]
        # repeat_interleave(rep, dim=1) 把第 1 维每个元素连续重复 rep 次：[k0, k1] -> [k0, k0, k1, k1]
        # 所以第 0、1 个 Q 头共用 k0，第 2、3 个共用 k1，和 Llama/Qwen 的分组方式一致
        # 这里为了清晰直接复制；推理引擎里会让 kernel 直接按组读取，避免这次复制
        # 代价：复制后 K/V 显存和读取量回到 MHA 水平，GQA 在带宽上的收益被抵消了
        if self.n_kv_heads != self.n_heads:
            # 每个 KV 头服务几个 Q 头，比如 32 / 8 = 4
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)  # [B, H, T, D]
            v = v.repeat_interleave(rep, dim=1)  # [B, H, T, D]
# --8<-- [end:module_gqa]

# --8<-- [start:module_out]
        # 函数也是对象，可以先选再调用
        attn = sdpa_attention if self.use_sdpa else naive_attention
        y = attn(q, k, v, causal=True)  # [B, H, T, D]

        # 合并多头：[B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        # transpose 之后内存不再连续，直接 view 会报错；contiguous() 真的复制一份成连续内存
        # 代价：一次 [B, T, C] 的读写，比 attention 本身小得多
        # -1 表示"自动推算这一维"
        y = y.transpose(1, 2).contiguous().view(b, t, -1)  # [B, T, C]
        # 输出投影把各个头的结果混合回 C 维：[B, T, C]
        return self.wo(y)
# --8<-- [end:module_out]
# --8<-- [end:module]


def attention_flops(b: int, t: int, d_model: int, n_kv_heads_ratio: float = 1.0) -> dict[str, float]:
    """一层 attention 前向的 FLOPs 拆解（不考虑因果 mask 能省下的一半计算）。"""
    # Q、O 各 C²，K、V 各 C²·ratio
    proj = 2 * b * t * d_model * d_model * (2 + 2 * n_kv_heads_ratio)
    # QK^T 和 AV 各 2·B·T²·C
    core = 2 * (2 * b * t * t * d_model)
    return {"projection": proj, "core_attention": core, "total": proj + core}


def score_matrix_bytes(b: int, n_heads: int, t: int, dtype_bytes: int = 2) -> int:
    """naive 实现中 [B, H, T, T] 分数矩阵占用的显存。"""
    return b * n_heads * t * t * dtype_bytes
