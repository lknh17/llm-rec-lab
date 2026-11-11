"""S1 第 2-3 课：一个完整的 Llama 风格 decoder-only GPT。

结构（每一层）：
    x -> RMSNorm -> Attention(RoPE, GQA) -> +x -> RMSNorm -> SwiGLU MLP -> +x
最后 RMSNorm -> lm_head 得到每个位置下一个 token 的 logits。

这个模型是后面所有阶段的"实验小白鼠"：
- S2 用它做 prefill/decode 的 roofline 对照（GPU 上可以配成 8B 规模）；
- S4 用它做显存、重计算、LoRA 实验；
- S5 的 mini 推理引擎直接复用它的权重；
- S6/S7 用它做生成式推荐模型（词表换成语义 ID）。

记号：B=batch，T=本次输入的 token 数，S=缓存里已有 + 本次的总长度，C=d_model，H/Hkv=Q/KV 头数，D=head_dim。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int = 256  # 默认按字节建词表（byte-level），不需要 tokenizer
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int | None = None  # 默认等于 n_head（标准多头）；更小就是 GQA
    d_model: int = 256
    ffn_hidden: int | None = None  # 默认取 8/3·C 并向上对齐到 64 的倍数（Llama 的做法）
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    tie_embeddings: bool = True  # 输入 embedding 和 lm_head 共享权重，小模型能省不少参数

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.ffn_hidden is None:
            self.ffn_hidden = 64 * math.ceil(8 * self.d_model / 3 / 64)
        assert self.d_model % self.n_head == 0 and self.n_head % self.n_kv_head == 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


# 预设规模：llama3_8b 用于 GPU 上的 S2 实验（随机权重即可测性能）
PRESETS = {
    "tiny": GPTConfig(),
    "llama3_8b": GPTConfig(
        vocab_size=128256, n_layer=32, n_head=32, n_kv_head=8, d_model=4096,
        ffn_hidden=14336, max_seq_len=8192, rope_theta=500000.0, tie_embeddings=False,
    ),
    "qwen2_5_1_5b": GPTConfig(
        vocab_size=151936, n_layer=28, n_head=12, n_kv_head=2, d_model=1536,
        ffn_hidden=8960, max_seq_len=8192, rope_theta=1000000.0, tie_embeddings=True,
    ),
}


# --8<-- [start:rmsnorm]
class RMSNorm(nn.Module):
    """y = x / rms(x) * g。相比 LayerNorm 去掉了减均值和 bias，少一次归约，效果相当，是 Llama 系的标配。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # eps 防止全零向量时除以 0；1e-6 相对正常激活的均方根（约 1）可以忽略
        self.eps = eps
        # nn.Parameter 把张量注册成可训练参数，optimizer 和 state_dict 都会看到它
        # 初始化为全 1，即一开始"只归一化、不缩放"；每层 C 个参数，相对 MLP 的 3CF 可忽略
        self.weight = nn.Parameter(torch.ones(dim))  # [C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在 FP32 下计算均方根，避免 FP16/BF16 平方后溢出或精度损失
        # FP16 最大约 65504，激活里出现 300 的离群值，平方就溢出成 inf
        x_f = x.float()  # [B, T, C] FP32
        # pow(2).mean(-1, keepdim=True)：每个 token 自己的 C 个数平方后取均值
        # dim=-1 沿最后一维（C）归约；keepdim 保留长度 1 的维，结果能广播回 [B, T, C]
        # rsqrt = 1/sqrt，一个 kernel 里算完，把"除以 rms"变成"乘以 1/rms"
        # 代价：读写 [B, T, C] 几遍，FLOPs 只有每元素几次，是纯访存密集型算子
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)  # [B, T, C]
        # type_as(x) 转回输入精度（如 BF16），再乘 weight：[B, T, C] * [C] 按最后一维广播
        return x_f.type_as(x) * self.weight  # [B, T, C]
# --8<-- [end:rmsnorm]


# --8<-- [start:rope]
# --8<-- [start:rope_table]
def rope_cos_sin(head_dim: int, max_len: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos/sin 表，形状均为 [max_len, D]。

    第 i 对维度的旋转频率 = theta^(-2i/D)：低维转得快（编码局部位置），高维转得慢（编码远距离位置）。
    """
    # arange(0, D, 2) 生成 0, 2, 4, ..., D-2 共 D/2 个数，即公式里的 2i
    # theta ** (2i/D) 从 1 涨到约 theta，取倒数得到频率：第 0 对每个位置转 1 弧度，最后一对最慢
    # D=64、theta=1 万时最慢的频率约 1.3e-4，转一圈要约 4.7 万个位置
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))  # [D/2]
    # 位置 0..max_len-1，转成 float 才能和频率相乘
    pos = torch.arange(max_len, device=device).float()  # [max_len]
    # outer 是外积：angles[m, i] = pos[m] · inv_freq[i]，即位置 m 在第 i 对维度上的旋转角
    angles = torch.outer(pos, inv_freq)  # [max_len, D/2]
    # 前半和后半放同一组角度：第 i 维和第 i+D/2 维配成一对（"rotate_half" 约定，与 HF Llama 一致）
    angles = torch.cat([angles, angles], dim=-1)  # [max_len, D]
    # 只算一次，后面每层每步都查表；max_len=8192、D=128、FP32 时两张表合计约 8 MB
    return angles.cos(), angles.sin()  # 各 [max_len, D]
# --8<-- [end:rope_table]


# --8<-- [start:rope_apply]
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]；cos/sin: [T, D] 或 [B, 1, T, D]（已按位置取好）。

    把每对维度 (x1, x2) 看成复数并旋转角度 pos·freq。
    关键性质：旋转后 q_m·k_n 只依赖相对位置 m-n，所以 attention 天然感知相对距离。
    """
    # chunk(2, dim=-1) 把最后一维切成前后两半：x1 当复数实部，x2 当虚部
    x1, x2 = x.chunk(2, dim=-1)  # 各 [B, H, T, D/2]
    # (x1 + i·x2) · i = -x2 + i·x1，所以 [-x2, x1] 就是"乘以 i"之后的实部和虚部
    rotated = torch.cat([-x2, x1], dim=-1)  # [B, H, T, D]
    # 复数乘 e^{iθ} = x·cos θ + (i·x)·sin θ；cos/sin 的 [T, D] 自动广播到 [B, H, T, D]
    # 先转 FP32：位置到几万时角度很大，BF16 只有 3 位有效数字，旋转误差会被放大
    # 代价：纯逐元素运算，读写 Q、K 各几遍，FLOPs 可忽略；vLLM 把它融进 QKV 投影后的 kernel
    return (x.float() * cos + rotated.float() * sin).type_as(x)  # [B, H, T, D]
# --8<-- [end:rope_apply]
# --8<-- [end:rope]


# --8<-- [start:causal_attention]
# --8<-- [start:causal_attn_fast]
def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q: [B, H, T, D]，k/v: [B, Hkv, S, D]，S ≥ T（前 S-T 个是缓存里的历史）。

    注意：SDPA 的 is_causal=True 在 T != S 时是"左上角对齐"，用于带缓存的场景会出错。
    所以 T == S 时用 is_causal；T == 1（decode）时不需要 mask；其他情况显式构造右下角对齐的 mask。
    enable_gqa=True：kernel 直接按组读取 KV，不需要像第 1 课那样先 repeat_interleave 复制 H/Hkv 份。
    """
    # t：本次新来的 query 个数；s：key 总数（缓存里的历史 + 本次的 t 个）
    t, s = q.size(-2), k.size(-2)
    # Q 头数和 KV 头数不同就是 GQA（本课模型 6 个 Q 头、2 个 KV 头）
    gqa = q.size(1) != k.size(1)
    # 情况 1：没有历史（训练、普通 prefill）。T == S 时左上角和右下角对齐是同一个下三角
    # is_causal=True 让融合 kernel 自己跳过上三角的分块，不用传 mask，计算量约减半
    if t == s:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)  # [B, H, T, D]
    # 情况 2：decode。唯一的 query 就是最新的 token，所有 s 个 key 都在它之前或就是它自己
    # 所以根本不需要 mask：一行 [1, S] 的打分全部可见
    if t == 1:
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=gqa)  # [B, H, 1, D]
# --8<-- [end:causal_attn_fast]
# --8<-- [start:causal_attn_mask]
    # 情况 3：1 < T < S，比如 chunked prefill（长 prompt 分块喂入）、投机解码一次验证多个 token
    # 新来的第 i 个 query 在全序列里的位置是 s-t+i，它能看到第 0 到 s-t+i 个 key
    # tril(diagonal=s-t) 把下三角的对角线右移 s-t 格，正好是"右下角对齐"
    # 例：t=2、s=5 时第 0 行可见 key 0-3，第 1 行可见 key 0-4
    # 若误用 is_causal=True（左上角对齐），第 0 行只能看到 key 0，不报错但结果错
    mask = torch.ones(t, s, dtype=torch.bool, device=q.device).tril(diagonal=s - t)  # [T, S]
    # SDPA 的布尔 mask 里 True 表示"参与 attention"；[T, S] 自动广播到 [B, H, T, S]
    # 显式 mask 会让部分后端退出最快的路径，所以只在这种少见的情况下才用
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)  # [B, H, T, D]
# --8<-- [end:causal_attn_mask]
# --8<-- [end:causal_attention]


# --8<-- [start:kv_cache]
# --8<-- [start:kv_cache_alloc]
class KVCache:
    """预分配的连续 KV cache：k/v 形状 [L, B, Hkv, S_max, D]。

    预分配而不是每步 torch.cat：cat 每步都要拷贝全部历史，O(S) 的额外访存。
    代价是按 S_max 预留显存，长短请求混在一起时会大量浪费 —— 这正是 S5 里 PagedAttention 要解决的问题。
    """

    def __init__(self, cfg: GPTConfig, batch: int, max_len: int, device, dtype):
        # 所有层放进一个大张量，第 0 维按层索引；max_len 就是 S_max，按最长可能的序列预留
        shape = (cfg.n_layer, batch, cfg.n_kv_head, max_len, cfg.head_dim)
        # K、V 各一个 [L, B, Hkv, S_max, D]，创建时就占满显存，不管之后实际用了多少
        # 本课模型 L=6、Hkv=2、D=64：每 token K+V 共 3 KB（BF16）
        # Llama-3-8B（L=32、Hkv=8、D=128）S_max=4K、BF16：每条请求 K+V 共约 0.54 GB
        self.k = torch.zeros(shape, device=device, dtype=dtype)  # [L, B, Hkv, S_max, D]
        self.v = torch.zeros(shape, device=device, dtype=dtype)  # [L, B, Hkv, S_max, D]
        # 已写入的 token 数（所有层相同）：位置 [0, length) 是有效历史，后面是还没写的空位
        self.length = 0
# --8<-- [end:kv_cache_alloc]

# --8<-- [start:kv_cache_update]
    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """k_new/v_new: [B, Hkv, T, D]，写入位置 [length, length+T)，返回该层的全部历史 [B, Hkv, S, D]。"""
        # T=1 是 decode，T>1 是 prefill 或 chunked prefill
        t = k_new.size(2)
        start = self.length
        # 切片赋值是原地写：只拷贝新来的 [B, Hkv, T, D]，历史不动，这是比 torch.cat 省的地方
        self.k[layer, :, :, start:start + t] = k_new
        self.v[layer, :, :, start:start + t] = v_new
        # 返回前 start+t 个位置的切片。切片是 view，不拷贝，和缓存共享同一块内存
        # 这个 view 不连续（每个头之间隔着没用到的 S_max-S 个空位），SDPA 能直接按步长读
        return self.k[layer, :, :, :start + t], self.v[layer, :, :, :start + t]  # [B, Hkv, S, D] × 2

    # update 不改 length：一次前向里 L 层都要写到同一个区间 [length, length+T)
    # 所以由 GPT.forward 在所有层跑完后统一调用 advance(T)
    def advance(self, t: int) -> None:
        self.length += t

    # 按预分配大小算，不是已用大小：这正是"预留浪费"的来源
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2
# --8<-- [end:kv_cache_update]
# --8<-- [end:kv_cache]


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)

    def qkv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """投影 + 拆头 + RoPE。x: [B, T, C] -> q [B, H, T, D]，k/v [B, Hkv, T, D]。
        单独拆出来是为了让 S5 的推理引擎可以复用投影，自己管理 KV 的存放方式。"""
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """q [B, H, T, D]，k/v [B, Hkv, S, D] -> [B, T, C]。mask: [B, 1, T, S] bool（True=可见），用于 packing。"""
        if mask is None:
            y = causal_attention(q, k, v)  # [B, H, T, D]
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=q.size(1) != k.size(1))
        b, _, t, _ = y.shape
        return self.wo(y.transpose(1, 2).reshape(b, t, -1))

    def forward(self, x, cos, sin, cache: KVCache | None = None, layer: int = 0, mask=None):
        q, k, v = self.qkv(x, cos, sin)
        if cache is not None:
            k, v = cache.update(layer, k, v)
        return self.attend(q, k, v, mask)


# --8<-- [start:swiglu_block]
# --8<-- [start:swiglu]
class SwiGLU(nn.Module):
    """MLP：down( silu(gate(x)) * up(x) )。门控让 MLP 表达力更强；三个矩阵，参数量 3·C·F。
    这是 LLM 里参数和 FLOPs 的大头（Llama-3-8B 每层约 81%）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # gate 和 up 都是 C -> F 的升维；Llama-3-8B 里 F=14336≈3.5C，每个矩阵 58.7M 参数
        # 推理引擎会把这两个合并成一个 [C, 2F] 的矩阵乘，少一次 kernel、矩阵更大
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)  # [C] -> [F]
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)    # [C] -> [F]
        # down 把 F 维压回 C 维，回到残差主干的宽度
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)  # [F] -> [C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 从里往外读：gate(x)、up(x) 各 [B, T, F]；silu(z) = z·sigmoid(z) 是平滑版 ReLU
        # F.silu 的 F 是 torch.nn.functional；形状注释里的 F 是 MLP 中间维度 ffn_hidden，两者无关
        # silu(gate) * up 是逐元素相乘："gate 这一路决定 up 这一路每一维放行多少"
        # 三次矩阵乘共 6·C·F FLOPs/token，Llama-3-8B 约 0.35 GFLOP/token/层
        # 中间的 [B, T, F] 是 C 的 3.5 倍宽，训练时要存下来做反向，是激活显存的大头之一
        return self.down(F.silu(self.gate(x)) * self.up(x))  # [B, T, C]
# --8<-- [end:swiglu]


# --8<-- [start:block]
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # 每个子层前各一个 RMSNorm，各自有 C 个参数，不共享
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, layer=0, mask=None):
        # x: [B, T, C]。cos/sin 是本次 token 的旋转表，cache/layer 给 KV cache 用，mask 给 packing 用
        # Pre-Norm 残差：主干 x 始终是"恒等通路"，梯度可以无衰减地穿过几十层，这是深层模型能训起来的关键
        # 归一化只作用在子层的输入上，子层输出直接加回未归一化的 x
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer, mask)  # [B, T, C]
        # 第二个子层看到的是"已经加上 attention 结果"的 x：先在 token 间搬信息，再逐 token 加工
        # 推理引擎把"残差加法 + 下一个 RMSNorm"融成一个 kernel，省掉一遍 [B, T, C] 的读写
        return x + self.mlp(self.mlp_norm(x))  # [B, T, C]
# --8<-- [end:block]
# --8<-- [end:swiglu_block]


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_cos_sin(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)  # [S_max, D]
        self.register_buffer("rope_sin", sin, persistent=False)
        self.grad_checkpointing = False  # S4：激活重计算，用约 1/3 的额外计算换掉每层的中间激活
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    # --8<-- [start:gpt_forward]
    # --8<-- [start:gpt_forward_pos]
    def forward(self, idx: torch.Tensor, cache: KVCache | None = None, positions: torch.Tensor | None = None,
                doc_ids: torch.Tensor | None = None):
        """idx: [B, T] token id -> logits [B, T, V]。

        有 cache 时，本次输入接在缓存的历史之后，位置从 cache.length 开始。
        positions / doc_ids 用于 packing（S4）：多篇文档拼在一行里，每篇位置从 0 开始、只能看到本篇的 token。
        """
        # idx 是整数 token id；训练和 prefill 时 T 是整段长度，decode 时 T=1
        b, t = idx.shape  # [B, T]
        # 本次第一个 token 的绝对位置：无缓存时从 0 开始；有缓存时接在已缓存的 cache.length 个 token 之后
        # prompt 长 P 时，第 n 个生成的 token（从 0 数）start = P+n；写成 0 就是第 3 课旋钮 C 的 bug
        start = cache.length if cache is not None else 0
        # cos/sin 表只预算到 max_seq_len，超出就没有角度可查；尽早报错而不是越界切出空张量
        assert start + t <= self.cfg.max_seq_len, f"序列长度 {start + t} 超过 max_seq_len={self.cfg.max_seq_len}"
        if positions is None:
            # 普通情况：所有序列共用连续位置 start..start+T-1，切片不复制数据
            # rope_cos 是 register_buffer 注册的 [max_seq_len, D] 表，跟着模型 .to(device)
            cos, sin = self.rope_cos[start:start + t], self.rope_sin[start:start + t]  # [T, D]
        else:
            # packing：positions 是 [B, T] 整数，每条序列每个 token 自带位置（每篇文档从 0 数）
            # 用整数张量做下标是 gather：[B, T] -> [B, T, D]，这一步会真的复制
            # unsqueeze(1) 插入头维，变成 [B, 1, T, D]，才能广播到 q/k 的 [B, H, T, D]
            cos, sin = self.rope_cos[positions].unsqueeze(1), self.rope_sin[positions].unsqueeze(1)  # [B, 1, T, D]
    # --8<-- [end:gpt_forward_pos]
    # --8<-- [start:gpt_forward_mask]
        # 默认不传 mask，attention 内部自己处理因果（见第 6 课 causal_attention）
        mask = None
        if doc_ids is not None:
            # doc_ids: [B, T]，标记每个 token 属于这一行里的第几篇文档
            # tril() 取下三角：位置 i 只能看到 ≤ i 的位置
            causal = torch.ones(t, t, dtype=torch.bool, device=idx.device).tril()  # [T, T]
            # 下标里写 None 表示在该位置插入一个长度为 1 的新维（等价于 unsqueeze）
            # [B, T, 1] == [B, 1, T] 广播比较：same_doc[b, i, j] 表示 i、j 是否同一篇
            same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]  # [B, T, T]
            # 两个条件同时满足才可见；[:, None] 插入头维以便广播到 H 个头
            # T=4096、B=8 时这个 bool mask 是 8·4096² ≈ 134 MB，长序列 packing 要用变长 kernel 代替
            mask = (causal & same_doc)[:, None]  # [B, 1, T, T]：块对角 + 因果
    # --8<-- [end:gpt_forward_mask]
    # --8<-- [start:gpt_forward_layers]
        # 查 embedding 表：每个 id 取出一行 C 维向量
        x = self.embed(idx)  # [B, T, C]
        # 逐层串行：每层的输入是上一层的输出，所以层数决定了串行步数（第 2 课旋钮 C）
        # 所有层共用同一份 cos/sin 和 mask；layer 编号 i 告诉 KV cache 写到哪一层
        for i, block in enumerate(self.blocks):
            # self.training 由 model.train()/model.eval() 切换；推理时不走重计算
            if self.grad_checkpointing and self.training:
                # 前向时不保存本层内部激活，反向时用保存的输入 x 重算一遍本层
                x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, cache, i, mask, use_reentrant=False)
            else:
                x = block(x, cos, sin, cache, i, mask)  # [B, T, C]
        # 所有层都写完本次的 K/V 后，才把缓存长度前移 T；放在循环里会让后面的层写错位置
        if cache is not None:
            cache.advance(t)
        # 最后一个 RMSNorm，再投影到词表：[B, T, C] @ [C, V]
        # V=128K、T=4096、FP32 时 logits 有 2 GB/条序列，训练时常只对需要的位置算
        return self.lm_head(self.norm(x))  # [B, T, V]
    # --8<-- [end:gpt_forward_layers]
    # --8<-- [end:gpt_forward]

    def num_params(self, include_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n if include_embedding else n - self.embed.weight.numel()


# --8<-- [start:count_params]
def count_params(cfg: GPTConfig) -> dict[str, int]:
    """不实例化模型，按公式数参数。用来估算 8B、70B 这类放不进本机内存的模型。"""
    # kv 是 K（或 V）投影的输出维度 Hkv·D；Llama-3-8B 是 8·128 = 1024，只有 C 的 1/4
    c, f, kv = cfg.d_model, cfg.ffn_hidden, cfg.n_kv_head * cfg.head_dim
    # Wq、Wo 各 C²；Wk、Wv 各 C·(Hkv·D)。Llama-3-8B：2·4096² + 2·4096·1024 ≈ 41.9M
    attn = c * c * 2 + c * kv * 2
    # gate、up、down 三个矩阵各 C·F。Llama-3-8B：3·4096·14336 ≈ 176M，是 attention 的 4.2 倍
    mlp = 3 * c * f
    # 每层两个 RMSNorm 的 weight，各 C 个；8B 模型 32 层合计只有 26 万，可忽略
    norms = 2 * c
    per_layer = attn + mlp + norms
    # embedding 表 [V, C]；Llama-3-8B 是 128256·4096 ≈ 525M
    embed = cfg.vocab_size * c
    # lm_head 是 [C] -> [V] 的 Linear，和 embedding 同样大；共享权重时不重复计数
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * c
    # 最后的 "+ c" 是 lm_head 前那个最终 RMSNorm 的 weight
    total = cfg.n_layer * per_layer + embed + head + c
    return {"attention_per_layer": attn, "mlp_per_layer": mlp, "per_layer": per_layer,
            "embedding": embed, "lm_head": head, "total": total}
# --8<-- [end:count_params]
