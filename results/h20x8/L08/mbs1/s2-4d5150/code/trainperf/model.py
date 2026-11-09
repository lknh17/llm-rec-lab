"""类 Qwen2.5 结构（GQA、RoPE、SwiGLU、RMSNorm、q/k/v 带 bias），按预设形状随机初始化。只用来测训练性能。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PRESETS = {   # 形状来自 Qwen2.5 的 config.json
    "tiny": dict(d=128, layers=2, heads=4, kv=2, ffn=384, vocab=1024, tie=True),
    "qwen0.5b": dict(d=896, layers=24, heads=14, kv=2, ffn=4864, vocab=151936, tie=True),
    "qwen1.5b": dict(d=1536, layers=28, heads=12, kv=2, ffn=8960, vocab=151936, tie=True),
    "qwen7b": dict(d=3584, layers=28, heads=28, kv=4, ffn=18944, vocab=152064, tie=False),
}


@dataclass
class Shape:
    d: int
    layers: int
    heads: int
    kv: int
    ffn: int
    vocab: int
    tie: bool

    @property
    def dh(self) -> int:
        return self.d // self.heads


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


# --8<-- [start:attention]
class Attention(nn.Module):
    def __init__(self, s: Shape, impl: str):
        super().__init__()
        self.s, self.impl = s, impl
        self.q = nn.Linear(s.d, s.heads * s.dh, bias=True)
        self.k = nn.Linear(s.d, s.kv * s.dh, bias=True)
        self.v = nn.Linear(s.d, s.kv * s.dh, bias=True)
        self.o = nn.Linear(s.heads * s.dh, s.d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        s = self.s
        q = rope(self.q(x).view(B, T, s.heads, s.dh).transpose(1, 2), cos, sin)   # [B, H, T, dh]
        k = rope(self.k(x).view(B, T, s.kv, s.dh).transpose(1, 2), cos, sin)      # [B, KV, T, dh]
        v = self.v(x).view(B, T, s.kv, s.dh).transpose(1, 2)
        if self.impl == "sdpa":
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:                                                                    # 显式 [B, H, T, T]
            rep = s.heads // s.kv
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            sc = (q @ k.transpose(-1, -2)) / math.sqrt(s.dh)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            a = torch.softmax(sc.masked_fill(~mask, float("-inf")).float(), -1).to(v.dtype) @ v
        return self.o(a.transpose(1, 2).reshape(B, T, -1))
# --8<-- [end:attention]


class Block(nn.Module):
    def __init__(self, s: Shape, impl: str):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(s.d, eps=1e-6), nn.RMSNorm(s.d, eps=1e-6)
        self.attn = Attention(s, impl)
        self.gate, self.up = nn.Linear(s.d, s.ffn, bias=False), nn.Linear(s.d, s.ffn, bias=False)
        self.down = nn.Linear(s.ffn, s.d, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class LM(nn.Module):
    def __init__(self, preset: str, impl: str = "sdpa", seq_len: int = 2048):
        super().__init__()
        self.s = s = Shape(**PRESETS[preset])
        self.emb = nn.Embedding(s.vocab, s.d)
        self.blocks = nn.ModuleList(Block(s, impl) for _ in range(s.layers))
        self.norm = nn.RMSNorm(s.d, eps=1e-6)
        self.head = None if s.tie else nn.Linear(s.d, s.vocab, bias=False)
        inv = 1.0 / (1e6 ** (torch.arange(0, s.dh, 2).float() / s.dh))
        f = torch.outer(torch.arange(seq_len).float(), inv)
        self.register_buffer("cos", torch.cat([f.cos(), f.cos()], -1)[None, None], persistent=False)
        self.register_buffer("sin", torch.cat([f.sin(), f.sin()], -1)[None, None], persistent=False)
        self.ckpt = False
        for m in self.modules():                                                 # Qwen2.5 的 initializer_range = 0.02
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        cos, sin = self.cos[..., :T, :].to(self.emb.weight.dtype), self.sin[..., :T, :].to(self.emb.weight.dtype)
        x = self.emb(ids)                                                        # [B, T, d]
        for b in self.blocks:
            if self.ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, cos, sin, use_reentrant=False)
            else:
                x = b(x, cos, sin)
        x = self.norm(x)
        w = self.emb.weight if self.head is None else self.head.weight
        return x @ w.T                                                           # [B, T, V]


# --8<-- [start:param_counts]
def param_counts(preset: str) -> dict:
    """解析计数（不建模型）。matmul 参数 = 每层 q/k/v/o + gate/up/down，再加输出层（共享时就是 embedding）。"""
    s = Shape(**PRESETS[preset])
    attn = s.d * s.heads * s.dh + 2 * s.d * s.kv * s.dh + s.heads * s.dh * s.d   # q, k, v, o 的权重
    bias = s.heads * s.dh + 2 * s.kv * s.dh
    mlp = 3 * s.d * s.ffn
    per_layer = attn + bias + mlp + 2 * s.d
    emb = s.vocab * s.d
    total = s.layers * per_layer + emb + s.d + (0 if s.tie else emb)
    matmul = s.layers * (attn + mlp) + emb                                       # 输出层算一次 [d, V] 的矩阵乘
    return {"params_total": total, "params_matmul": matmul, "embedding": emb}
# --8<-- [end:param_counts]


# --8<-- [start:flops]
def flops_per_token(preset: str, seq_len: int) -> dict:
    """训练每 token：6 × matmul 参数（前向 2、反向 4）+ 因果 attention 6 × L × T × d（QKᵀ 与 AV 前向各 L·T·d，只算下三角一半）。
    hfu 口径再加一次重计算的前向：+ 2 × matmul + 2 × L × T × d。"""
    s = Shape(**PRESETS[preset])
    pc = param_counts(preset)
    dense = 6 * pc["params_matmul"]
    attn = 6 * s.layers * seq_len * s.d
    return {"model": dense + attn, "dense": dense, "attn": attn, "recompute": 2 * pc["params_matmul"] + 2 * s.layers * seq_len * s.d}
# --8<-- [end:flops]
