"""小 GPT：Pre-RMSNorm、RoPE、SwiGLU、输入输出 embedding 共享（tied）。和 Qwen2 / Llama 同一族结构，只是小。

形状约定：B = batch，T = 序列长度，C = d_model，H = 头数，D = C / H，V = 词表（补齐到 64 的倍数）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def padded_vocab(v: int, multiple: int = 64) -> int:
    """词表补齐到 64 的倍数：矩阵乘的维度对齐，GPU 上更快；多出来的几行永远不会被选中。"""
    return (v + multiple - 1) // multiple * multiple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def rope_cache(T: int, D: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=device).float() / D))  # [D/2]
    ang = torch.outer(torch.arange(T, device=device).float(), inv)              # [T, D/2]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, T, D]；把每个头的向量两两一组按位置旋转。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, C: int, H: int, d_ff: int):
        super().__init__()
        self.H = H
        self.ln1, self.ln2 = RMSNorm(C), RMSNorm(C)
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)
        self.gate_up = nn.Linear(C, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, C, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=-1)                  # 3 × [B, T, C]
        q, k, v = (t.view(B, T, self.H, C // self.H).transpose(1, 2) for t in (q, k, v))  # [B, H, T, D]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # [B, H, T, D]
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        g, u = self.gate_up(self.ln2(x)).chunk(2, dim=-1)                 # 2 × [B, T, d_ff]
        return x + self.down(F.silu(g) * u)


# --8<-- [start:gpt]
class GPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        C, H = cfg["d_model"], cfg["n_head"]
        if C % H:
            raise ValueError(f"d_model={C} 不能被 n_head={H} 整除")
        self.vocab_real, self.V = vocab_size, padded_vocab(vocab_size)
        self.wte = nn.Embedding(self.V, C)                                 # [V, C]，同时当输出层（tied）
        self.blocks = nn.ModuleList(Block(C, H, cfg["d_ff"]) for _ in range(cfg["n_layer"]))
        self.ln_f = RMSNorm(C)
        self.T, self.D = cfg["seq_len"], C // H
        self.apply(self._init)
        for b in self.blocks:  # 残差分支的输出层按层数缩小初始化（GPT-2 的做法），深层时激活不随层数放大
            for w in (b.proj.weight, b.down.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        """idx: [B, T] token id → logits [B, T, V]。"""
        cos, sin = rope_cache(idx.shape[1], self.D, idx.device)
        x = self.wte(idx)                                                  # [B, T, C]
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.ln_f(x) @ self.wte.weight.T                            # [B, T, V]
# --8<-- [end:gpt]


def param_counts(model: GPT) -> dict:
    """embedding（= 输出层，共享）和 Transformer 主体各多少参数。"""
    emb = model.wte.weight.numel()
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "trainable": total, "embedding": emb, "body": total - emb,
            "embedding_share": round(emb / total, 4), "vocab_padded": model.V, "vocab_real": model.vocab_real}


def flops_per_token(model: GPT, cfg: dict) -> dict:
    """训练每 token 的 FLOPs（前向 + 反向 ≈ 3 × 前向）。矩阵乘：前向 2 × 参数；注意力：前向 2 × 2 × T × C × 层数（QK 与 AV）。
    embedding 查表不算 FLOPs，但共享权重的输出层（C × V 的矩阵乘）要算。"""
    pc = param_counts(model)
    body, head = pc["body"], model.V * cfg["d_model"]
    attn = 2 * 2 * cfg["seq_len"] * cfg["d_model"] * cfg["n_layer"] / 2  # 因果：平均只看一半位置
    fwd = 2 * body + 2 * head + attn
    return {"train_flops_per_token": 3 * fwd, "lm_head_share": round(2 * head / fwd, 4),
            "body_share": round(2 * body / fwd, 4), "attn_share": round(attn / fwd, 4)}
