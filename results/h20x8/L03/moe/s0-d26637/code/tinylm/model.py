"""小 GPT：Pre-RMSNorm、RoPE、SwiGLU、输入输出 embedding 共享（tied）。和 Qwen2 / Llama 同一族结构，只是小。

L03 起可以换结构（默认值就是 L01 的结构，默认路径的模块名和创建顺序不变，同种子初始化逐位一致）：
  注意力  n_kv_head < n_head → GQA（=1 是 MQA）；kv_lora_rank > 0 → MLA（KV 压成一个低秩潜向量 + 共享的 RoPE key）
  位置    pos_emb = rope / none（NoPE，只靠因果 mask 隐式感知位置）/ learned（绝对位置 embedding）
  Norm    norm_pos = pre（x + f(norm(x))）/ post（norm(x + f(x))，原始 Transformer）
  FFN     ffn = dense / moe（moe_experts 个 SwiGLU 专家，每 token 选 top-k，Switch 负载均衡损失）
L04 起：doc_mask = True 时按分隔符（EOS）分段，块对角因果 mask，每篇文档的位置从 0 重新计。

形状约定：B = batch，T = 序列长度，C = d_model，H = 头数，Hkv = KV 头数，D = C / H，V = 词表（补齐到 64 的倍数），
E = 专家数，K = top-k，R = MLA 潜向量维度。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def padded_vocab(v: int, multiple: int = 64) -> int:
    """词表补齐到 64 的倍数：矩阵乘的维度对齐，GPU 上更快；多出来的几行永远不会被选中。"""
    return (v + multiple - 1) // multiple * multiple


def _get(cfg: dict, key: str, default):
    """旧 checkpoint / 旧测试的 cfg 里没有 L02 以后的新键：取默认值（= L01 的结构）。"""
    v = cfg.get(key, default)
    return default if v is None else v


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
    """x: [B, H, T, D]；把每个头的向量两两一组按位置旋转。
    cos/sin: [T_max, D/2]（所有样本位置相同）或 [B, T, D/2]（doc_mask 时每篇文档位置从 0 重新计）。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    if cos.dim() == 3:
        c, s = cos[:, None, :T].to(x.dtype), sin[:, None, :T].to(x.dtype)       # [B, 1, T, D/2]
    else:
        c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)                          # [T, D/2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


# --8<-- [start:docmask]
def doc_layout(idx: torch.Tensor, eos_id: int):
    """按分隔符分段：idx [B, T] → 文档编号 doc [B, T]、文档内位置 pos [B, T]、注意力 mask [B, 1, T, T]（True = 可见）。
    token 流是 [EOS, 文档1, EOS, 文档2, ...]，EOS 当作下一篇文档的开头（它负责预测下一篇的第一个 token）。"""
    B, T = idx.shape
    doc = (idx == eos_id).long().cumsum(dim=1)                                  # [B, T]
    ar = torch.arange(T, device=idx.device).expand(B, T)
    start = torch.where(idx == eos_id, ar, torch.zeros_like(ar)).cummax(dim=1).values   # 每个位置所在文档的起点
    pos = ar - start                                                            # [B, T]
    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    mask = (doc[:, :, None] == doc[:, None, :]) & causal                        # [B, T, T]
    return doc, pos, mask[:, None]
# --8<-- [end:docmask]


class MoEFFN(nn.Module):
    """E 个 SwiGLU 专家 + 线性路由。每个 token 选 top-k 个专家，输出按路由概率（重新归一化）加权求和。"""

    def __init__(self, C: int, d_ff: int, E: int, K: int):
        super().__init__()
        self.E, self.K = E, K
        self.router = nn.Linear(C, E, bias=False)
        self.w_gate_up = nn.Parameter(torch.empty(E, C, 2 * d_ff))              # [E, C, 2F]
        self.w_down = nn.Parameter(torch.empty(E, d_ff, C))                     # [E, F, C]
        self.aux_loss = torch.zeros(())
        self.last_load = None

    # --8<-- [start:moe]
    def forward(self, x):
        B, T, C = x.shape
        h = x.reshape(-1, C)                                                    # [N, C]，N = B × T
        probs = self.router(h).float().softmax(dim=-1)                          # [N, E]
        topv, topi = probs.topk(self.K, dim=-1)                                 # [N, K]
        topv = topv / topv.sum(-1, keepdim=True)                                # 选中的 K 个重新归一化
        out = torch.zeros_like(h)
        for e in range(self.E):                                                 # 朴素逐专家循环（grouped GEMM 见 S3 第 9 课）
            tok, slot = (topi == e).nonzero(as_tuple=True)                      # 选中专家 e 的 token 和它是第几选择
            if tok.numel() == 0:
                continue
            g, u = (h[tok] @ self.w_gate_up[e].to(h.dtype)).chunk(2, dim=-1)    # [n_e, F] × 2
            y = (F.silu(g) * u) @ self.w_down[e].to(h.dtype)                    # [n_e, C]
            out.index_add_(0, tok, (y * topv[tok, slot, None].to(y.dtype)).to(out.dtype))
        # Switch Transformer 负载均衡损失：E × Σ_e f_e × P_e，f_e = 分到专家 e 的比例，P_e = 平均路由概率。
        # 完全均衡时等于 1；坍缩到一个专家时趋向 E。
        load = torch.bincount(topi.flatten(), minlength=self.E).float() / topi.numel()   # [E]，和为 1
        self.aux_loss = self.E * (load.detach() * probs.mean(0)).sum()
        self.last_load = load.detach()
        self.last_entropy = (-(probs * probs.clamp_min(1e-9).log()).sum(-1).mean() / math.log(self.E)).detach()
        return out.view(B, T, C)
    # --8<-- [end:moe]


class Block(nn.Module):
    def __init__(self, C: int, H: int, d_ff: int, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}
        self.H, self.D = H, C // H
        self.Hkv = _get(cfg, "n_kv_head", 0) or H
        self.R = _get(cfg, "kv_lora_rank", 0)
        self.Dr = _get(cfg, "mla_rope_dim", 32)
        self.post = _get(cfg, "norm_pos", "pre") == "post"
        self.use_rope = _get(cfg, "pos_emb", "rope") == "rope"
        if H % self.Hkv:
            raise ValueError(f"n_head={H} 不能被 n_kv_head={self.Hkv} 整除")
        # 默认路径的模块创建顺序必须是 ln1、ln2、qkv、proj、gate_up、down（初始化的随机数顺序和 L01 一致）
        self.ln1, self.ln2 = RMSNorm(C), RMSNorm(C)
        if self.R:                                                              # MLA
            self.q = nn.Linear(C, H * (self.D + self.Dr), bias=False)          # 每个头：D 维内容 + Dr 维位置
            self.kv_down = nn.Linear(C, self.R + self.Dr, bias=False)          # 潜向量 R 维 + 共享 RoPE key Dr 维（缓存这些）
            self.kv_norm = RMSNorm(self.R)
            self.kv_up = nn.Linear(self.R, 2 * H * self.D, bias=False)         # 潜向量 → 每个头的 k_nope、v
        else:                                                                   # MHA / GQA / MQA
            self.qkv = nn.Linear(C, C + 2 * self.Hkv * self.D, bias=False)
        self.proj = nn.Linear(C, C, bias=False)
        if _get(cfg, "ffn", "dense") == "moe":
            K = _get(cfg, "moe_top_k", 2)
            self.moe = MoEFFN(C, _get(cfg, "moe_d_ff", 0) or d_ff // K, _get(cfg, "moe_experts", 8), K)
        else:
            self.gate_up = nn.Linear(C, 2 * d_ff, bias=False)
            self.down = nn.Linear(d_ff, C, bias=False)

    # --8<-- [start:attention]
    def attn(self, x, cos, sin, mask):
        B, T, C = x.shape
        if self.R:
            return self.mla(x, cos, sin, mask)
        q, k, v = self.qkv(x).split([C, self.Hkv * self.D, self.Hkv * self.D], dim=-1)
        q = q.view(B, T, self.H, self.D).transpose(1, 2)                       # [B, H, T, D]
        k = k.view(B, T, self.Hkv, self.D).transpose(1, 2)                     # [B, Hkv, T, D]：KV cache 存的就是 k、v
        v = v.view(B, T, self.Hkv, self.D).transpose(1, 2)                     # [B, Hkv, T, D]
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        gqa = self.Hkv != self.H                                                # 每 H / Hkv 个查询头共享一组 KV
        if mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
        else:                                                                   # 显式 mask：走不了 flash kernel
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)
        return a.transpose(1, 2).reshape(B, T, C)                               # [B, T, C]
    # --8<-- [end:attention]

    # --8<-- [start:mla]
    def mla(self, x, cos, sin, mask):
        """Multi-head Latent Attention（DeepSeek-V2）：KV cache 只存 R 维潜向量 c 和 Dr 维共享 RoPE key。
        推理时 kv_up 可以吸收进 q 和 proj，不必展开成每个头的 k、v；这里训练时直接展开。"""
        B, T, C = x.shape
        H, D, Dr = self.H, self.D, self.Dr
        q = self.q(x).view(B, T, H, D + Dr).transpose(1, 2)                    # [B, H, T, D + Dr]
        q_nope, q_rope = q.split([D, Dr], dim=-1)
        c, k_rope = self.kv_down(x).split([self.R, Dr], dim=-1)                 # [B, T, R]、[B, T, Dr]
        k_nope, v = self.kv_up(self.kv_norm(c)).view(B, T, H, 2 * D).transpose(1, 2).split(D, dim=-1)  # [B, H, T, D] × 2
        k_rope = k_rope[:, None]                                                # [B, 1, T, Dr]，所有头共享
        if self.use_rope:
            cr, sr = (cos[..., : Dr // 2], sin[..., : Dr // 2])
            q_rope, k_rope = apply_rope(q_rope, cr, sr), apply_rope(k_rope, cr, sr)
        qf = torch.cat([q_nope, q_rope], dim=-1)                                # [B, H, T, D + Dr]
        kf = torch.cat([k_nope, k_rope.expand(B, H, T, Dr)], dim=-1)           # [B, H, T, D + Dr]
        if mask is None:
            a = F.scaled_dot_product_attention(qf, kf, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(qf, kf, v, attn_mask=mask)
        return a.transpose(1, 2).reshape(B, T, C)
    # --8<-- [end:mla]

    def ffn(self, x):
        if hasattr(self, "moe"):
            return self.moe(x)
        g, u = self.gate_up(x).chunk(2, dim=-1)                                 # 2 × [B, T, d_ff]
        return self.down(F.silu(g) * u)

    # --8<-- [start:norm]
    def forward(self, x, cos, sin, mask=None):
        if self.post:                                                           # Post-Norm：残差流每层都被归一化
            x = self.ln1(x + self.proj(self.attn(x, cos, sin, mask)))
            return self.ln2(x + self.ffn(x))
        x = x + self.proj(self.attn(self.ln1(x), cos, sin, mask))               # Pre-Norm：残差流不经过 norm
        return x + self.ffn(self.ln2(x))
    # --8<-- [end:norm]


# --8<-- [start:gpt]
class GPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        C, H = cfg["d_model"], cfg["n_head"]
        if C % H:
            raise ValueError(f"d_model={C} 不能被 n_head={H} 整除")
        self.vocab_real, self.V = vocab_size, padded_vocab(vocab_size)
        self.wte = nn.Embedding(self.V, C)                                 # [V, C]，同时当输出层（tied）
        self.blocks = nn.ModuleList(Block(C, H, cfg["d_ff"], cfg) for _ in range(cfg["n_layer"]))
        self.ln_f = RMSNorm(C)
        self.T, self.D = cfg["seq_len"], C // H
        self.pos_emb = _get(cfg, "pos_emb", "rope")
        self.doc_mask, self.eos_id = bool(_get(cfg, "doc_mask", False)), cfg.get("eos_id")
        if self.pos_emb == "learned":
            self.wpe = nn.Embedding(self.T, C)                             # [T, C]：只认得训练长度以内的位置
        self.apply(self._init)
        for b in self.blocks:  # 残差分支的输出层按层数缩小初始化（GPT-2 的做法），深层时激活不随层数放大
            for w in (b.proj.weight, b.moe.w_down if hasattr(b, "moe") else b.down.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * cfg["n_layer"]))
            if hasattr(b, "moe"):
                nn.init.normal_(b.moe.w_gate_up, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        """idx: [B, T] token id → logits [B, T, V]。"""
        T = idx.shape[1]
        mask, pos = None, None
        if self.doc_mask:
            _, pos, mask = doc_layout(idx, self.eos_id)                    # pos [B, T]，mask [B, 1, T, T]
        cos, sin = rope_cache(T, self.D, idx.device)                       # [T, D/2]
        if pos is not None:
            cos, sin = cos[pos], sin[pos]                                  # [B, T, D/2]：每篇文档从位置 0 开始
        x = self.wte(idx)                                                  # [B, T, C]
        if self.pos_emb == "learned":
            p = pos if pos is not None else torch.arange(T, device=idx.device)
            x = x + self.wpe(p)
        for b in self.blocks:
            x = b(x, cos, sin, mask)
        return self.ln_f(x) @ self.wte.weight.T                            # [B, T, V]
# --8<-- [end:gpt]

    def aux_loss(self):
        """MoE 各层负载均衡损失的平均（dense 模型为 0）。"""
        ls = [b.moe.aux_loss for b in self.blocks if hasattr(b, "moe")]
        return torch.stack(ls).mean() if ls else None

    def moe_stats(self) -> dict | None:
        """最近一个 batch 各层的负载（层平均）：每个专家分到的比例、最大 / 平均、路由熵（归一化到 0-1）。"""
        ms = [b.moe for b in self.blocks if hasattr(b, "moe") and b.moe.last_load is not None]
        if not ms:
            return None
        load = torch.stack([m.last_load for m in ms])                      # [L, E]
        return {"load": load.cpu().tolist(), "max_over_mean": float((load.max(1).values * load.shape[1]).mean()),
                "entropy": float(torch.stack([m.last_entropy for m in ms]).mean())}


def param_counts(model: GPT) -> dict:
    """embedding（= 输出层，共享）和 Transformer 主体各多少参数；MoE 时另算每 token 实际参与计算的（激活）参数。"""
    emb = model.wte.weight.numel()
    total = sum(p.numel() for p in model.parameters())
    b0 = model.blocks[0]
    attn = sum(p.numel() for n, p in b0.named_parameters() if n.split(".")[0] in ("qkv", "q", "kv_down", "kv_norm", "kv_up", "proj"))
    ffn = sum(p.numel() for n, p in b0.named_parameters() if n.split(".")[0] in ("gate_up", "down", "moe"))
    inactive = 0
    if hasattr(b0, "moe"):
        per_expert = b0.moe.w_gate_up[0].numel() + b0.moe.w_down[0].numel()
        inactive = (b0.moe.E - b0.moe.K) * per_expert * len(model.blocks)
    out = {"total": total, "trainable": total, "embedding": emb, "body": total - emb,
           "embedding_share": round(emb / total, 4), "vocab_padded": model.V, "vocab_real": model.vocab_real,
           "attn_per_layer": attn, "ffn_per_layer": ffn, "active": total - inactive, "active_body": total - emb - inactive}
    if hasattr(model, "wpe"):
        out["pos_emb"] = model.wpe.weight.numel()
    return out


def flops_per_token(model: GPT, cfg: dict) -> dict:
    """训练每 token 的 FLOPs（前向 + 反向 ≈ 3 × 前向）。矩阵乘：前向 2 × 激活参数；注意力：QKᵀ 和 AV 各 2 × T × (H × d) × 层数，
    因果只算一半。embedding 查表不算 FLOPs，但共享权重的输出层（C × V 的矩阵乘）要算。
    另给 6ND 口径（N = 全部参与矩阵乘的参数，含共享的输出层）：两者之比 = 1 + 注意力那部分。"""
    pc = param_counts(model)
    body = pc["active_body"] - pc.get("pos_emb", 0)
    head = model.V * cfg["d_model"]
    b0 = model.blocks[0]
    d_qk = b0.D + (b0.Dr if b0.R else 0)
    attn = 2 * cfg["seq_len"] * b0.H * (d_qk + b0.D) * len(model.blocks) / 2  # 因果：平均只看一半位置
    fwd = 2 * body + 2 * head + attn
    return {"train_flops_per_token": 3 * fwd, "lm_head_share": round(2 * head / fwd, 4),
            "body_share": round(2 * body / fwd, 4), "attn_share": round(attn / fwd, 4),
            "six_n_per_token": 6 * (body + head)}


# --8<-- [start:kvcache]
def kv_cache_stats(model: GPT, bytes_per_elem: int = 2) -> dict:
    """推理时每个 token 要缓存多少：MHA/GQA 存 2 × Hkv × D（k 和 v），MLA 存 R + Dr（潜向量 + 共享 RoPE key）。bf16 每元素 2 字节。"""
    b0 = model.blocks[0]
    per_layer = (b0.R + b0.Dr) if b0.R else 2 * b0.Hkv * b0.D
    mha = 2 * b0.H * b0.D
    return {"elems_per_token_layer": per_layer, "bytes_per_token": per_layer * len(model.blocks) * bytes_per_elem,
            "ratio_vs_mha": round(per_layer / mha, 4), "kind": "mla" if b0.R else ("mha" if b0.Hkv == b0.H else
                                                                                   ("mqa" if b0.Hkv == 1 else "gqa"))}
# --8<-- [end:kvcache]
