"""四个序列推荐模型。输入统一是左补齐的物品序列（最近的行为在最右边），0 = padding。

batch：x [B, L] 物品；ts [B, L] 每个行为的时间；ts_next [B, L] 每个位置的"请求时刻"（下一个行为的时间，
       评估时最后一个位置是目标的时间）；y [B, L] 每个位置的下一个物品（0 = 不监督）。
所有模型的输出层都和物品 embedding 共享：logits = 用户表示 · E[c]，对全部物品打分。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

POS_BUCKETS_EXACT = 16


# --8<-- [start:rab]
def pos_bucket(dist: torch.Tensor, n_buckets: int, max_dist: int) -> torch.Tensor:
    """相对距离 → 桶号（T5 式）：0..15 各占一个桶，之后按对数分到剩下的桶，覆盖到 max_dist。"""
    d = dist.clamp(min=0)
    exact = POS_BUCKETS_EXACT
    far = exact + (torch.log(d.float().clamp(min=1) / exact) / math.log(max(max_dist, exact + 1) / exact)
                   * (n_buckets - exact)).long()
    return torch.where(d < exact, d, far.clamp(max=n_buckets - 1))


def time_bucket(dt: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """时间间隔（秒）→ 桶号：⌊log2(1 + Δt)⌋，截到 [0, n_buckets − 1]。1 分钟约 6，1 天约 16，1 年约 25。"""
    return torch.floor(torch.log2(1 + dt.clamp(min=0).float())).long().clamp(0, n_buckets - 1)
# --8<-- [end:rab]


class BucketGather(torch.autograd.Function):
    """out = bias[:, idx]：bias [H, n_buckets]，idx 任意形状 → [H, *idx.shape]。
    前向就是普通索引；反向不走 index 的默认反向（CUDA 上是按下标排序后每个桶一个 warp 串行累加，
    几千万个元素只落进 32-64 个桶时每个 warp 要串行加几十万次，一步要几十秒），
    改用 bincount（桶数 < 100 时是 shared memory 直方图，所有 block 并行）。"""

    @staticmethod
    def forward(ctx, bias: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.n_buckets = bias.shape[1]
        return bias[:, idx]

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        (idx,) = ctx.saved_tensors
        flat = idx.reshape(-1)
        gb = torch.stack([torch.bincount(flat, weights=g[h].reshape(-1).float(), minlength=ctx.n_buckets)
                          for h in range(g.shape[0])])                     # [H, n_buckets]
        return gb.to(g.dtype), None


# --8<-- [start:mask]
def build_mask(valid: torch.Tensor, q0: int, q1: int, mode: str) -> torch.Tensor:
    """query 行 [q0, q1) 对 key 列 [0, q1) 的可见性 [B, 1, q1−q0, q1]。
      causal+pad   因果 + key 是真实行为 + query 是真实行为（正确）
      causal_only  只有因果（教学故障）：左补齐时 padding 在"过去"，因果 mask 挡不住，pad 被当成历史"""
    dev = valid.device
    causal = torch.arange(q0, q1, device=dev)[:, None] >= torch.arange(q1, device=dev)[None]   # [q, k]
    m = causal[None, None]
    if mode == "causal+pad":
        m = m & valid[:, None, None, :q1] & valid[:, None, q0:q1, None]
    elif mode != "causal_only":
        raise ValueError(mode)
    return m
# --8<-- [end:mask]


class MeanPool(nn.Module):
    """用户表示 = 截至当前位置所有历史物品 embedding 的平均（每个位置都监督）。"""

    def __init__(self, V: int, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, b: dict) -> torch.Tensor:
        valid = (b["x"] != 0).unsqueeze(-1)                                 # [B, L, 1]
        e = self.emb(b["x"]) * valid
        u = e.cumsum(1) / valid.cumsum(1).clamp(min=1)                      # 前缀均值 [B, L, d]
        return self.norm(self.proj(u)) @ self.emb.weight.T                  # [B, L, V]


# --8<-- [start:din]
class DIN(nn.Module):
    """target attention：对每个候选 c，按 h_jᵀ W e_c 给历史打分，加权求和得到"针对这个候选"的用户表示。
    a_j(c) = softmax_{j ≤ t}(h_jᵀ W e_c / √d)，logit_t(c) = Σ_{j ≤ t} a_j(c) · h_jᵀ e_c。
    和原版 DIN 一样不带位置信息。没有位置偏置时，"位置 t 只看 j ≤ t"的 softmax 可以用前缀和一次算完所有位置：
      logit_t(c) = Σ_{j≤t} exp(s1[j,c]) · s2[j,c] / Σ_{j≤t} exp(s1[j,c])
    所以每个位置都能监督（和 SASRec 一样多的监督信号）。代价：每个位置对全部 V 个候选各算一份，[B, L, V]。"""

    def __init__(self, V: int, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.d = d
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.W = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, b: dict) -> torch.Tensor:
        x = b["x"]
        h = self.emb(x)                                                     # [B, L, d]
        s1 = ((self.W(h) @ self.emb.weight.T) / math.sqrt(self.d)).float()  # [B, L, V] 历史 j 对候选 c 的注意力打分
        s2 = (h @ self.emb.weight.T).float()                                # [B, L, V] 被加权的值 h_jᵀ e_c
        w = torch.exp(s1.clamp(-30, 30)) * (x != 0)[..., None]              # padding 权重为 0
        num, den = (w * s2).cumsum(1), w.cumsum(1)                          # 前缀和：位置 t 汇总 j ≤ t
        return num / den.clamp(min=1e-20)                                   # [B, L, V]
# --8<-- [end:din]


class Block(nn.Module):
    """SASRec 的一层：RMSNorm → 多头自注意力（softmax）→ 残差；RMSNorm → GELU MLP（4d）→ 残差。每层 12d² 参数。"""

    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.h, self.dh = h, d // h
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, impl: str) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # 每个 [B, H, L, dh]
        if impl == "sdpa":
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)    # 不传 mask，能走 flash 后端
        else:                                                               # 手写：显式 [B, H, L, L]
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
            causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
            a = torch.softmax(s.masked_fill(~causal, float("-inf")), dim=-1) @ v
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.mlp(self.n2(x)))


# --8<-- [start:sasrec]
def to_right(t: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """左补齐 → 右补齐（逐行循环左移 L − n 位）。右补齐时 padding 在"未来"，因果 mask 天然看不到它们，
    SDPA 只需要 is_causal=True、不用传 padding mask，就能走 flash 后端。"""
    L = t.shape[1]
    idx = (torch.arange(L, device=t.device)[None] + (L - n)[:, None]) % L
    return t.gather(1, idx if t.dim() == 2 else idx[..., None].expand_as(t))


def to_left(t: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    L = t.shape[1]
    idx = (torch.arange(L, device=t.device)[None] - (L - n)[:, None]) % L
    return t.gather(1, idx if t.dim() == 2 else idx[..., None].expand_as(t))


class SASRec(nn.Module):
    def __init__(self, V: int, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.impl = cfg["attn_impl_eff"]
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.pos = nn.Embedding(cfg["max_len_eff"], d)                      # 绝对位置（右补齐后从 0 数）
        self.blocks = nn.ModuleList(Block(d, cfg["n_head"], cfg["dropout"]) for _ in range(cfg["n_layer"]))
        self.norm = nn.RMSNorm(d)
        self.drop = nn.Dropout(cfg["dropout"])
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, b: dict) -> torch.Tensor:
        x = b["x"]
        n = (x != 0).sum(1)
        xr = to_right(x, n)                                                 # [B, L] 右补齐
        h = self.drop(self.emb(xr) + self.pos(torch.arange(x.shape[1], device=x.device))[None])
        for blk in self.blocks:
            h = blk(h, self.impl)
        return to_left(self.norm(h), n) @ self.emb.weight.T                 # 转回左补齐 [B, L, V]
# --8<-- [end:sasrec]


# --8<-- [start:hstu_attn]
class HSTULayer(nn.Module):
    """U, V, Q, K = split(SiLU(f1(LN(x))))；A = SiLU(QKᵀ/√dh + rab) · mask / n；out = x + f2(LN(A·V) ⊙ U)。
    没有 softmax：权重逐点独立，不按行归一化，所以也不需要 online softmax，按 query 分块计算时每块互不依赖。"""

    def __init__(self, d: int, h: int, cfg: dict):
        super().__init__()
        self.h, self.dh = h, d // h
        self.norm_in, self.norm_attn = nn.LayerNorm(d), nn.LayerNorm(d)
        self.f1, self.f2 = nn.Linear(d, 4 * d), nn.Linear(d, d)
        self.nb, self.nt, self.max_len = cfg["pos_buckets"], cfg["time_buckets"], cfg["max_len_eff"]
        self.use_time, self.mask_mode = cfg["time_bias"], cfg["mask"]
        self.pos_bias = nn.Parameter(torch.zeros(h, self.nb))
        self.time_bias = nn.Parameter(torch.zeros(h, self.nt))
        self.last_attn: torch.Tensor | None = None                         # 诊断用：最后一个位置的 |A| [B, L]

    def attend(self, q, k, v, b, valid, n, q0, q1):
        """query 行 [q0, q1) 对 key [0, q1)：[B, H, q, dh]。"""
        s = (q[:, :, q0:q1] @ k[:, :, :q1].transpose(-1, -2)) / math.sqrt(self.dh)   # [B, H, q, k]
        i, j = torch.arange(q0, q1, device=q.device), torch.arange(q1, device=q.device)
        s = s + BucketGather.apply(self.pos_bias, pos_bucket(i[:, None] - j[None], self.nb, self.max_len)).to(s.dtype)[None]
        if self.use_time:                                                   # Δt = 请求时刻 − 行为时刻
            dt = b["ts_next"][:, q0:q1, None] - b["ts"][:, None, :q1]       # [B, q, k]
            s = s + BucketGather.apply(self.time_bias, time_bucket(dt, self.nt)).to(s.dtype).permute(1, 0, 2, 3)
        a = F.silu(s) * build_mask(valid, q0, q1, self.mask_mode) / n      # mask 乘在 SiLU 之后
        if q1 == valid.shape[1]:
            self.last_attn = a[:, :, -1].abs().sum(1).detach()
        return a @ v[:, :, :q1]

    def forward(self, x: torch.Tensor, b: dict, valid: torch.Tensor, impl: str, chunk: int) -> torch.Tensor:
        B, L, d = x.shape
        u, v, q, k = F.silu(self.f1(self.norm_in(x))).view(B, L, 4, self.h, self.dh).permute(2, 0, 3, 1, 4)
        n = valid.sum(1).clamp(min=1).view(B, 1, 1, 1).to(x.dtype)
        if impl == "manual":
            o = self.attend(q, k, v, b, valid, n, 0, L)
        else:                                                               # chunked：每块单独 checkpoint，只存一块的 [q, k]
            outs = []
            for q0 in range(0, L, chunk):
                q1 = min(L, q0 + chunk)
                outs.append(checkpoint(self.attend, q, k, v, b, valid, n, q0, q1, use_reentrant=False)
                            if torch.is_grad_enabled() else self.attend(q, k, v, b, valid, n, q0, q1))
            o = torch.cat(outs, 2)
        o = o.transpose(1, 2).reshape(B, L, d)
        return x + self.f2(self.norm_attn(o) * u.transpose(1, 2).reshape(B, L, d))
# --8<-- [end:hstu_attn]


class HSTU(nn.Module):
    def __init__(self, V: int, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.impl, self.chunk = cfg["attn_impl_eff"], cfg["attn_chunk"]
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.layers = nn.ModuleList(HSTULayer(d, cfg["n_head"], cfg) for _ in range(cfg["n_layer"]))
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg["dropout"])
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, b: dict) -> torch.Tensor:
        valid = b["x"] != 0
        h = self.drop(self.emb(b["x"]))
        for layer in self.layers:
            h = layer(h, b, valid, self.impl, self.chunk)
        return self.norm(h) @ self.emb.weight.T                              # [B, L, V]


MODELS = {"mean": MeanPool, "din": DIN, "sasrec": SASRec, "hstu": HSTU}


def resolve_impl(cfg: dict) -> str:
    """auto：SASRec 走 SDPA（flash）；HSTU 按 query 分块（10K 时整块的 [B, H, L, L] 加上时间桶下标要几十 GB）。"""
    if cfg["attn_impl"] != "auto":
        return cfg["attn_impl"]
    return {"sasrec": "sdpa", "hstu": "chunked"}.get(cfg["model"], "none")


def build(V: int, cfg: dict) -> nn.Module:
    return MODELS[cfg["model"]](V, cfg)


def param_count(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = model.emb.weight.numel()
    return {"total": total, "trainable": total, "embedding": emb, "body": total - emb,
            "embedding_share": round(emb / total, 4), "train_mode": "full"}
