"""R06：序列推荐里的 attention 打分。两种打分 × 四种实现。

softmax：  A = softmax(QKᵀ/√D)，O = A V                           —— 标准 attention（SASRec）
pointwise：A = SiLU(QKᵀ/√D + rab[i − j]) / L，O = A V（因果）     —— HSTU 式逐点注意力，SDPA 不支持

实现：naive（PyTorch 逐步，物化 L × L）、sdpa（只 softmax）、compiled（torch.compile(naive)）、triton（融合 kernel，不物化 L × L）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False


# --8<-- [start:naive]
def naive(q, k, v, kind: str, rab: torch.Tensor | None) -> torch.Tensor:
    """q, k, v [B, H, L, D]。会在显存里物化 [B, H, L, L] 的打分、权重等多个中间张量。"""
    B, H, L, D = q.shape
    s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(D))                   # [B, H, L, L]
    causal = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
    if kind == "softmax":
        a = torch.softmax(s.masked_fill(~causal, float("-inf")).float(), -1).to(v.dtype)
    else:
        i = torch.arange(L, device=q.device)
        dist = (i[:, None] - i[None]).clamp(0, rab.numel() - 1)
        s = s + rab.to(s.dtype)[dist]                                        # 相对位置偏置 [L, L]
        a = (F.silu(s.float()) * causal / L).to(v.dtype)
    return a @ v                                                            # [B, H, L, D]
# --8<-- [end:naive]


def sdpa(q, k, v, kind: str, rab=None):
    if kind != "softmax":
        raise ValueError("SDPA 只支持 softmax 打分")
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


_compiled = {}


def compiled(q, k, v, kind: str, rab=None):
    key = kind
    if key not in _compiled:
        _compiled[key] = torch.compile(naive, dynamic=False)
    return _compiled[key](q, k, v, kind, rab)


if HAS_TRITON:
    # --8<-- [start:triton_kernel]
    @triton.jit
    def attn_fwd_kernel(Q, K, V, O, RAB,
                        sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sob, soh, sot,
                        H, L, n_rab, sm_scale, inv_len,
                        D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        POINTWISE: tl.constexpr, DIAG_CAUSAL: tl.constexpr):
        """一个 program 负责一个 (b, h) 的一个 query 块 [BLOCK_M, D]，循环读 K / V 块，L × L 只在片上出现。
        softmax：online softmax（每行维护最大值 m、指数和 l，块之间要重新缩放）；
        pointwise：每块直接 SiLU 后累加，块之间互不依赖——比 softmax 少一组规约。"""
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        b, h = bh // H, bh % H
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqt + offs_d[None, :], mask=offs_m[:, None] < L, other=0.0)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        hi = tl.minimum((pid_m + 1) * BLOCK_M, L)                           # 因果：对角线右侧的整块不读
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_mask = offs_n[:, None] < L                                    # 尾块：越界的行不读
            k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skt + offs_d[None, :], mask=kv_mask, other=0.0)
            v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svt + offs_d[None, :], mask=kv_mask, other=0.0)
            s = tl.dot(q, tl.trans(k)) * sm_scale                           # [BLOCK_M, BLOCK_N]，FP32
            if DIAG_CAUSAL:
                valid = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < L)
            else:                                                           # 教学故障：只靠"不读右侧整块"实现因果，
                valid = offs_n[None, :] < L                                 # 对角块里同块更靠后的 key 也被算进来
            if POINTWISE:
                dist = tl.minimum(tl.maximum(offs_m[:, None] - offs_n[None, :], 0), n_rab - 1)
                s = s + tl.load(RAB + dist).to(tl.float32)
                p = s * tl.sigmoid(s) * inv_len                             # SiLU / L
                p = tl.where(valid, p, 0.0)
                acc += tl.dot(p.to(v.dtype), v)
            else:
                s = tl.where(valid, s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, 1))
                p = tl.exp(s - m_new[:, None])
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new
        if not POINTWISE:
            acc = acc / l_i[:, None]
        tl.store(O + b * sob + h * soh + offs_m[:, None] * sot + offs_d[None, :], acc.to(O.dtype.element_ty), mask=offs_m[:, None] < L)
    # --8<-- [end:triton_kernel]


def triton_attn(q, k, v, kind: str, rab: torch.Tensor | None, block_m=64, block_n=64, num_warps=4, num_stages=2, diag_causal=True):
    assert HAS_TRITON and q.is_cuda and q.shape[-1] in (16, 32, 64, 128)
    B, H, L, D = q.shape
    o = torch.empty_like(q)
    rab_t = rab if rab is not None else torch.zeros(1, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(L, block_m), B * H)
    attn_fwd_kernel[grid](q, k, v, o, rab_t,
                          q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                          v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
                          H, L, rab_t.numel(), 1.0 / math.sqrt(D), 1.0 / L,
                          D=D, BLOCK_M=block_m, BLOCK_N=block_n, POINTWISE=kind == "pointwise", DIAG_CAUSAL=diag_causal,
                          num_warps=num_warps, num_stages=num_stages)
    return o


# --8<-- [start:block_mirror]
def block_mirror(q, k, v, kind: str, rab, block_m: int = 64, block_n: int = 64, diag_causal: bool = True):
    """和 Triton kernel 同样的分块循环，用 PyTorch 写（CPU 上验证算法和故障；一次只算一个 (b, h)）。"""
    B, H, L, D = q.shape
    out = torch.zeros_like(q, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    for b in range(B):
        for h in range(H):
            for m0 in range(0, L, block_m):
                offs_m = torch.arange(m0, m0 + block_m)
                qb = torch.zeros(block_m, D)
                n_q = min(block_m, L - m0)
                qb[:n_q] = q[b, h, m0:m0 + n_q].float()
                acc = torch.zeros(block_m, D)
                m_i = torch.full((block_m,), float("-inf"))
                l_i = torch.zeros(block_m)
                hi = min(m0 + block_m, L)                                   # 因果：右侧整块不读
                for n0 in range(0, hi, block_n):
                    offs_n = torch.arange(n0, n0 + block_n)
                    kb, vb = torch.zeros(block_n, D), torch.zeros(block_n, D)
                    n_k = min(block_n, L - n0)
                    kb[:n_k], vb[:n_k] = k[b, h, n0:n0 + n_k].float(), v[b, h, n0:n0 + n_k].float()
                    s = qb @ kb.T * scale
                    valid = offs_n[None] < L
                    if diag_causal:
                        valid = valid & (offs_m[:, None] >= offs_n[None])
                    if kind == "pointwise":
                        dist = (offs_m[:, None] - offs_n[None]).clamp(0, rab.numel() - 1)
                        s = s + rab.float().cpu()[dist]
                        p = torch.where(valid, torch.nn.functional.silu(s) / L, 0.0)
                        acc += p @ vb
                    else:
                        s = torch.where(valid, s, float("-inf"))
                        m_new = torch.maximum(m_i, s.max(1).values)
                        p = torch.exp(s - m_new[:, None])
                        alpha = torch.exp(m_i - m_new)
                        l_i = l_i * alpha + p.sum(1)
                        acc = acc * alpha[:, None] + p @ vb
                        m_i = m_new
                if kind != "pointwise":
                    acc = acc / l_i[:, None]
                out[b, h, m0:m0 + n_q] = acc[:n_q]
    return out
# --8<-- [end:block_mirror]


# --8<-- [start:accounting]
def attn_flops(B, H, L, D) -> float:
    """因果：QKᵀ 和 AV 各约 B·H·L²·D 次乘加的一半 → 合计 2·B·H·L²·D FLOPs。"""
    return 2.0 * B * H * L * L * D


def attn_bytes(B, H, L, D, esz: int, impl: str) -> float:
    """融合实现：读 Q、K、V、写 O，约 4·B·H·L·D 个元素（忽略 K / V 的重复读）。
    naive：再加上 L × L 中间张量的读写（打分写 → 读、权重写 → 读，至少 4 次 B·H·L² 个元素）。"""
    io = 4 * B * H * L * D * esz
    return io if impl in ("sdpa", "triton") else io + 4 * B * H * L * L * esz
# --8<-- [end:accounting]


def run_impl(impl, q, k, v, kind, rab, cfg, diag_causal=None):
    if impl == "naive":
        return naive(q, k, v, kind, rab)
    if impl == "sdpa":
        return sdpa(q, k, v, kind, rab)
    if impl == "compiled":
        return compiled(q, k, v, kind, rab)
    return triton_attn(q, k, v, kind, rab, cfg["block_m"], cfg["block_n"], cfg["num_warps"], cfg["num_stages"],
                       cfg["diag_causal"] if diag_causal is None else diag_causal)
