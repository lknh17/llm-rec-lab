<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 15 步：代码导读

> 阶段 1 / 6：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① naive

s [B, H, L, L] 先写出来，再 masked_fill / softmax（或加偏置、SiLU），再乘 V。L = 8192、B·H = 32 时一个 bf16 的 L × L 张量就是 4.3 GB。

[`recops/attn.py#L25-L37`](../../../projects/recops/attn.py#L25-L37)

```python
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
```

### ② Triton kernel

hi = min((pid_m+1)·BLOCK_M, L)：右侧整块不读。POINTWISE 分支：s·sigmoid(s)·(1/L) 直接累加；softmax 分支：维护 m_i、l_i，acc 按 alpha 重新缩放。DIAG_CAUSAL 控制对角块里的逐元素 mask。

[`recops/attn.py#L59-L104`](../../../projects/recops/attn.py#L59-L104)

```python
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
```

### ③ 分块镜像

同样的块循环、同样的 mask 逻辑，用 PyTorch 写。测试里它和 naive 对拍（含长度不是块倍数的情形），也复现故障。

[`recops/attn.py#L124-L165`](../../../projects/recops/attn.py#L124-L165)

```python
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
```

### ④ 计数

因果下 QKᵀ 和 AV 各约 B·H·L²·D/2 次乘加。融合实现读写 Q、K、V、O；naive 多出至少 4 次 L × L 的读写。

[`recops/attn.py#L170-L179`](../../../projects/recops/attn.py#L170-L179)

```python
def attn_flops(B, H, L, D) -> float:
    """因果：QKᵀ 和 AV 各约 B·H·L²·D 次乘加的一半 → 合计 2·B·H·L²·D FLOPs。"""
    return 2.0 * B * H * L * L * D


def attn_bytes(B, H, L, D, esz: int, impl: str) -> float:
    """融合实现：读 Q、K、V、写 O，约 4·B·H·L·D 个元素（忽略 K / V 的重复读）。
    naive：再加上 L × L 中间张量的读写（打分写 → 读、权重写 → 读，至少 4 次 B·H·L² 个元素）。"""
    io = 4 * B * H * L * D * esz
    return io if impl in ("sdpa", "triton") else io + 4 * B * H * L * L * esz
```

---

[← 追踪](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
