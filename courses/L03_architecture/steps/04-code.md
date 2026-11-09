<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一层注意力](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 15 步：代码导读

> 阶段 1 / 7：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① MHA / GQA / MQA

qkv 投影的输出是 C + 2 × Hkv × D：查询头 H 个，KV 头 Hkv 个。SDPA 的 enable_gqa 让每 H / Hkv 个查询头共用一组 k、v，不必复制。

[`tinylm/model.py#L143-L158`](../../../projects/tinylm/model.py#L143-L158)

```python
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
```

### ② MLA

kv_down 把 x 压成 R + Dr 维（这就是缓存的全部）；kv_up 再展开成每个头的 k_nope、v。q 的每个头也分成 D 维内容 + Dr 维位置，和 [k_nope, k_rope] 拼起来算内积。

[`tinylm/model.py#L162-L181`](../../../projects/tinylm/model.py#L162-L181)

```python
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
```

### ③ Pre-Norm 与 Post-Norm

只差一行：norm 包住 f 的输入，还是包住残差相加之后的结果。

[`tinylm/model.py#L191-L196`](../../../projects/tinylm/model.py#L191-L196)

```python
    def forward(self, x, cos, sin, mask=None):
        if self.post:                                                           # Post-Norm：残差流每层都被归一化
            x = self.ln1(x + self.proj(self.attn(x, cos, sin, mask)))
            return self.ln2(x + self.ffn(x))
        x = x + self.proj(self.attn(self.ln1(x), cos, sin, mask))               # Pre-Norm：残差流不经过 norm
        return x + self.ffn(self.ln2(x))
```

### ④ MoE 路由与负载均衡

probs [N, E] → topk [N, K] → 重新归一化 → 逐专家 gather / scatter。aux = E × Σ f_e × P_e：f 不可导（计数），梯度通过 P 推着路由把概率分给冷门专家。

[`tinylm/model.py#L89-L109`](../../../projects/tinylm/model.py#L89-L109)

```python
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
```

### ⑤ KV cache 字节数

MHA / GQA：2 × Hkv × D；MLA：R + Dr。乘层数、乘 2 字节（bf16）。

[`tinylm/model.py#L297-L304`](../../../projects/tinylm/model.py#L297-L304)

```python
def kv_cache_stats(model: GPT, bytes_per_elem: int = 2) -> dict:
    """推理时每个 token 要缓存多少：MHA/GQA 存 2 × Hkv × D（k 和 v），MLA 存 R + Dr（潜向量 + 共享 RoPE key）。bf16 每元素 2 字节。"""
    b0 = model.blocks[0]
    per_layer = (b0.R + b0.Dr) if b0.R else 2 * b0.Hkv * b0.D
    mha = 2 * b0.H * b0.D
    return {"elems_per_token_layer": per_layer, "bytes_per_token": per_layer * len(model.blocks) * bytes_per_elem,
            "ratio_vs_mha": round(per_layer / mha, 4), "kind": "mla" if b0.R else ("mha" if b0.Hkv == b0.H else
                                                                                   ("mqa" if b0.Hkv == 1 else "gqa"))}
```

---

[← 追踪一层注意力](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
