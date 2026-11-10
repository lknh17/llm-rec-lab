<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一个用户](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 17 步：代码导读

> 阶段 1 / 6：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 合成长序列

回访簇从长期偏好排在后一半的簇里挑，背景几乎不碰它；回访从序列末尾往前每隔约 P 步写一段 [τ, 8 个回访簇物品]。

[`recseq/synth.py#L38-L89`](../../../projects/recseq/synth.py#L38-L89)

```python
def gen_user(rng: np.random.Generator, cfg: dict, sc: dict, zipf_p: np.ndarray) -> dict:
    K, m, b = cfg["synth_clusters"], cfg["synth_items_per_cluster"], cfg["synth_burst"]
    N = int(np.exp(rng.uniform(np.log(sc["len_min"]), np.log(sc["len_max"] + 1))))
    pi = rng.dirichlet(np.full(K, cfg["synth_alpha"]))                     # 长期偏好：集中在少数几个簇
    # 会话：几何分布的会话长度，标出每个会话的起点
    sess = rng.geometric(1.0 / cfg["synth_session_mean"], size=N)
    starts = np.cumsum(np.r_[0, sess])[:-1]
    is_start = np.zeros(N, bool)
    is_start[starts[starts < N]] = True
    # 簇：会话起点或以 1 − stay 的概率换簇；每段从 π_u 抽一个簇
    change = is_start | (rng.random(N) > cfg["synth_stay"])
    change[0] = True
    seg = np.cumsum(change) - 1
    cl = rng.choice(K, size=seg[-1] + 1, p=pi)[seg]
    rank = rng.choice(m, size=N, p=zipf_p)
    items = (FIRST_BG + cl * m + rank).astype(np.int32)
    burst = np.zeros(N, np.int8)                                           # 0 = 背景；1 = τ；j + 1 = 回访串第 j 个
    # 长程信号：周期 P ~ LogUniform，至少放得下两段
    p_hi = max(sc["p_min"] + 1, min(sc["p_max"], N - 2 * (b + 1)))
    P = int(np.exp(rng.uniform(np.log(sc["p_min"]), np.log(p_hi))))
    tau = int(rng.integers(1, N_TAU + 1))
    rare = np.flatnonzero(pi < np.sort(pi)[K // 2])                         # 长期偏好排在后一半的簇
    f = int(rng.choice(rare)) if len(rare) else int(np.argmin(pi))
    pos, seg_starts = N - (b + 1), []
    while pos >= 0:
        seq = np.r_[tau, FIRST_BG + f * m + rng.choice(m, size=b, p=zipf_p)].astype(np.int32)
        noisy = rng.random(b) < cfg["synth_noise"]                       # 以概率 ε 把回访里的物品换成随机物品
        seq[1:][noisy] = rng.integers(FIRST_BG, FIRST_BG + K * m, size=int(noisy.sum()))
        items[pos:pos + b + 1] = seq
        burst[pos:pos + b + 1] = np.arange(1, b + 2)
        is_start[pos] = True                                               # 每段回访都是一个新会话
        is_start[pos + 1:pos + b + 1] = False
        seg_starts.append(pos)
        pos -= max(b + 2, int(round(P * (1 + rng.uniform(-0.05, 0.05)))))
    gaps = np.where(is_start, rng.lognormal(np.log(86400), 1.0, size=N), rng.exponential(120.0, size=N))
    ts = (1_000_000_000 + np.cumsum(1 + gaps)).astype(np.int64)                # 间隔至少 1 秒
    return {"items": items, "ts": ts, "is_start": is_start, "burst": burst, "N": N, "P": P, "tau": tau,
            "bursts": sorted(seg_starts), "revisit_cluster": f}


def generate(cfg: dict, n_users: int, seed: int) -> dict:
    """n_users 个用户拼成扁平数组 + 偏移量。"""
    rng = np.random.default_rng(seed)
    sc = scaled(cfg)
    m = cfg["synth_items_per_cluster"]
    zipf_p = 1.0 / np.arange(1, m + 1)
    zipf_p /= zipf_p.sum()
    users = [gen_user(rng, cfg, sc, zipf_p) for _ in range(n_users)]
    off = np.r_[0, np.cumsum([u["N"] for u in users])].astype(np.int64)
    cat = {k: np.concatenate([u[k] for u in users]) for k in ("items", "ts", "is_start", "burst")}
    return {**cat, "off": off, "N": off[1:] - off[:-1], "P": np.array([u["P"] for u in users]),
            "tau": np.array([u["tau"] for u in users])}
```

### ② 相对位置 / 时间桶

位置桶覆盖到 10240；时间桶 ⌊log2(1+Δt)⌋：1 分钟约 6、1 天约 16。Δt 用的是请求时刻（下一个行为的时间），新会话的第一个位置能看到"隔了一天"。

[`recseq/models.py#L21-L32`](../../../projects/recseq/models.py#L21-L32)

```python
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
```

### ③ HSTU 注意力

SiLU(QKᵀ/√dh + rab) · mask / n：没有 softmax，每个 (i, j) 独立。按 query 分块时每块只需要 [B, H, 1024, L] 的中间结果，块之间不需要 online softmax 的 max / sum 规约。

[`recseq/models.py#L182-L224`](../../../projects/recseq/models.py#L182-L224)

```python
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
```

### ④ mask

三个条件取与：因果、key 有效、query 有效。只保留因果（causal_only）就是本课的故障。

[`recseq/models.py#L58-L69`](../../../projects/recseq/models.py#L58-L69)

```python
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
```

### ⑤ SASRec 的右补齐

SDPA 传了 mask 就走不了 flash。先把左补齐转成右补齐，padding 到了"未来"一侧，只用 is_causal 就够。

[`recseq/models.py#L143-L177`](../../../projects/recseq/models.py#L143-L177)

```python
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
```

### ⑥ 成本探针

随机输入、B = 1，预热后用 CUDA event 计 5 次前向 + 反向的平均毫秒；每次前清零峰值显存统计；OOM 记为 oom。

[`recseq/cost.py#L86-L123`](../../../projects/recseq/cost.py#L86-L123)

```python
def probe(cfg: dict, V: int, L: int, impl: str, device, B: int = 1) -> dict:
    """随机输入、B 条长度 L 的序列，前向 + 反向的平均毫秒和峰值显存（CUDA 上）；放不下记 oom。"""
    c = {**cfg, "attn_impl_eff": impl, "max_len_eff": max(L, cfg["max_len_eff"])}
    out = {"ms": None, "peak_gb": None, "est_gb": est_attn_gb(c, L, B) if impl == "manual" else None, "oom": False}
    try:
        model = build(V, c).to(device)
        b = _fake_batch(V, L, B, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        out["ms"] = round(_time(lambda: _fwd_bwd(model, b, device, c), device), 3)
        if device.type == "cuda":
            out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except torch.cuda.OutOfMemoryError:
        out["oom"] = True
    finally:
        model = b = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def jagged_probe(cfg: dict, V: int, lengths: list[int], device) -> dict:
    """同一批用户：补齐到最长后整批前向 + 反向，对比逐条不补齐分别算再求和。"""
    L = max(lengths)
    c = {**cfg, "max_len_eff": max(L, cfg["max_len_eff"])}
    try:
        model = build(V, c).to(device)
        padded = _fake_batch(V, L, len(lengths), device, lengths)
        singles = [_fake_batch(V, n, 1, device) for n in lengths]
        ms_p = _time(lambda: _fwd_bwd(model, padded, device, c), device, reps=3)
        ms_j = _time(lambda: [_fwd_bwd(model, s, device, c) for s in singles], device, reps=3)
        return {"lengths": lengths, "ms_padded": round(ms_p, 2), "ms_jagged": round(ms_j, 2), "speedup": round(ms_p / ms_j, 2)}
    except torch.cuda.OutOfMemoryError:
        return {"lengths": lengths, "ms_padded": None, "ms_jagged": None, "speedup": None, "oom": True}
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
```

---

[← 追踪一个用户](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
