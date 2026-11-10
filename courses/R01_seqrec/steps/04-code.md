<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一个用户](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 14 步：代码导读

> 阶段 1 / 6：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 截断

recent 取 seq[-L:]，earliest 取 seq[:L]。训练窗口和评估输入都经过这一个函数。

[`recseq/data.py#L37-L43`](../../../projects/recseq/data.py#L37-L43)

```python
def truncate(n: int, L: int, mode: str) -> tuple[int, int]:
    """长度 n 的历史只保留 L 个：返回保留的区间 [a, b)。
      recent    最近的 L 个：seq[-L:]
      earliest  最早的 L 个：seq[:L]（教学故障）"""
    if n <= L:
        return 0, n
    return (n - L, n) if mode == "recent" else (0, L)
```

### ② 训练窗口

随机终点 e，取 truncate(seq[:e], L+1)，x = 窗口[:-1]、y = 窗口[1:]，左补齐：最近的行为总在最右边。

[`recseq/data.py#L56-L67`](../../../projects/recseq/data.py#L56-L67)

```python
def train_batch(d: SeqData, rng: np.random.Generator, B: int, L: int, mode: str) -> dict:
    """B 条训练窗口（左补齐到 L）：x [B, L]，y [B, L]（下一个物品，pad 处为 0），ts [B, L]，ts_next [B, L]。"""
    users = d.train_users[rng.integers(0, len(d.train_users), size=B)]
    xs, ys, t0, t1 = [], [], [], []
    for u in users:
        items, ts = d.seq(u)
        e = int(rng.integers(2, d.n_train[u] + 1))                         # 窗口终点（不含）：用 seq[:e]
        a, b = truncate(e, L + 1, mode)
        w, tw = items[a:b], ts[a:b]
        xs.append(w[:-1]); ys.append(w[1:]); t0.append(tw[:-1]); t1.append(tw[1:])
    return {"x": _left_pad(xs, L, np.int64), "y": _left_pad(ys, L, np.int64),
            "ts": _left_pad(t0, L, np.int64), "ts_next": _left_pad(t1, L, np.int64)}
```

### ③ DIN

s1 = h W Eᵀ 是"历史 j 对候选 c"的注意力打分 [B, L, V]，s2 = h Eᵀ 是被加权的值。没有位置偏置时，位置 t 只看 j ≤ t 的 softmax 可以用前缀和一次算完。

[`recseq/models.py#L92-L114`](../../../projects/recseq/models.py#L92-L114)

```python
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
```

### ④ SASRec

先把左补齐转成右补齐（padding 移到"未来"），这样 SDPA 只用 is_causal=True、不传 mask，就能走 flash 后端；算完再转回来。

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

### ⑤ 评估

名次 = 打分高于目标的物品数。采样评估的负例从用户没交互过的物品里抽，和模型无关，各场景相同。

[`recseq/train.py#L39-L92`](../../../projects/recseq/train.py#L39-L92)

```python
@torch.no_grad()
def scores(model, d: SeqData, users, cuts, cfg: dict, device, L: int) -> tuple[np.ndarray, np.ndarray, list]:
    """每个评估样本：目标在全部物品里的名次（0 = 第 1 名）和最后一个位置的 logits 前 k 个。返回 (rank, targets, kept)。"""
    model.eval()
    eb = max(1, min(512, 65536 // L))
    ranks, tgts, kept = [], [], []
    for s in range(0, len(users), eb):
        b = eval_batch(d, users[s:s + eb], cuts[s:s + eb], L, cfg["truncate"])
        bt = to_device(b, device)
        with autocast(device, cfg):
            logits = model(bt)[:, -1].float()                                # [B, V] 最后一个位置
        logits[:, 0] = float("-inf")
        tgt = bt["target"]
        t_score = logits.gather(1, tgt[:, None])
        ranks.append((logits > t_score).sum(1).cpu().numpy())
        tgts.append(b["target"]); kept += b["kept"]
    model.train()
    return np.concatenate(ranks), np.concatenate(tgts), kept


def recall_ndcg(rank: np.ndarray, k: int = 10) -> dict:
    hit = rank < k
    return {"recall10": round(float(hit.mean()), 5) if len(rank) else None,
            "ndcg10": round(float((hit / np.log2(rank + 2)).mean()), 5) if len(rank) else None, "n": int(len(rank))}


@torch.no_grad()
def sampled_rank(model, d: SeqData, users, cuts, cfg: dict, device, L: int, n_neg: int, seed: int = 12345) -> np.ndarray:
    """目标 + n_neg 个随机负例（从用户没交互过的物品里均匀抽）中的名次。负例和模型无关，每个场景相同。"""
    rng = np.random.default_rng(seed)
    model.eval()
    eb = max(1, min(512, 65536 // L))
    out = []
    for s in range(0, len(users), eb):
        uu, cc = users[s:s + eb], cuts[s:s + eb]
        b = eval_batch(d, uu, cc, L, cfg["truncate"])
        negs = []
        for u in uu:
            seen = set(d.seq(int(u))[0].tolist())
            cand = rng.integers(1, d.n_items, size=n_neg * 3)
            cand = [c for c in cand if c not in seen][:n_neg]
            while len(cand) < n_neg:
                c = int(rng.integers(1, d.n_items))
                if c not in seen:
                    cand.append(c)
            negs.append(cand)
        bt = to_device(b, device)
        with autocast(device, cfg):
            logits = model(bt)[:, -1].float()
        neg = torch.tensor(np.array(negs), device=device)
        t_score = logits.gather(1, bt["target"][:, None])
        out.append((logits.gather(1, neg) > t_score).sum(1).cpu().numpy())
    model.train()
    return np.concatenate(out)
```

### ⑥ 成本估算

每层每 token：SASRec 线性 12d²、注意力 L·d；DIN 推理对每个候选重新做一次注意力。

[`recseq/cost.py#L19-L41`](../../../projects/recseq/cost.py#L19-L41)

```python
def estimate(cfg: dict, V: int, L: int) -> dict:
    """每 token 前向乘加（MAC）：线性层 SASRec 12d²（注意力投影 4d² + MLP 8d²）、HSTU 5d²（f1 4d² + f2 d²），每层；
    注意力 QKᵀ 和 AV 在因果下平均每个 token 看 L/2 个 key，合计 L·d，每层。注意力占比 = L / (L + 12d) 或 L / (L + 5d)。
    推理（只算最后一个位置的打分，对全部 V 个候选）：
      DIN    ≈ 2·L·V·d（每个候选都要对 L 个历史打分、加权）
      SASRec ≈ n_layer·(12d²·L + L²·d) + V·d（整段过一遍编码器，打分只是一次 [d] × [d, V]）"""
    d, nl, m = cfg["d_model"], cfg["n_layer"], cfg["model"]
    lin = {"sasrec": 12, "hstu": 5}.get(m, 0) * d * d
    attn = L * d if m in ("sasrec", "hstu") else 0
    if m == "din":
        infer = 2 * L * V * d
    elif m == "mean":
        infer = L * d + d * d + V * d
    else:
        infer = nl * (lin * L + L * L * d) + V * d
    return {"linear_per_token": nl * lin, "attn_per_token": nl * attn,
            "attn_share": round(attn / (attn + lin), 4) if attn + lin else 0.0, "infer_macs_per_user": int(infer)}


def est_attn_gb(cfg: dict, L: int, B: int = 1) -> float | None:
    """手写注意力反向保存的 [B, H, L, L] 张量：n_layer · B · H · L² · 每元素字节。"""
    beta = BYTES_PER_ELEM.get(cfg["model"])
    return round(cfg["n_layer"] * B * cfg["n_head"] * L * L * beta / 1e9, 3) if beta else None
```

---

[← 追踪一个用户](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
