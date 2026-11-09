<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一条样本](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 13 步：代码导读

> 阶段 1 / 5：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 三种切分共用一个未来留出集

先按全局时间分位点取 t80、t90，test = ts ≥ t90 对所有模式都一样；random 用行号上的随机数取 1/9；user_last 在 pool 内按 (用户, 时间) 排序，取每个用户最后 k 条。

[`recbase/data.py#L78-L103`](../../../projects/recbase/data.py#L78-L103)

```python
def split(ts: np.ndarray, user: np.ndarray, mode: str, data_seed: int = 0, k: int = 5) -> tuple[np.ndarray, dict]:
    """每行的切分编号：0 = train，1 = valid，2 = test。行已按时间排序。
    test 固定为 ts ≥ t90；mode 只决定 t90 之前的 pool 怎么分：
      time      train < t80 ≤ valid < t90（训练集里没有"未来"）
      random    pool 内按行号哈希随机 1/9 进 valid（同一用户之后的行为可能在训练集里）
      user_last pool 内每个用户最后 k 条进 valid（行数不到 2k 的用户全留在 train）"""
    t80, t90 = (int(x) for x in np.quantile(ts, [0.8, 0.9]))
    s = np.where(ts >= t90, 2, 0).astype(np.int8)
    pool = s == 0
    if mode == "time":
        s[pool & (ts >= t80)] = 1
    elif mode == "random":
        rng = np.random.default_rng(1_000_003 + data_seed)
        s[pool & (rng.random(len(ts)) < 1 / 9)] = 1
    elif mode == "user_last":
        idx = np.flatnonzero(pool)
        o = idx[np.lexsort((np.arange(len(idx)), user[idx]))]           # pool 内按 (用户, 时间) 排
        u = user[o]
        start = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
        size = np.diff(np.r_[start, len(o)])
        rank_from_end = np.repeat(start + size, size) - np.arange(len(o))  # 1 = 该用户最后一条
        ok = (rank_from_end <= k) & (np.repeat(size, size) >= 2 * k)
        s[o[ok]] = 1
    else:
        raise ValueError(mode)
    return s, {"t80": t80, "t90": t90}
```

### ② 训练集词表

np.unique 只作用在训练行上；searchsorted 查不到的就是 0。词表里多一个用户，embedding 表就多一行 d 个参数。

[`recbase/data.py#L120-L130`](../../../projects/recbase/data.py#L120-L130)

```python
def vocab(values: np.ndarray) -> tuple[np.ndarray, int]:
    """只用训练集建词表：返回已排序的原始 ID（下标 + 1 就是编码），0 留给训练集没见过的（OOV，冷启动）。"""
    u = np.unique(values)
    return u, len(u) + 1


def encode(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    if len(known) == 0:
        return np.zeros(len(values), dtype=np.int64)
    pos = np.minimum(np.searchsorted(known, values), len(known) - 1)
    return np.where(known[pos] == values, pos + 1, 0).astype(np.int64)
```

### ③ 切分体检

train_has_future：训练集最晚时间是否晚于验证集最早时间；valid_rows_with_later_train_same_user：验证行里，有多少行的用户在训练集里还有更晚的评分；冷启动行占比。

[`recbase/data.py#L135-L157`](../../../projects/recbase/data.py#L135-L157)

```python
def _later_same_key(key: np.ndarray, ts: np.ndarray, s: np.ndarray, target: int) -> np.ndarray:
    """target 集合里的每一行：训练集里同一个 key 有没有时间更晚的行。"""
    last = np.full(int(key.max()) + 1, -1, dtype=np.int64)                # 每个 key 在训练集里的最晚时间
    np.maximum.at(last, key[s == 0], ts[s == 0])
    rows = s == target
    return last[key[rows]] > ts[rows]


def leak_checks(r: dict, s: np.ndarray, uid: np.ndarray, mid: np.ndarray) -> dict:
    """切分的体检：训练集里有没有未来；验证集里多少行的用户在训练集还有更晚的行为；冷启动占比。"""
    ts = r["ts"]
    v, t = s == 1, s == 2
    later_u = _later_same_key(r["user"], ts, s, 1)
    later_i = _later_same_key(r["movie"], ts, s, 1)
    cold = {}
    for name, m in (("valid", v), ("test", t)):
        cold[name] = {"user_rows": round(float((uid[m] == 0).mean()), 4) if m.any() else None,
                      "item_rows": round(float((mid[m] == 0).mean()), 4) if m.any() else None,
                      "users": int(len(np.unique(r["user"][m])))}
    return {"leak": {"train_has_future": bool(v.any() and ts[s == 0].max() > ts[v].min()),
                     "valid_rows_with_later_train_same_user": round(float(later_u.mean()), 4) if len(later_u) else 0.0,
                     "valid_rows_with_later_train_same_item": round(float(later_i.mean()), 4) if len(later_i) else 0.0},
            "cold": cold, "later_user_mask_valid": later_u}
```

### ④ Embedding + MLP

field_vectors 输出 [B, F, d]，flatten 成 [B, F·d] 再接 MLP。dcn / deepfm 是 R0b 用的变体，这一课只用 mlp。

[`recbase/model.py#L36-L96`](../../../projects/recbase/model.py#L36-L96)

```python
class RecModel(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], cfg: dict):
        super().__init__()
        d, self.arch = cfg["embedding_dim"], cfg["arch"]
        self.sparse, self.dense, self.use_genres = fields_for(cfg["feature_groups"])
        # 每个离散字段一张表 [V, d]。下标 0 = 训练集没见过（OOV / 冷启动），也有自己的向量
        self.emb = nn.ModuleDict({f: nn.Embedding(vocab_sizes[f], d) for f in self.sparse})
        if self.use_genres:  # 一部电影最多 6 个题材：查 6 个向量，padding（0）不参与平均
            self.genre_emb = nn.Embedding(vocab_sizes["genre"], d, padding_idx=0)
        self.n_fields = len(self.sparse) + int(self.use_genres)
        self.in_dim = d * self.n_fields + len(self.dense)
        layers, h = [], self.in_dim
        for w in parse_hidden(cfg["hidden"]):
            layers += [nn.Linear(h, w), nn.ReLU(), nn.Dropout(cfg["dropout"])]
            h = w
        self.mlp = nn.Sequential(*layers)
        if self.arch == "dcn":  # DCN-v2：x_{l+1} = x0 ⊙ (W x_l + b) + x_l，和 MLP 并联，最后拼起来过输出层
            self.cross = nn.ModuleList(nn.Linear(self.in_dim, self.in_dim) for _ in range(CROSS_LAYERS))
            h += self.in_dim
        if self.arch == "deepfm":  # FM 一阶项：每个字段每个取值一个标量
            self.first = nn.ModuleDict({f: nn.Embedding(vocab_sizes[f], 1) for f in self.sparse})
            for e in self.first.values():
                nn.init.zeros_(e.weight)
        self.head = nn.Linear(h, 1)
        for e in self.emb.values():
            nn.init.normal_(e.weight, std=0.05)
        if self.use_genres:
            nn.init.normal_(self.genre_emb.weight, std=0.05)
            with torch.no_grad():
                self.genre_emb.weight[0].zero_()

    def field_vectors(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.emb[f](batch[f]) for f in self.sparse]                # 每个 [B, d]
        if self.use_genres:
            g = batch["genres"]                                             # [B, 6]，0 = padding
            v = self.genre_emb(g)                                           # [B, 6, d]
            k = (g > 0).sum(-1, keepdim=True).clamp(min=1)                  # [B, 1] 实际题材数
            parts.append(v.sum(1) / k)                                      # [B, d] 平均
        return torch.stack(parts, 1)                                        # [B, F, d]

    def concat(self, v: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = v.flatten(1)                                                    # [B, F·d]
        if self.dense:
            x = torch.cat([x, torch.stack([batch[f] for f in self.dense], -1)], -1)  # [B, in_dim]
        return x

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        v = self.field_vectors(batch)                                       # [B, F, d]
        x = self.concat(v, batch)                                           # [B, in_dim]
        h = self.mlp(x)                                                     # [B, h_last]
        if self.arch == "dcn":
            c = x
            for lin in self.cross:
                c = x * lin(c) + c                                          # [B, in_dim]
            h = torch.cat([h, c], -1)
        logit = self.head(h).squeeze(-1)                                    # [B]
        if self.arch == "deepfm":
            s = v.sum(1)                                                    # [B, d]
            logit = logit + 0.5 * (s * s - (v * v).sum(1)).sum(-1)          # FM 二阶：两两内积之和
            logit = logit + sum(self.first[f](batch[f]).squeeze(-1) for f in self.sparse)
        return logit                                                        # [B]，未过 sigmoid
```

### ⑤ AUC 与 GAUC

AUC：按分数排序、并列分组，每个正例贡献"排在它前面的负例数 + 并列负例数 / 2"。GAUC：先按 (用户, 分数) 排序求用户内的秩，只算正负都有的用户，按行数加权。

[`recbase/metrics.py#L22-L72`](../../../projects/recbase/metrics.py#L22-L72)

```python
def _sorted_ties(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按分数升序排好的下标 o 和每个位置的并列组编号 g（分数相同 = 同一组）。"""
    o = np.argsort(score, kind="mergesort")
    s = score[o]
    return o, np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1


def _auc_presorted(o: np.ndarray, g: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """AUC = Σ_正例 w ×（排在它前面的负例权重 + 并列负例权重 / 2）÷（正例总权重 × 负例总权重）。"""
    yy, ww = y[o], w[o]
    pos_g = np.bincount(g, weights=ww * yy)
    neg_g = np.bincount(g, weights=ww * (1 - yy))
    P, N = pos_g.sum(), neg_g.sum()
    if P == 0 or N == 0:
        return float("nan")
    neg_before = np.cumsum(neg_g) - neg_g
    return float((pos_g * (neg_before + 0.5 * neg_g)).sum() / (P * N))


def auc(label: np.ndarray, score: np.ndarray, w: np.ndarray | None = None) -> float:
    """随机抽一个正例和一个负例，正例分数更高的概率（并列算一半）。只有一类样本时无定义，返回 nan。"""
    y = np.asarray(label, np.float64)
    o, g = _sorted_ties(np.asarray(score))
    return _auc_presorted(o, g, y, np.ones(len(y)) if w is None else np.asarray(w, np.float64))


def gauc(label: np.ndarray, score: np.ndarray, user: np.ndarray) -> tuple[float, float]:
    """按用户分别算 AUC，再按用户的行数加权平均。只有正例或只有负例的用户没有定义，排除在外。
    返回 (gauc, 覆盖率 = 参与计算的行数 / 总行数)。"""
    n = len(label)
    if n == 0:
        return float("nan"), 0.0
    o = np.lexsort((score, user))                                        # 先按用户、再按分数
    u, s, y = user[o], score[o], np.asarray(label, np.float64)[o]
    new_u = np.r_[True, u[1:] != u[:-1]]
    u_start = np.maximum.accumulate(np.where(new_u, np.arange(n), 0))   # 每行所在用户的起始位置
    new_t = new_u | np.r_[True, s[1:] != s[:-1]]
    t_start = np.flatnonzero(new_t)
    t_size = np.diff(np.r_[t_start, n])
    tg = np.cumsum(new_t) - 1
    rank = (t_start[tg] - u_start) + (t_size[tg] + 1) / 2               # 用户内的秩（并列取平均），从 1 开始
    uid = np.cumsum(new_u) - 1
    n_all = np.bincount(uid)
    n_pos = np.bincount(uid, weights=y)
    n_neg = n_all - n_pos
    r_pos = np.bincount(uid, weights=rank * y)
    ok = (n_pos > 0) & (n_neg > 0)
    if not ok.any():
        return float("nan"), 0.0
    per_user = (r_pos[ok] - n_pos[ok] * (n_pos[ok] + 1) / 2) / (n_pos[ok] * n_neg[ok])
    return float((n_all[ok] * per_user).sum() / n_all[ok].sum()), float(n_all[ok].sum() / n)
```

### ⑥ 训练循环

全部张量一次放上 GPU，按打乱后的下标取 batch。只有本 batch 出现过的用户行有梯度。

[`recbase/train.py#L96-L139`](../../../projects/recbase/train.py#L96-L139)

```python
def train(ctx, cfg: dict, model: RecModel, X: dict[str, torch.Tensor], train_idx: np.ndarray,
          on_epoch: Callable[[int], dict]) -> dict:
    """训 cfg["epochs"] 遍；每遍重新打乱。每遍结束调用 on_epoch(第几遍) 评估，返回值记进 epochs 列表。"""
    opt = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(cfg["seed"])
    bs, dev = cfg["batch_size"], next(iter(X.values())).device
    steps_per_epoch = (len(train_idx) + bs - 1) // bs
    ctx.info("train", {"train_samples": int(len(train_idx)), "steps_per_epoch": steps_per_epoch,
                       "planned_steps": steps_per_epoch * cfg["epochs"]})
    step, epochs, t0 = 0, [], time.time()
    train_seconds = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        te = time.time()
        perm = train_idx[torch.randperm(len(train_idx), generator=gen).numpy()]
        window, ep_sum, ep_n, t_last = [], 0.0, 0, time.time()
        for s in range(0, len(perm), bs):
            idx = torch.as_tensor(perm[s: s + bs], device=dev)
            batch = {k: v[idx] for k, v in X.items()}                       # 每个字段 [B] 或 [B, 6]
            if step == 0:
                loss_val = trace_first_update(ctx, model, opt, batch, cfg)
            else:
                loss = bce_loss(model(batch), batch["label"])
                opt.zero_grad(set_to_none=True)
                loss.backward()                                             # 只有本 batch 出现的 embedding 行有梯度
                opt.step()
                loss_val = loss.item()
            step += 1
            window.append(loss_val)
            ep_sum += loss_val * len(idx)
            ep_n += len(idx)
            if step % LOG_EVERY == 0:
                dt = time.time() - t_last
                ctx.metric("train_loss", float(np.mean(window)), step=step, epoch=epoch,
                           samples_per_s=round(len(window) * bs / max(dt, 1e-9)))
                window, t_last = [], time.time()
                ctx.check()
        train_seconds += time.time() - te
        ev = {"epoch": epoch, "step": step, "train_logloss": round(ep_sum / max(ep_n, 1), 5), **on_epoch(epoch)}
        epochs.append(ev)
        ctx.log(f"epoch {epoch}: train logloss {ev['train_logloss']}  " +
                "  ".join(f"{k} {v}" for k, v in ev.items() if k not in ("epoch", "step", "train_logloss")))
    return {"epochs": cfg["epochs"], "steps": step, "seconds": round(time.time() - t0, 2),
            "train_seconds": round(train_seconds, 2),
            "samples_per_s": round(len(train_idx) * cfg["epochs"] / max(train_seconds, 1e-9)), "per_epoch": epochs}
```

---

[← 追踪一条样本](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
