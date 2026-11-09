<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一条样本的四种口径](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 19 步：代码导读

> 阶段 1 / 7：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 四种口径

point_in_time 用 (key, 时间) 排序后的组前缀和：同一 key、同一秒的行并成一组，彼此看不见。include_self 在它上面加 1 和本行标签；daily 把时间换成天；full 直接 bincount 全部行。

[`recbase/features.py#L40-L58`](../../../projects/recbase/features.py#L40-L58)

```python
def stats(key: np.ndarray, ts: np.ndarray, label: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """每行的 (此前次数 cnt, 此前正例数 pos)。
      point_in_time  时间严格早于本行（线上请求时刻能拿到的；同一秒的行互相看不见）
      daily          严格早于本行所在的那一天（T+1 天级任务产出的特征）
      include_self   point_in_time + 本行自己的标签（特征里含答案）
      full           该 key 的全部行，含本行和未来（离线一次性聚合再拼回样本）"""
    if mode in ("point_in_time", "include_self"):
        cnt, pos, _ = _grouped(key, ts, label)
        if mode == "include_self":
            cnt, pos = cnt + 1, pos + label.astype(np.int64)
    elif mode == "daily":
        cnt, pos, _ = _grouped(key, ts // 86400, label)
    elif mode == "full":
        tot_n = np.bincount(key)
        tot_p = np.bincount(key, weights=label).astype(np.int64)
        cnt, pos = tot_n[key], tot_p[key]
    else:
        raise ValueError(mode)
    return cnt, pos
```

### ② 训练口径与线上口径

np.where(offline, 场景口径, 线上口径)：留出行永远退回线上口径。*_serve 是全部行都用线上口径的版本，valid_serving 用它给同一批验证行再打一次分。future_share = 统计次数 > 截至本行时刻的行数 的占比。

[`recbase/experiment.py#L135-L162`](../../../projects/recbase/experiment.py#L135-L162)

```python
def build_feats(cfg: dict, rows: dict, d: Path) -> None:
    """训练 / 验证行用 stats_mode 口径；留出行永远用线上口径（serve_mode）。另存一份全部行都用线上口径的版本：
    同一批验证行"换成线上能拿到的特征"再打一次分，就是 valid_serving。"""
    s, ts, y = rows["s"], rows["ts"], rows["label"]
    mode, serve = cfg["stats_mode"], F.serve_mode(cfg["stats_mode"])
    prior = float(y[s == 0].mean())
    alpha = cfg["stats_smooth"]
    out, check = {}, {"mode": mode, "serve_mode": serve, "prior": round(prior, 4), "single_auc": {}}
    offline = s != 2
    for name, key in (("user", rows["user"]), ("item", rows["movie"])):
        pit = F.stats(key, ts, y, "point_in_time")
        cm, pm = F.stats(key, ts, y, mode) if mode != "point_in_time" else pit
        cs, ps = F.stats(key, ts, y, serve) if serve != "point_in_time" else pit
        cnt, pos = np.where(offline, cm, cs), np.where(offline, pm, ps)     # 留出行退回线上口径
        out[f"{name}_cnt_bucket"], out[f"{name}_rate"], out[f"{name}_lcnt"] = \
            F.cnt_bucket(cnt), F.smooth_rate(cnt, pos, prior, alpha), F.log_cnt(cnt)
        out[f"{name}_cnt_bucket_serve"], out[f"{name}_rate_serve"], out[f"{name}_lcnt_serve"] = \
            F.cnt_bucket(cs), F.smooth_rate(cs, ps, prior, alpha), F.log_cnt(cs)
        out[f"{name}_prev"] = pit[0]                                        # 分组用：样本时刻之前的评分数
        upto = F.rows_upto_now(key, ts)
        check[f"{name}_rows_using_future_share"] = round(float((cm[offline] > upto[offline]).mean()), 4)
        check[f"{name}_rows_using_self_share"] = 1.0 if mode in ("include_self", "full") else 0.0
        rate, rate_s = out[f"{name}_rate"], out[f"{name}_rate_serve"]
        check["single_auc"][f"{name}_rate"] = {
            "train": round(M.auc(y[s == 0], rate[s == 0]), 4), "valid": round(M.auc(y[s == 1], rate[s == 1]), 4),
            "valid_serving": round(M.auc(y[s == 1], rate_s[s == 1]), 4)}
    np.savez(d / "feats.npz", **out)
    (d / "check.json").write_text(json.dumps(check, ensure_ascii=False))
```

### ③ 分组指标

分组键用时点正确的"此前评分数"，不随口径变，各场景的分组是同一批行。

[`recbase/metrics.py#L114-L137`](../../../projects/recbase/metrics.py#L114-L137)

```python
def group_masks(uid: np.ndarray, u_prev: np.ndarray, i_prev: np.ndarray, fut: np.ndarray) -> dict[str, np.ndarray]:
    """uid = 训练集词表编码（0 = 没见过）；u_prev / i_prev = 样本时刻之前的评分数（时点正确，不随统计口径变）；
    fut = 训练集里有没有这个用户更晚的行为。"""
    return {"u_unseen": uid == 0, "u_seen": uid != 0,
            "u_new": u_prev == 0, "u_low": (u_prev >= 1) & (u_prev < 20), "u_mid": (u_prev >= 20) & (u_prev < 100),
            "u_high": (u_prev >= 100) & (u_prev < 400), "u_heavy": u_prev >= 400,
            "i_tail": i_prev < 50, "i_mid": (i_prev >= 50) & (i_prev < 500), "i_head": i_prev >= 500,
            "fut_user": fut, "no_fut_user": ~fut}


def groups(label: np.ndarray, prob: np.ndarray, user: np.ndarray, masks: dict[str, np.ndarray]) -> tuple[dict, list]:
    """整体 AUC 的变化可能只来自某一类用户。每组都报行数：行数少的组别下结论。返回 dict（模板 / 断言用）和列表（网页表格用）。"""
    by_key, rows = {}, []
    for key, dim, val in GROUPS:
        m = masks[key]
        n = int(m.sum())
        if n:
            g, _ = gauc(label[m], prob[m], user[m])
            d = {"n": n, "label_rate": _r(label[m].mean(), 4), "auc": _r(auc(label[m], prob[m])), "gauc": _r(g)}
        else:
            d = {"n": 0, "label_rate": None, "auc": None, "gauc": None}
        by_key[key] = d
        rows.append({"key": key, "group": dim, "value": val, **d})
    return by_key, rows
```

### ④ 特征组与 DCN-v2

fields_for 按特征组决定查哪些表；DCN-v2 的 cross 层 x_{l+1} = x0 ⊙ (W x_l + b) + x_l，与 MLP 并联后拼接。

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

---

[← 追踪一条样本的四种口径](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
