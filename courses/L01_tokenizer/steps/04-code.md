<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一段文本](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 15 步：代码导读

> 阶段 1 / 6：**理解任务**

全部代码在 `tinylm/` 下。读的时候只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① MinHash + LSH

shingle 哈希 [n] → 签名 [64] → 16 段 × 4 行分桶；同桶的候选再用签名估计 Jaccard ≥ 0.7 确认。阈值决定了删句多的副本会不会漏。

[`tinylm/corpus.py#L112-L165`](../../../projects/tinylm/corpus.py#L112-L165)

```python
N_PERM, BANDS, ROWS, SHINGLE, JACCARD = 64, 16, 4, 5, 0.7
_MASK61 = np.uint64((1 << 61) - 1)


def shingle_hashes(text: str, k: int = SHINGLE, max_chars: int = 4000) -> np.ndarray:
    """文档 → 所有 k 字片段（shingle）的 64 位哈希，去重后返回。用 numpy 一次算完，不逐个片段调 Python。"""
    a = np.frombuffer(text[:max_chars].encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)  # [n_chars]
    if len(a) < k:
        a = np.concatenate([a, np.zeros(k - len(a), dtype=np.uint64)])
    n = len(a) - k + 1
    h = np.zeros(n, dtype=np.uint64)
    for j in range(k):  # 多项式滚动哈希，uint64 溢出即取模 2^64
        h = h * np.uint64(1_000_003) + a[j:j + n]
    return np.unique(h)  # [n_shingles]


def signature(h: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """MinHash 签名：64 个随机线性哈希 (a·x + b) mod p 下的最小值。两篇文档签名相同位置的比例 ≈ Jaccard。"""
    x = h & _MASK61                                   # [n]
    v = (x[:, None] * A[None, :] + B[None, :]) & _MASK61  # [n, 64]
    return v.min(axis=0)                               # [64]


def minhash_clusters(docs: list[str], seed: int = 0) -> tuple[list[int], dict]:
    """LSH：签名切成 16 段 × 4 行，任一段完全相同的文档对成为候选，再用签名估计的 Jaccard ≥ 0.7 确认。
    返回每篇文档所属簇的代表（第一次出现的文档下标）。"""
    rs = np.random.default_rng(seed)
    A = rs.integers(1, (1 << 61) - 1, N_PERM, dtype=np.uint64)
    B = rs.integers(0, (1 << 61) - 1, N_PERM, dtype=np.uint64)
    t0 = time.time()
    sigs = np.stack([signature(shingle_hashes(d), A, B) for d in docs])  # [n_docs, 64]
    parent = list(range(len(docs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    candidates = confirmed = 0
    for b in range(BANDS):
        buckets: dict[bytes, int] = {}
        band = sigs[:, b * ROWS:(b + 1) * ROWS]
        for i in range(len(docs)):
            key = band[i].tobytes()
            j = buckets.setdefault(key, i)
            if j != i:
                candidates += 1
                ri, rj = find(i), find(j)
                if ri != rj and float((sigs[i] == sigs[j]).mean()) >= JACCARD:
                    parent[max(ri, rj)] = min(ri, rj)
                    confirmed += 1
    rep = [find(i) for i in range(len(docs))]
    return rep, {"candidate_pairs": candidates, "merged_pairs": confirmed, "seconds": round(time.time() - t0, 1)}
```

### ② 语料管线的顺序

留出集在注入副本之前拿走；去重在切分之前；切分按（来源, 副本号）哈希。顺序错一步，验证集就会泄漏。

[`tinylm/corpus.py#L220-L281`](../../../projects/tinylm/corpus.py#L220-L281)

```python
def build(cfg: dict, log=print) -> dict:
    """返回 {train, valid, test, stats}。所有随机性来自 data_seed，和训练种子无关。"""
    rng = random.Random(10_007 + cfg["data_seed"])
    raw = load_raw(cfg["corpus"])
    cleaned = [clean(t) for t in raw]
    docs = [t for t in cleaned if len(t) >= cfg["min_chars"]]
    stats = {"raw_docs": len(raw), "raw_bytes": nbytes(raw), "short_dropped": len(cleaned) - len(docs),
             "brackets_removed_bytes": nbytes(raw) - nbytes(cleaned)}
    if cfg["source_dedup"] != "none":                          # ⓪ 底稿去重：维基自带的模板化近重复条目
        before = nbytes(docs)
        keep, sd = dedup(docs, cfg["source_dedup"], cfg["data_seed"])
        docs = [docs[i] for i in keep]
        stats["source_dedup"] = {"removed_docs": sd["removed"], "removed_bytes_frac": round(1 - nbytes(docs) / before, 4)}
    rng.shuffle(docs)
    test = take_bytes(docs, cfg["test_mb"])                    # ① 干净留出：注入副本之前拿走
    rest = docs[len(test):]
    tagged, dup_src, inj = inject_duplicates(rest, cfg["dup_ratio"], rng)  # ② 模拟转载：[(来源, 副本号, 文本)]
    stats["injected"] = inj
    # ③ 去重：只看文本。同一簇保留最先出现的一篇，这里让原文排在副本前面（相当于保留最早发布的版本），
    #    这样各种去重方式下，验证集里的原文是同一批
    order = sorted(range(len(tagged)), key=lambda i: tagged[i][1] > 0)
    kept, dd = dedup([tagged[i][2] for i in order], cfg["dedup"], cfg["data_seed"])
    pool = [tagged[i] for i in sorted(order[j] for j in kept)]  # 恢复打乱后的顺序
    stats["dedup"] = dd
    frac = min(1.0, cfg["valid_mb"] * 1e6 / max(nbytes(rest), 1))  # ④ 按 URL 哈希切分：原文约 valid_mb
    is_valid = [in_valid(cfg["data_seed"], s, k, frac) for s, k, _ in pool]
    valid_t = [p for p, v in zip(pool, is_valid) if v]
    train_all = [p for p, v in zip(pool, is_valid) if not v]
    train_t = train_all[:len(take_bytes([t for _, _, t in train_all], cfg["train_mb"]))]
    train, valid = [t for _, _, t in train_t], [t for _, _, t in valid_t]
    # 验证集里有多少文档的同源副本进了训练集：泄漏的直接证据（只用于统计）
    train_src = Counter(s for s, _, _ in train_t)
    leaked = sum(1 for s, _, _ in valid_t if s in train_src)
    train_copies = len(train_t) - len(train_src)
    valid_self_dup = len(valid_t) - len({s for s, _, _ in valid_t})          # 验证集内部互为副本的文档
    valid_boiler = sum(1 for t in valid if any(b in t for b in BOILERPLATE))  # 带转载声明的文档
    stats.update({"valid_self_dup_frac": round(valid_self_dup / max(len(valid_t), 1), 4),
                  "valid_boilerplate_frac": round(valid_boiler / max(len(valid), 1), 4),
                  "after_dedup_docs": len(pool), "train_docs": len(train), "train_bytes": nbytes(train),
                  "train_copy_docs": train_copies, "train_copy_frac": round(train_copies / max(len(train_t), 1), 4),
                  "train_copy_bytes_frac": round(nbytes([t for _, k, t in train_t if k > 0]) / max(nbytes(train), 1), 4),
                  "train_available_bytes": nbytes([t for _, _, t in train_all]),
                  "valid_docs": len(valid), "valid_bytes": nbytes(valid),
                  "valid_copy_docs": sum(1 for _, k, _ in valid_t if k > 0),
                  "test_docs": len(test), "test_bytes": nbytes(test), "valid_with_train_copy": leaked,
                  "valid_leak_frac": round(leaked / max(len(valid), 1), 4),
                  "valid_leaked_train_copies_mean": round(sum(train_src[s] for s, _, _ in valid_t if s in train_src)
                                                          / max(leaked, 1), 2),
                  "train_chars": sum(len(d) for d in train)})
    sd_removed = stats.get("source_dedup", {}).get("removed_docs", 0)
    log(f"语料：原始 {len(raw)} 篇 → 清洗后 {len(docs) + sd_removed} 篇 → 底稿去重后 {len(docs)} 篇；注入副本 {inj['copies']} 份；"
        f"去重（{cfg['dedup']}）去掉 {dd['removed']} 篇；训练 {len(train)} 篇 {stats['train_bytes'] / 1e6:.1f} MB，"
        f"验证 {len(valid)} 篇（其中 {leaked} 篇在训练集有副本），留出 {len(test)} 篇")
    # 每篇验证文档的标记（只用于分组统计，训练看不到）：
    #   leaked = 同源文档进了训练集；train_n = 训练集里同源文档几篇；orig = 原文（不是副本）；
    #   dup_src = 这篇的来源被转载过；boiler = 带转载声明
    return {"train": train, "valid": valid, "test": test, "stats": stats,
            "valid_flags": {"leaked": [s in train_src for s, _, _ in valid_t],
                            "train_n": [train_src.get(s, 0) for s, _, _ in valid_t],
                            "orig": [k == 0 for _, k, _ in valid_t],
                            "dup_src": [s in dup_src for s, _, _ in valid_t],
                            "boiler": [any(b in t for b in BOILERPLATE) for t in valid]}}
```

### ③ 训练字节级 BPE

ByteLevel 预处理把每个字节映射成可见字符；initial_alphabet 保证 256 个字节都在词表里。

[`tinylm/tokenizer.py#L86-L100`](../../../projects/tinylm/tokenizer.py#L86-L100)

```python
def train_bpe(texts: list[str], vocab_size: int) -> HFTokenizer:
    """字节级 BPE：先按 Qwen2 的正则预切分，再把每个片段转成字节，从 256 个字节出发反复合并最高频的相邻对，
    直到词表达到 vocab_size。"""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOS], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(texts, trainer=trainer)
    return HFTokenizer(tk, "bpe")
```

### ④ 小 GPT

[B, T] → wte → [B, T, C] → N 层 → RMSNorm → 与 wte 共享的输出层 → [B, T, V]。词表补齐到 64 的倍数。

[`tinylm/model.py#L201-L243`](../../../projects/tinylm/model.py#L201-L243)

```python
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
```

### ⑤ bits-per-byte

逐块求交叉熵之和；预测分隔符（EOS）的位置不计入，因为它不对应原文字节。

[`tinylm/train.py#L81-L118`](../../../projects/tinylm/train.py#L81-L118)

```python
@torch.no_grad()
def evaluate(model, stream: np.ndarray, n_bytes: int, eos_id: int, cfg: dict, device, max_windows: int | None = None,
             T: int | None = None) -> dict:
    """不重叠的定长窗口过一遍。分隔符 EOS 作为预测目标时不计（它不对应原文字节）。
    返回 loss（每 token 平均 NLL，nats）、bpb（总 NLL / ln2 / 原文字节数）。T 不写时用训练的 seq_len。"""
    model.eval()
    T = T or cfg["seq_len"]
    B = max(1, cfg["batch_size"] * cfg["seq_len"] // T)
    off = cfg.get("target_offset", 1)
    n_win = max(1, (len(stream) - 1 + T - 1) // T)
    if max_windows:
        n_win = min(n_win, max_windows)
    total_nll, total_tok = 0.0, 0
    for w0 in range(0, n_win, B):
        xs, ys = [], []
        for w in range(w0, min(w0 + B, n_win)):
            seg = stream[w * T: w * T + T + 1]
            x = np.full(T, eos_id, dtype=np.int64)
            y = np.full(T, -100, dtype=np.int64)
            n = len(seg) - 1
            x[:n], y[:n] = seg[:-1], seg[off:off + n]
            y[y == eos_id] = -100
            xs.append(x)
            ys.append(y)
        x = torch.from_numpy(np.stack(xs)).to(device)                      # [b, T]
        y = torch.from_numpy(np.stack(ys)).to(device)                      # [b, T]
        with autocast(device, cfg):
            logits = model(x)                                              # [b, T, V]
        nll = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction="sum")
        total_nll += float(nll)
        total_tok += int((y != -100).sum())
    model.train()
    if total_tok == 0:  # 空子集（例如去重后验证集里没有泄漏文档）
        return {"loss": None, "bpb": None, "tokens": 0, "bytes": 0}
    frac = n_win / max(1, (len(stream) - 1 + T - 1) // T)                  # 只评估了一部分窗口时按比例折算字节
    loss = total_nll / max(total_tok, 1)
    return {"loss": round(loss, 5), "bpb": round(total_nll / math.log(2) / max(n_bytes * frac, 1), 5),
            "tokens": total_tok, "bytes": int(n_bytes * frac)}
```

---

[← 追踪一段文本](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
