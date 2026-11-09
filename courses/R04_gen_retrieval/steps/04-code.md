<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一个用户](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 16 步：代码导读

> 阶段 1 / 8：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① encoder-decoder

encoder 每个历史物品一个位置（不是每个 token 一个位置），序列短 4 倍；decoder 只有 4 个位置。选 encoder-decoder 而不是 decoder-only：历史只编码一次，W 条 beam 共享。

[`genrec/model.py#L74-L110`](../../../projects/genrec/model.py#L74-L110)

```python
class TigerEncDec(nn.Module):
    """encoder：每个历史物品一个位置，输入 = 它各位 token 的 embedding 之和 + 位置编码，[B, T, d]。
    decoder：从 BOS 开始自回归生成目标物品的 D 个 token，cross-attend encoder 输出。输出层是全词表 V。
    选 encoder-decoder：历史只编码一次，W 条 beam 共享同一份 memory，不用把历史复制 W 份。"""

    def __init__(self, vocab: int, depth: int, cfg: dict):
        super().__init__()
        d, h = cfg["d_model"], cfg["n_head"]
        self.vocab, self.depth = vocab, depth
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(cfg["max_hist"], d)
        self.dpos = nn.Embedding(depth + 1, d)
        mk = lambda: dict(d_model=d, nhead=h, dim_feedforward=4 * d, dropout=cfg["dropout"], batch_first=True, norm_first=True)  # noqa: E731
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(**mk()), cfg["enc_layers"], enable_nested_tensor=False)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(**mk()), cfg["dec_layers"])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)
        nn.init.normal_(self.tok.weight, std=0.02)

    def encode(self, hist_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hist_tokens [B, T, D]（左补齐，padding 物品全为 0）→ memory [B, T, d]、padding mask [B, T]。"""
        B, T, _ = hist_tokens.shape
        pad = hist_tokens[..., 0] == PAD
        x = self.tok(hist_tokens).sum(2) + self.pos(torch.arange(T, device=hist_tokens.device))[None]
        return self.enc(x, src_key_padding_mask=pad), pad

    def decode(self, prefix: torch.Tensor, mem: torch.Tensor, mem_pad: torch.Tensor) -> torch.Tensor:
        """prefix [B', t]（以 BOS 开头）→ 每个位置下一个 token 的 logits [B', t, V]。"""
        t = prefix.shape[1]
        y = self.tok(prefix) + self.dpos(torch.arange(t, device=prefix.device))[None]
        causal = torch.triu(torch.full((t, t), float("-inf"), device=prefix.device), 1)
        return self.out(self.norm(self.dec(y, mem, tgt_mask=causal, memory_key_padding_mask=mem_pad)))

    def forward(self, hist_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        mem, pad = self.encode(hist_tokens)
        prefix = torch.cat([torch.full_like(target_tokens[:, :1], BOS), target_tokens[:, :-1]], 1)
        return self.decode(prefix, mem, pad)                                # [B, D, V]
```

### ② beam search

score[:, 1:] = −∞ 那一行是关键：只有第 0 条 beam 能扩展，第一步才会选出 W 个不同的 token。cand [B, W·V] 取 top-W，beam = idx // V，token = idx % V。

[`genrec/model.py#L115-L143`](../../../projects/genrec/model.py#L115-L143)

```python
@torch.no_grad()
def beam_search(model: TigerEncDec, hist_tokens: torch.Tensor, trie: TensorTrie | None, W: int, init: str = "one_root"):
    """按 [B, W] 批量的 beam search：每步在 W·V 个候选里取全局 top-W。返回 token 序列 [B, W, D] 和分数 [B, W]。
    init="one_root"：只有第 0 条 beam 从 0 分开始，其余 −∞，第一步才会展开成 W 条不同的前缀。
    init="all_zero"（故障）：W 条 beam 一模一样，全局 top-W 选到的是同一个 token 的 W 份拷贝。"""
    B = hist_tokens.shape[0]
    mem, pad = model.encode(hist_tokens)
    mem = mem.repeat_interleave(W, 0)                                       # [B·W, T, d]：共享同一份历史编码
    pad = pad.repeat_interleave(W, 0)
    dev = hist_tokens.device
    score = torch.zeros(B, W, device=dev)
    if init == "one_root":
        score[:, 1:] = float("-inf")
    seq = torch.full((B, W, 1), BOS, dtype=torch.long, device=dev)
    nodes = torch.zeros(B, W, dtype=torch.long, device=dev)
    V = model.vocab
    for step in range(model.depth):
        logits = model.decode(seq.view(B * W, -1), mem, pad)[:, -1].float()   # [B·W, V]
        logp = F.log_softmax(logits, -1).view(B, W, V)
        if trie is not None:
            logp = logp.masked_fill(~trie.allowed(nodes), float("-inf"))
        cand = (score[..., None] + logp).view(B, W * V)                     # [B, W·V]
        top, idx = cand.topk(W, dim=1)
        beam, tok = idx // V, idx % V
        seq = torch.cat([seq.gather(1, beam[..., None].expand(-1, -1, seq.shape[2])), tok[..., None]], 2)
        if trie is not None:
            nodes = trie.step(nodes.gather(1, beam).clamp(min=0), tok)
        score = top
    return seq[..., 1:], score
```

### ③ GPU trie

next_node [节点数, V]，−1 = 这个前缀下不存在。合法掩码 = next_node[nodes] ≥ 0，状态转移 = next_node[nodes, tokens]。

[`genrec/model.py#L35-L69`](../../../projects/genrec/model.py#L35-L69)

```python
class TensorTrie:
    """前缀树展开成稠密转移表：next_node[node, token] = 子节点（−1 = 这个前缀下不存在）；leaf_bucket[node] = 叶子对应的桶号。
    约束解码每一步只要一次 gather，没有 Python 循环。"""

    def __init__(self, item_tokens: torch.Tensor, vocab: int, device):
        seqs = item_tokens[1:].tolist()
        ids, edges, leaf = {(): 0}, [], {}
        for item, seq in enumerate(seqs, start=1):
            parent = 0
            for d in range(len(seq)):
                key = tuple(seq[:d + 1])
                if key not in ids:
                    ids[key] = len(ids)
                    edges.append((parent, seq[d], ids[key]))
                parent = ids[key]
            leaf.setdefault(parent, []).append(item)
        n = len(ids)
        nxt = torch.full((n, vocab), -1, dtype=torch.int32)
        for p, t, c in edges:
            nxt[p, t] = c
        self.next_node = nxt.to(device)
        self.leaf_items = leaf                                              # 叶子 → 物品列表（没有去重位时一个桶里可能有多个）
        self.nodes = n
        self.depth = len(seqs[0])
        self.mb = round(nxt.numel() * 4 / 1e6, 3)
        lv = torch.zeros(n, dtype=torch.long)
        for key, i in ids.items():
            lv[i] = len(key)
        self.branching = [round(float(((nxt[lv == d] >= 0).sum(1)).float().mean()), 2) for d in range(self.depth)]

    def allowed(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.next_node[nodes] >= 0                                  # [..., V]

    def step(self, nodes: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.next_node[nodes, tokens].long()
```

### ④ 训练

随机取用户和终点，历史 = seq[:end] 的最后 200 个，目标 = seq[end] 的 4 个 token；交叉熵在全词表上算。

[`genrec/train.py#L108-L146`](../../../projects/genrec/train.py#L108-L146)

```python
def train(ctx, cfg, model: TigerEncDec, d, tok, trie, dec, device) -> dict:
    rng = np.random.default_rng(cfg["seed"] + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    item_tok = tok.item_tokens.to(device)
    vu, vc = d.evals["valid"]
    sub = np.random.default_rng(7).permutation(len(vu))[:1000]
    total, warm = cfg["max_steps"], max(1, cfg["max_steps"] // 50)
    best = {"recall10": -1.0, "step": 0, "state": None}
    t0, win = time.time(), []
    for step in range(total):
        lr = cfg["lr"] * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
        for g in opt.param_groups:
            g["lr"] = lr
        users = d.train_users[rng.integers(0, len(d.train_users), size=cfg["batch_size"])]
        ends = np.array([int(rng.integers(1, d.n_train[u])) for u in users])     # 目标 = seq[end]，历史 = seq[:end]
        h = torch.from_numpy(hist_batch(d, users, ends, cfg["max_hist"])).to(device)
        tgt = torch.from_numpy(np.array([d.seq(int(u))[0][e] for u, e in zip(users, ends)], np.int64)).to(device)
        with autocast(device):
            logits = model(item_tok[h], item_tok[tgt])                      # [B, D, V]
            loss = F.cross_entropy(logits.float().flatten(0, 1), item_tok[tgt].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        win.append(loss.item())
        if (step + 1) % 100 == 0:
            ctx.metric("train_loss", float(np.mean(win)), step=step + 1, lr=lr)
            win = []
            ctx.check()
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == total:
            r, _, _ = evaluate(model, d, tok, trie, dec, vu[sub], vc[sub], cfg, device, 10, True, "one_root")
            ctx.metric("valid_recall10", r["recall10"], step=step + 1, split="valid")
            ctx.log(f"step {step + 1}/{total}  loss {loss.item():.4f}  验证 Recall@10（beam 10）{r['recall10']:.4f}")
            if r["recall10"] > best["recall10"]:
                best = {"recall10": r["recall10"], "step": step + 1,
                        "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
    model.load_state_dict(best["state"])
    return {"seconds": round(time.time() - t0, 2), "best_step": best["step"], "best_valid_recall10": best["recall10"],
            "steps": total, "params": sum(p.numel() for p in model.parameters())}
```

### ⑤ 成本基准

1000 万 × d 的 bf16 物品表，256 个用户一批打分取 top-10，和 beam 10 解码的每用户毫秒比。

[`genrec/cost.py#L11-L35`](../../../projects/genrec/cost.py#L11-L35)

```python
@torch.no_grad()
def full_scoring_ms(n_items: int, d: int, device, users: int = 256, reps: int = 3) -> dict:
    """bf16 物品表 [N, d]，一批 users 个用户向量打分取 top-10，每用户毫秒数（中位数）。放不下记 oom。"""
    try:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        table = torch.randn(n_items, d, device=device, dtype=dtype)
        u = torch.randn(users, d, device=device, dtype=dtype)
        ts = []
        for i in range(reps + 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for s in range(0, n_items, 4_000_000):                         # 分块打分，再合并各块的 top-10
                (u @ table[s:s + 4_000_000].T).float().topk(10, dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i:
                ts.append((time.perf_counter() - t0) * 1000 / users)
        return {"ms_per_user": round(sorted(ts)[len(ts) // 2], 4), "table_gb": round(n_items * d * 2 / 1e9, 3), "oom": False}
    except torch.cuda.OutOfMemoryError:
        return {"ms_per_user": None, "table_gb": round(n_items * d * 2 / 1e9, 3), "oom": True}
    finally:
        table = u = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
```

---

[← 追踪一个用户](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
