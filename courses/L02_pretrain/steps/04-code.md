<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一个 batch](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 16 步：代码导读

> 阶段 1 / 8：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 切块与右移

blk [B, T + 1]；x = blk[:, :T]，y = blk[:, 1:T + 1]。这一行写错（y 不右移），训练和评估会一起错，而且看起来非常好。

[`tinylm/train.py#L72-L76`](../../../projects/tinylm/train.py#L72-L76)

```python
def make_batch(stream: np.ndarray, starts, T: int, offset: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """从 token 流里切 [B, T + 1] 的块：输入 x = 前 T 个，目标 y = 右移 offset 位的 T 个。
    offset = 1 是 next-token；offset = 0（教学故障）时 y 就是 x 本身，模型只要"抄"当前位置。"""
    blk = np.stack([stream[s:s + T + 1] for s in starts]).astype(np.int64)    # [B, T + 1]
    return blk[:, :T], blk[:, offset:offset + T]                               # [B, T]、[B, T]
```

### ② 学习率调度

warmup 线性升到峰值，再按余弦降到峰值的 10%。total 是这个运行自己的总步数：token 预算小，余弦就压缩得短。

[`tinylm/train.py#L32-L40`](../../../projects/tinylm/train.py#L32-L40)

```python
def lr_at(step: int, total: int, cfg: dict) -> float:
    """线性 warmup，然后余弦降到峰值的 10%（cosine）或保持峰值不变（constant）。"""
    warm = max(1, int(total * cfg["warmup_frac"]))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    if cfg.get("lr_schedule", "cosine") == "constant":
        return cfg["lr"]
    p = (step - warm) / max(1, total - warm)
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))
```

### ③ 每 token FLOPs

flops_per_token：前向 = 2 × 主体参数 + 2 × V × d（共享的输出层）+ 注意力；训练 = 3 × 前向。6ND 口径少了注意力那一项。

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

### ④ 前缀、遍数与预算

order = 每遍各自打乱的块编号拼接；budget_steps 给定时步数取 min(预算, 数据能提供的)。

[`tinylm/train.py#L142-L158`](../../../projects/tinylm/train.py#L142-L158)

```python
def plan_steps(n_blocks: int, cfg: dict, budget_steps: int | None = None) -> dict:
    """训练多少步、用哪些块：只用前 data_frac 的块（训练流的前缀），过 epochs 遍，每遍重新打乱；
    给了算力预算时步数 = 预算 ÷ 每步 FLOPs（不超过数据能提供的步数）。"""
    B = cfg["batch_size"]
    n_used = max(1, int(n_blocks * cfg.get("data_frac", 1.0)))
    epochs = int(cfg.get("epochs", 1))
    order = np.concatenate([np.random.default_rng(cfg["seed"] + 1000 * e).permutation(n_used) for e in range(epochs)])
    total = len(order) // B
    capped = False
    if budget_steps:
        capped = budget_steps > total
        total = min(total, budget_steps)
    if cfg["max_steps"]:
        total = min(total, cfg["max_steps"])
    if total < 1:
        raise ValueError(f"训练集只有 {n_used} 个块，不够一步（batch_size={B}）")
    return {"order": order, "total": total, "n_used": n_used, "epochs": epochs, "budget_capped": capped}
```

### ⑤ 等算力换算

预算 = 比例 × 基线模型把训练流过一遍的 FLOPs；小模型每 token 便宜，同样的预算能多走几步。

[`tinylm/experiment.py#L43-L52`](../../../projects/tinylm/experiment.py#L43-L52)

```python
def budget_steps(cfg: dict, vocab_size: int, n_tokens: int, fpt: float) -> int | None:
    """等算力对照：预算 = flops_budget_frac × "默认结构（d=384、6 层）把整个训练流过一遍"的训练 FLOPs；
    步数 = 预算 ÷ (本模型每 token FLOPs × 每步 token)。模型越小，同样算力能看越多 token。"""
    if not cfg.get("flops_budget_frac"):
        return None
    ref_cfg = {**cfg, **{k: PARAMS.by_name[k].default for k in MODEL_KEYS}}
    with torch.device("meta"):
        ref = GPT(vocab_size, ref_cfg)
    budget = cfg["flops_budget_frac"] * flops_per_token(ref, ref_cfg)["train_flops_per_token"] * n_tokens
    return max(1, round(budget / (fpt * cfg["batch_size"] * cfg["seq_len"])))
```

---

[← 追踪一个 batch](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
