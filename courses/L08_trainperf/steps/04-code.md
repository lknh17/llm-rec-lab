<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 19 步：代码导读

> 阶段 1 / 8：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 多卡启动

rank0 留在 labkit worker 进程里（持有 ctx、写 summary），其余 7 个 rank 用 spawn 起子进程，继承同一份代码快照。

[`trainperf/train.py#L264-L294`](../../../projects/trainperf/train.py#L264-L294)

```python
def _child(rank, world, port, cfg):
    run_rank(rank, world, port, cfg, None)


def launch(cfg: dict, ctx) -> dict:
    """rank0 留在当前进程（持有 ctx、写结果），rank 1..N−1 用 spawn 起子进程。子进程失败时 rank0 会在集合通信处超时（3 分钟）。"""
    world = cfg["world_size"] if cfg["strategy"] != "none" else 1
    if world == 1:
        return run_rank(0, 1, 0, cfg, ctx)
    import multiprocessing as mp

    port = free_port()
    mpc = mp.get_context("spawn")
    procs = [mpc.Process(target=_child, args=(r, world, port, cfg), daemon=False) for r in range(1, world)]
    for p in procs:
        p.start()
    try:
        out = run_rank(0, world, port, cfg, ctx)
        for p in procs:
            p.join(120)
        bad = [p.exitcode for p in procs if p.exitcode not in (0, None)]
        if bad:
            ctx.warn("rank_exit", f"有 rank 异常退出：exitcode {bad}")
        return out
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(30)
            if p.is_alive():
                p.kill()
```

### ② DDP / FSDP2

FSDP2 先对每个层 fully_shard，再对整个模型；reshard_after_forward=True 是 FULL_SHARD（前向后立刻释放整份参数），False 近似 ZeRO-2。

[`trainperf/train.py#L38-L59`](../../../projects/trainperf/train.py#L38-L59)

```python
def wrap(model: LM, cfg: dict, device, world: int):
    """none：原样；ddp：DistributedDataParallel（gradient_as_bucket_view：梯度直接放在通信桶里，省一份 4Ψ）；
    fsdp_full / fsdp_grad_op：FSDP2 fully_shard 每个层再包整个模型（reshard_after_forward 分别为 True / False）。"""
    if cfg["ckpt"] == "full":
        model.ckpt = True
    if cfg["compile"]:
        for i, b in enumerate(model.blocks):
            model.blocks[i] = torch.compile(b)
    s = cfg["strategy"]
    if s == "none" or world == 1:
        return model
    if s == "ddp":
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                                         gradient_as_bucket_view=True)
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if cfg["precision"] == "bf16" else MixedPrecisionPolicy()
    reshard = s == "fsdp_full"
    for b in model.blocks:
        fully_shard(b, mp_policy=mp, reshard_after_forward=reshard)
    fully_shard(model, mp_policy=mp, reshard_after_forward=reshard)
    return model
```

### ③ 逐阶段计时

只有 timed=True 时才在每段前后同步，否则同步本身就会拖慢训练；吞吐用不同步的那 15 步算。

[`trainperf/train.py#L142-L187`](../../../projects/trainperf/train.py#L142-L187)

```python
        def one_step(step: int, timed: bool) -> dict:
            """一个优化器步 = gas 个 micro step + optimizer.step()。timed 时每段前后同步，拿到真实耗时。"""
            t = {"data": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0}
            losses = []
            calls_before_last = 0
            for micro in range(gas):
                st["stage"] = "forward"
                t0 = time.perf_counter()
                batch = get_batch()                                          # [mbs, T + 1]
                if timed:
                    sync(device)
                t1 = time.perf_counter()
                skip = ddp and cfg["no_sync"] and micro < gas - 1
                with (model.no_sync() if skip else contextlib.nullcontext()):
                    with autocast(cfg, device):
                        logits = model(batch[:, :-1])                       # [mbs, T, V]
                        loss = F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten()) / gas
                    if timed:
                        sync(device)
                    t2 = time.perf_counter()
                    st["stage"] = "backward"
                    if micro == gas - 1:
                        calls_before_last = calls[0]
                    loss.backward()
                    if timed:
                        sync(device)
                    t3 = time.perf_counter()
                t["data"] += (t1 - t0) * 1000
                t["fwd"] += (t2 - t1) * 1000
                t["bwd"] += (t3 - t2) * 1000
                losses.append(loss.detach())
                if step == 0 and micro == 0 and "after_backward" not in snaps:
                    del logits
                    snaps["after_backward"] = M.snapshot(device)
            st["stage"] = "optimizer"
            t4 = time.perf_counter()
            opt.step()
            opt.zero_grad(set_to_none=False if step == 0 else True)
            if timed:
                sync(device)
            t["opt"] = (time.perf_counter() - t4) * 1000
            if step == 0:
                snaps["after_step"] = M.snapshot(device)
            st["bucket_calls"] = max(calls[0] - calls_before_last, 1)          # 一次同步调用几次桶 all-reduce（DDP 第一步后会重排桶）
            t["loss"] = torch.stack(losses).sum()
            return t
```

### ④ 显存构成

估算：16 字节/参数 ÷（FSDP 时）N；实测：三个快照的差值，峰值减去它们是激活与其他。

[`trainperf/measure.py#L13-L39`](../../../projects/trainperf/measure.py#L13-L39)

```python
def estimate(params: int, strategy: str, world: int, precision: str) -> dict:
    """静态显存（每卡，GB）。PyTorch 混合精度的实际口径：fp32 主参数 4 + fp32 梯度 4 + AdamW m、v 8 = 16 字节/参数。
    （讲义里常见的 2 + 2 + 12 口径是 bf16 参数 / 梯度 + fp32 主参数和 m、v，合计同样 16。）
    DDP：每卡一整份；FSDP（ZeRO-3）：三项都 ÷ N；ZeRO-2（fsdp_grad_op）存储同样 ÷ N，但前向后保留整份 bf16 参数。"""
    n = world if strategy.startswith("fsdp") else 1
    p, g, o = params * 4 / n, params * 4 / n, params * 8 / n
    out = {"params_gb": p / GB, "grads_gb": g / GB, "optimizer_gb": o / GB}
    if strategy == "fsdp_grad_op":
        out["unsharded_params_gb"] = params * (2 if precision == "bf16" else 4) / GB
    out["static_gb"] = sum(v for k, v in out.items() if k != "unsharded_params_gb")
    return {k: round(v, 3) for k, v in out.items()}


def snapshot(device) -> float:
    return torch.cuda.memory_allocated(device) / GB if device.type == "cuda" else 0.0


def breakdown(snaps: dict, peak_gb: float | None) -> dict:
    """实测快照：建模后 = 参数；首个反向后 − 建模后 = 梯度；首次 step 后 − 首个反向后 = 优化器状态；峰值 − 三者 = 激活与其他
    （logits、通信缓冲、FSDP 的整份参数、分配器碎片都在这一项）。"""
    p = snaps["after_build"]
    g = max(snaps["after_backward"] - p, 0.0)
    o = max(snaps["after_step"] - snaps["after_backward"], 0.0)
    out = {"params_gb": round(p, 3), "grads_gb": round(g, 3), "optimizer_gb": round(o, 3)}
    if peak_gb is not None:
        out["activations_other_gb"] = round(max(peak_gb - p - g - o, 0.0), 3)
    return out
```

### ⑤ MFU

每 token 训练 FLOPs = 6 × matmul 参数 + 6 × 层数 × T × d（因果 attention）；重计算时 HFU 再加一遍前向。

[`trainperf/train.py#L299-L307`](../../../projects/trainperf/train.py#L299-L307)

```python
def mfu(cfg: dict, tokens_per_s_per_gpu: float, peak_tflops: float, measured_tflops: float | None) -> dict:
    """MFU = 每卡 tokens/s × 每 token 训练 FLOPs ÷ 峰值算力；HFU 把重计算的那一遍前向也算上。"""
    f = flops_per_token(cfg["preset"], cfg["seq_len"])
    achieved = tokens_per_s_per_gpu * f["model"] / 1e12
    hw = achieved + (tokens_per_s_per_gpu * f["recompute"] / 1e12 if cfg["ckpt"] == "full" else 0.0)
    return {"mfu": round(achieved / peak_tflops, 4), "mfu_dense_only": round(tokens_per_s_per_gpu * f["dense"] / 1e12 / peak_tflops, 4),
            "hfu": round(hw / peak_tflops, 4), "model_tflops_per_gpu": round(achieved, 2), "peak_tflops_used": peak_tflops,
            "mfu_vs_measured_peak": round(achieved / measured_tflops, 4) if measured_tflops else None,
            "attn_flops_share": round(f["attn"] / f["model"], 4)}
```

### ⑥ CPU 数据管线

每条样本：规范化 → 贪心分词 → 400 轮在线 BPE 合并，全是 Python。num_workers=0 时在训练主进程里串行做。

[`trainperf/data.py#L22-L107`](../../../projects/trainperf/data.py#L22-L107)

```python
def normalize(text: str) -> str:
    """逐字符：NFKC、去重音、小写、把非字母数字换成空格。"""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        base = unicodedata.normalize("NFKD", ch)[0]
        out.append(base.lower() if base.isalnum() else " ")
    return "".join(out)


def bpe_merges(ids: list[int], rounds: int, vocab: int) -> list[int]:
    """在线 BPE：每一轮数一遍相邻 token 对、把最常见的那一对合并成一个新 token。纯 Python，每轮 O(长度)。
    rounds 控制 CPU 工作量（慢分词器、逐样本的在线处理就是这个量级）。"""
    for _ in range(rounds):
        counts: dict[tuple[int, int], int] = {}
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break
        pair = max(counts, key=counts.get)
        new = (pair[0] * 1_000_003 + pair[1]) % (vocab - 1) + 1
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


class GreedyTokenizer:
    """词表 = 字符 + 常见子串（最长 6 个字符）；从左到右每次取最长的匹配。纯 Python，O(字符数 × 6) 次字典查找。"""

    def __init__(self, vocab: int, seed: int = 0):
        rng = random.Random(seed)
        pieces = set()
        for w in WORDS:
            n = normalize(w)
            for i in range(len(n)):
                for j in range(i + 1, min(len(n), i + 6) + 1):
                    pieces.add(n[i:j])
        pieces |= {chr(c) for c in range(32, 127)}
        self.ids = {p: (rng.randrange(1, vocab)) for p in sorted(pieces)}

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for L in range(6, 0, -1):
                tok = self.ids.get(text[i:i + L])
                if tok is not None:
                    out.append(tok)
                    i += L
                    break
            else:
                out.append(0)
                i += 1
        return out


class TextDataset(torch.utils.data.Dataset):
    """每条样本：随机拼一段"原始文本" → 规范化 → 贪心分词 → prep_passes 轮在线 BPE 合并 → 截断 / 补齐到 T + 1。
    所有工作都在 __getitem__ 里按样本做；文本按 3T 个词准备，合并之后仍够 T + 1 个 token。"""

    def __init__(self, T: int, vocab: int, passes: int, n: int, seed: int):
        self.T, self.vocab, self.passes, self.n, self.seed = T, vocab, passes, n, seed
        self.tok = GreedyTokenizer(vocab, seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        text = " ".join(rng.choice(WORDS) for _ in range(3 * self.T + 8))
        ids = bpe_merges(self.tok.encode(normalize(text)), self.passes, self.vocab)[: self.T + 1]
        ids += [0] * (self.T + 1 - len(ids))
        return torch.tensor(ids, dtype=torch.long)


def cpu_loader(cfg: dict, vocab: int, rank: int, n_batches: int) -> torch.utils.data.DataLoader:
    ds = TextDataset(cfg["seq_len"], vocab, cfg["prep_passes"], n_batches * cfg["mbs"], cfg["seed"] * 100 + rank)
    kw = {"batch_size": cfg["mbs"], "num_workers": cfg["num_workers"], "pin_memory": cfg["pin_memory"], "shuffle": False}
    if cfg["num_workers"] > 0:
        kw["prefetch_factor"] = cfg["prefetch_factor"]
    return torch.utils.data.DataLoader(ds, **kw)
```

---

[← 追踪](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
