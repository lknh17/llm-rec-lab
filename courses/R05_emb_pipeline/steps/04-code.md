<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 17 步：代码导读

> 阶段 1 / 7：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 计时

热身 → 每次前冲刷 L2（写 2 × L2 大小的缓冲，不计时）→ CUDA event 包住被测函数 → 一轮只同步一次 → 取中位数和 p10 / p90。

[`recops/bench.py#L31-L71`](../../../projects/recops/bench.py#L31-L71)

```python
    def time(self, fn, inner: int = 1) -> dict:
        """先热身，再至少计时 iters 次、总时长至少 min_time_ms；每次之前冲刷 L2（冲刷不计时）。返回每次调用的毫秒统计。"""
        for _ in range(self.warmup):
            fn()
        sync(self.device)
        ts = []
        if self.device.type == "cuda":
            total, n = 0.0, 0
            while n < self.iters or total < self.min_ms:
                evs = []
                for _ in range(max(1, self.iters)):
                    if self.buf is not None:
                        self.buf.zero_()
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(inner):
                        fn()
                    e.record()
                    evs.append((s, e))
                torch.cuda.synchronize()                                   # 一轮只同步一次
                batch = [s.elapsed_time(e) / inner for s, e in evs]
                ts += batch
                total += sum(batch) * inner
                n += len(batch)
                if n >= 10 * self.iters:
                    break
        else:
            t_end = time.perf_counter() + self.min_ms / 1000
            while len(ts) < self.iters or time.perf_counter() < t_end:
                t0 = time.perf_counter()
                for _ in range(inner):
                    fn()
                sync(self.device)
                ts.append((time.perf_counter() - t0) * 1000 / inner)
                if len(ts) >= 10 * self.iters:
                    break
        a = np.array(ts)
        med = float(np.median(a))
        p10, p90 = float(np.percentile(a, 10)), float(np.percentile(a, 90))
        return {"median_ms": round(med, 5), "p10_ms": round(p10, 5), "p90_ms": round(p90, 5),
                "cv": round((p90 - p10) / med, 4) if med else None, "n": len(a)}
```

### ② 查表微基准

有效带宽 = 读的字节 ÷ 时间，读的字节 = 样本数 × bag × d × 元素字节。理论下限按 HBM 标称带宽算。

[`recops/embops.py#L34-L59`](../../../projects/recops/embops.py#L34-L59)

```python
def emb_bench(cfg, device, timer: Timer, rows: int, dim: int, dtype: str, bag: int, dist: str) -> dict:
    """EmbeddingBag(mode="sum") 前向：B 个样本、每个 bag 个 id。
    有效带宽 = 读的字节（B·bag·d·元素字节 + id）÷ 时间；理论下限按 HBM 标称带宽算。"""
    out = {"rows": rows, "dim": dim, "dtype": dtype, "bag": bag, "dist": dist, "oom": False}
    table_gb = rows * dim * torch.tensor([], dtype=DT[dtype]).element_size() / 1e9
    out["table_gb"] = round(table_gb, 3)
    try:
        emb = nn.EmbeddingBag(rows, dim, mode="sum").to(device=device, dtype=DT[dtype])
        B = cfg["emb_batch"]
        ids = ids_for(B * bag, rows, dist, cfg["zipf_alpha"], cfg["seed"], device)
        off = torch.arange(0, B * bag, bag, device=device)
        esz = emb.weight.element_size()
        nbytes = B * bag * (dim * esz + 8)
        with torch.no_grad():
            t = timer.time(lambda: emb(ids, off))
        out.update(t)
        out["eff_gbs"] = round(nbytes / (t["median_ms"] / 1000) / 1e9, 1)
        out["floor_ms"] = round(nbytes / (cfg["hw_hbm_gbs"] * 1e9) * 1000, 4)
        out["unique_frac"] = round(float(torch.unique(ids).numel()) / ids.numel(), 4)
    except torch.cuda.OutOfMemoryError:
        out["oom"] = True
    finally:
        emb = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out
```

### ③ 稀疏 vs 稠密优化器

sparse=True 时 backward 产生稀疏梯度（只含本批碰到的行），SparseAdam 只读写这些行；否则梯度是整张表大小的稠密张量。

[`recops/embops.py#L75-L107`](../../../projects/recops/embops.py#L75-L107)

```python
def optim_bench(cfg, device, timer, rows: int, dim: int) -> dict:
    """一步 = 查表 → 求和 → backward → optimizer.step()。
    sparse=True：梯度是稀疏张量（只含本批碰到的行），SparseAdam 只更新这些行；
    sparse=False：梯度是 [rows, dim] 的稠密张量，Adam 每步读写整张表和两份状态。"""
    out = {}
    B, bag = cfg["emb_batch"], 10
    ids = ids_for(B * bag, rows, "uniform", 1.05, cfg["seed"], device)
    off = torch.arange(0, B * bag, bag, device=device)
    for kind in ("sparse_adam", "adam"):
        try:
            emb = nn.EmbeddingBag(rows, dim, mode="sum", sparse=kind == "sparse_adam").to(device)
            opt = torch.optim.SparseAdam(emb.parameters(), lr=1e-3) if kind == "sparse_adam" else torch.optim.Adam(emb.parameters(), lr=1e-3)

            def step():
                opt.zero_grad(set_to_none=True)
                emb(ids, off).sum().backward()
                opt.step()

            step()
            mem = peak_delta(step, device)
            t = timer.time(step)
            state = sum(v.numel() * v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
            out[kind] = {**t, "peak_delta_gb": mem, "state_gb": round(state / 1e9, 3),
                         "table_gb": round(rows * dim * 4 / 1e9, 3), "oom": False}
        except torch.cuda.OutOfMemoryError:
            out[kind] = {"median_ms": None, "oom": True}
        finally:
            emb = opt = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if out["sparse_adam"].get("median_ms") and out["adam"].get("median_ms"):
        out["dense_over_sparse"] = round(out["adam"]["median_ms"] / out["sparse_adam"]["median_ms"], 2)
    return out
```

### ④ 特征处理

两种写法结果完全相同（测试里对拍），差别只在一个是整批 numpy、一个是 Python 三重循环。

[`recops/embops.py#L125-L138`](../../../projects/recops/embops.py#L125-L138)

```python
def process_vectorized(raw: RawShard, idx: np.ndarray, rows: int) -> dict:
    """整批：log1p 数值、取模哈希类别和多值，全部是 numpy 向量操作。"""
    return {"dense": np.log1p(raw.dense[idx]), "cat": raw.cat[idx] % rows, "mh": raw.mh[idx] % rows, "label": raw.label[idx]}


def process_python(raw: RawShard, idx: np.ndarray, rows: int) -> dict:
    """同样的结果，逐样本、逐特征用 Python 循环算（教学故障：一批 4096 × 141 次 Python 操作）。"""
    dense, cat, mh = [], [], []
    for i in idx:
        dense.append([float(np.log1p(v)) for v in raw.dense[i]])
        cat.append([int(v) % rows for v in raw.cat[i]])
        mh.append([[int(v) % rows for v in row] for row in raw.mh[i]])
    return {"dense": np.asarray(dense, np.float32), "cat": np.asarray(cat, np.int64), "mh": np.asarray(mh, np.int64),
            "label": raw.label[idx]}
```

### ⑤ 端到端逐阶段计时

每个阶段前后都同步，拿到真实耗时（会略微降低吞吐）；吞吐窗口里不插同步。

[`recops/embops.py#L219-L279`](../../../projects/recops/embops.py#L219-L279)

```python
def e2e_bench(raw, cfg, device) -> dict:
    """端到端训练：先按 e2e_profile_steps 步逐阶段计时（每段前后同步），再不插同步地跑吞吐窗口。"""
    sparse = cfg["emb_optim"] == "sparse_adam"
    model = DLRM(cfg["e2e_rows"], cfg["n_tables"], sparse=sparse).to(device)
    emb_params = [p for t in model.tables for p in t.parameters()]
    other = [p for n, p in model.named_parameters() if not n.startswith("tables")]
    opts = [torch.optim.Adam(other, lr=1e-3),
            torch.optim.SparseAdam(emb_params, lr=1e-3) if sparse else torch.optim.Adam(emb_params, lr=1e-3)]
    w = effective_workers(cfg)
    warm = 3 + w * cfg["prefetch_factor"]                                   # 热身先耗尽预取缓冲，计时窗口落在稳态
    n_total = warm + cfg["e2e_profile_steps"] + cfg["e2e_steps"] + 5
    it = iter(make_loader(raw, cfg, n_total, w))
    nb = cfg["non_blocking"] and cfg["pin_memory"]

    def to_dev(b):
        return {k: v.to(device, non_blocking=nb) for k, v in b.items()}

    def step(bd):
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(bd), bd["label"])
        loss.backward()
        for o in opts:
            o.step()
        return loss

    for _ in range(warm):                                                   # 热身
        step(to_dev(next(it)))
    sync(device)
    stages = {"wait_data": [], "h2d": [], "forward_backward": [], "optimizer": []}
    for _ in range(cfg["e2e_profile_steps"]):
        t0 = time.perf_counter(); b = next(it); t1 = time.perf_counter()
        bd = to_dev(b); sync(device); t2 = time.perf_counter()
        for o in opts:
            o.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(model(bd), bd["label"]).backward(); sync(device); t3 = time.perf_counter()
        for o in opts:
            o.step()
        sync(device); t4 = time.perf_counter()
        for k, a, z in (("wait_data", t0, t1), ("h2d", t1, t2), ("forward_backward", t2, t3), ("optimizer", t3, t4)):
            stages[k].append((z - a) * 1000)
    st = {k: round(float(np.median(v)), 3) for k, v in stages.items()}
    tot = sum(st.values())
    share = {k: round(v / tot, 4) for k, v in st.items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, n = time.perf_counter(), 0
    for _ in range(cfg["e2e_steps"]):
        step(to_dev(next(it)))
        n += 1
        if time.perf_counter() - t0 > cfg["e2e_max_seconds"]:
            break
    sync(device)
    secs = time.perf_counter() - t0
    emb_gb = sum(p.numel() * p.element_size() for p in emb_params) / 1e9
    state_gb = sum(v.numel() * v.element_size() for o in opts[1:] for s in o.state.values() for v in s.values() if torch.is_tensor(v)) / 1e9
    return {"samples_per_s": round(n * cfg["batch_size"] / secs, 1), "steps_measured": n, "seconds": round(secs, 2),
            "stage_ms": st, "stage_share": share, "workers": w,
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
            "emb_gb": round(emb_gb, 2), "opt_state_gb": round(state_gb, 2),
            "params": sum(p.numel() for p in model.parameters())}
```

---

[← 追踪](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
