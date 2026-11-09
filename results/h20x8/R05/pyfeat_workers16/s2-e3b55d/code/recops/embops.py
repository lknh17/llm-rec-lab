"""R05：Embedding 查表的带宽、表规模与精度、稀疏 vs 稠密优化器、数据管线、端到端 DLRM 的逐阶段耗时。"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .bench import Timer, peak_delta, sync

DT = {"fp32": torch.float32, "bf16": torch.bfloat16}


def ids_for(n: int, rows: int, dist: str, alpha: float, seed: int, device) -> torch.Tensor:
    """n 个查表 id。zipf：rank = ⌊U^(−1/(α−1))⌋（截断到 rows），再用固定的随机置换打散热门 id 在表里的位置。"""
    g = np.random.default_rng(seed)
    if dist == "uniform":
        ids = g.integers(0, rows, size=n)
    else:
        u = g.random(n)
        with np.errstate(over="ignore"):
            r = np.floor(np.minimum(u ** (-1.0 / (alpha - 1.0)), float(rows)))          # 先在浮点里截断，避免溢出后转整数出错
        rank = np.minimum(r.astype(np.int64) - 1, rows - 1)
        perm = np.random.default_rng(seed + 1).permutation(min(rows, 10_000_000))
        ids = np.where(rank < len(perm), perm[np.minimum(rank, len(perm) - 1)], rank)
    return torch.from_numpy(ids.astype(np.int64)).to(device)


# --8<-- [start:emb_bench]
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
# --8<-- [end:emb_bench]


def emb_grid(cfg, device, timer) -> dict:
    """表规模 × 维度 × 精度：显存占用和查表耗时（bag 10、均匀分布）。"""
    out = {}
    for r in [int(float(x)) for x in str(cfg["grid_rows"]).split(",")]:
        for d in [int(x) for x in str(cfg["grid_dims"]).split(",")]:
            for dt in str(cfg["grid_dtypes"]).split(","):
                key = f"r{r:.0e}_d{d}_{dt}".replace("+0", "").replace("+", "")
                out[key] = emb_bench(cfg, device, timer, r, d, dt, 10, "uniform")
    return out


# --8<-- [start:optim]
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
# --8<-- [end:optim]


# ------------------------------------------------------------------ 数据管线
class RawShard:
    """一份"原始样本"：13 个数值 + 26 个类别（字符串哈希前的整数）+ 2 个多值特征。按样本下标取，模拟读出的原始日志。"""

    def __init__(self, n: int, n_tables: int, mh: int, seed: int):
        g = np.random.default_rng(seed)
        self.dense = g.lognormal(0, 1, size=(n, 13)).astype(np.float32)
        self.cat = g.integers(0, 2 ** 40, size=(n, n_tables), dtype=np.int64)
        self.mh = g.integers(0, 2 ** 40, size=(n, 2, mh), dtype=np.int64)
        self.label = (g.random(n) < 0.25).astype(np.float32)
        self.n = n


# --8<-- [start:feature_proc]
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
# --8<-- [end:feature_proc]


class BatchSet(torch.utils.data.Dataset):
    """每个 __getitem__ 返回一整批（DataLoader 的 batch_size=None）：批量处理才谈得上向量化。"""

    def __init__(self, raw: RawShard, B: int, rows: int, proc: str, n_batches: int, seed: int):
        self.raw, self.B, self.rows, self.proc, self.n, self.seed = raw, B, rows, proc, n_batches, seed

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        idx = np.random.default_rng(self.seed + k).integers(0, self.raw.n, size=self.B)
        f = process_vectorized if self.proc == "vectorized" else process_python
        b = f(self.raw, idx, self.rows)
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}


def make_loader(raw, cfg, n_batches: int, workers: int):
    ds = BatchSet(raw, cfg["batch_size"], cfg["e2e_rows"], cfg["feature_proc"], n_batches, cfg["seed"])
    kw = {"batch_size": None, "num_workers": workers, "pin_memory": cfg["pin_memory"] and torch.cuda.is_available()}
    if workers > 0:
        kw.update(prefetch_factor=cfg["prefetch_factor"], persistent_workers=False)
    return torch.utils.data.DataLoader(ds, **kw)


def effective_workers(cfg) -> int:
    return min(cfg["num_workers"], max(1, (os.cpu_count() or 8) // 2)) if cfg["num_workers"] > 0 else 0


def loader_bench(raw, cfg, device) -> dict:
    """只取数（不训练）：每秒能产出多少批；另测一批的特征处理耗时（单进程）。"""
    w = effective_workers(cfg)
    n = 60
    it = iter(make_loader(raw, cfg, n, w))
    next(it)
    t0 = time.perf_counter()
    for _ in range(n - 1):
        next(it)
    per = (time.perf_counter() - t0) / (n - 1)
    idx = np.random.default_rng(0).integers(0, raw.n, size=cfg["batch_size"])
    f = process_vectorized if cfg["feature_proc"] == "vectorized" else process_python
    t1 = time.perf_counter()
    f(raw, idx, cfg["e2e_rows"])
    proc_ms = (time.perf_counter() - t1) * 1000
    return {"workers": w, "batches_per_s": round(1 / per, 2), "samples_per_s": round(cfg["batch_size"] / per, 1),
            "proc_ms_per_batch": round(proc_ms, 2)}


# ------------------------------------------------------------------ DLRM
# --8<-- [start:dlrm]
class DLRM(nn.Module):
    """13 个数值 → MLP(512-256-64)；26 张单值表 + 2 张多值表（每张 e2e_rows × 64，sparse 梯度）；
    28 个 embedding 和数值塔输出两两内积（dot interaction）→ MLP(512-256-1)。"""

    def __init__(self, rows: int, n_tables: int, d: int = 64, sparse: bool = True):
        super().__init__()
        self.tables = nn.ModuleList(nn.EmbeddingBag(rows, d, mode="sum", sparse=sparse) for _ in range(n_tables + 2))
        self.bot = nn.Sequential(nn.Linear(13, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, d), nn.ReLU())
        f = n_tables + 3
        self.top = nn.Sequential(nn.Linear(d + f * (f - 1) // 2, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.n_tables = n_tables
        self.iu = None

    def forward(self, b):
        B = b["dense"].shape[0]
        x = self.bot(b["dense"])                                            # [B, d]
        embs = [self.tables[j](b["cat"][:, j:j + 1]) for j in range(self.n_tables)]   # 每个 [B, d]
        for k in range(2):
            embs.append(self.tables[self.n_tables + k](b["mh"][:, k]))    # 多值：bag 求和 [B, d]
        z = torch.stack([x] + embs, 1)                                      # [B, F, d]
        inter = z @ z.transpose(1, 2)                                       # [B, F, F]
        if self.iu is None or self.iu[0].device != z.device:
            self.iu = torch.triu_indices(z.shape[1], z.shape[1], 1, device=z.device)
        return self.top(torch.cat([x, inter[:, self.iu[0], self.iu[1]]], 1)).squeeze(-1)
# --8<-- [end:dlrm]


# --8<-- [start:e2e]
def e2e_bench(raw, cfg, device) -> dict:
    """端到端训练：先按 e2e_profile_steps 步逐阶段计时（每段前后同步），再不插同步地跑吞吐窗口。"""
    sparse = cfg["emb_optim"] == "sparse_adam"
    model = DLRM(cfg["e2e_rows"], cfg["n_tables"], sparse=sparse).to(device)
    emb_params = [p for t in model.tables for p in t.parameters()]
    other = [p for n, p in model.named_parameters() if not n.startswith("tables")]
    opts = [torch.optim.Adam(other, lr=1e-3),
            torch.optim.SparseAdam(emb_params, lr=1e-3) if sparse else torch.optim.Adam(emb_params, lr=1e-3)]
    w = effective_workers(cfg)
    n_total = cfg["e2e_profile_steps"] + cfg["e2e_steps"] + 5
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

    for _ in range(3):                                                      # 热身
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
# --8<-- [end:e2e]
