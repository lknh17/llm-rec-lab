"""每个 rank 的训练循环：包装（DDP / FSDP2 / 重计算 / compile）→ 热身 → 吞吐窗口（不插同步）→ 逐阶段计时窗口（每段同步）→ profiler。
rank0 在 labkit worker 进程里跑（持有 ctx），其余 rank 用 spawn 起子进程；结束时把各 rank 的统计 gather 到 rank0。"""

from __future__ import annotations

import contextlib
import datetime
import os
import platform
import socket
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import data as D
from . import measure as M
from .model import LM, flops_per_token, param_counts

STAGES = ("build", "forward", "backward", "optimizer")
WARMUP, THROUGHPUT = 5, 15


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --8<-- [start:wrap]
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
# --8<-- [end:wrap]


def make_opt(model, cfg):
    kw = {"lr": cfg["lr"], "betas": (0.9, 0.95), "weight_decay": 0.0}
    if cfg["adam"] == "fused":
        kw["fused"] = True
    elif cfg["adam"] == "foreach":
        kw["foreach"] = True
    else:
        kw["foreach"] = False
    return torch.optim.AdamW(model.parameters(), **kw)


def autocast(cfg, device):
    on = cfg["precision"] == "bf16" and not cfg["strategy"].startswith("fsdp")    # FSDP 靠 MixedPrecisionPolicy
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=on)


def run_rank(rank: int, world: int, port: int, cfg: dict, ctx=None) -> dict | None:
    """一个 rank 的全部工作。world == 1 时不建进程组。返回 rank0 汇总后的结果（其他 rank 返回 None）。"""
    use_cuda = cfg["device"] in ("auto", "cuda") and torch.cuda.is_available()
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        if platform.system() == "Darwin":
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        backend = cfg["backend"] if cfg["backend"] != "auto" else ("nccl" if use_cuda else "gloo")
        dist.init_process_group(backend, init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                                timeout=datetime.timedelta(minutes=3), device_id=device if use_cuda else None)
    try:
        return _train(rank, world, cfg, device, ctx)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(rank: int, world: int, cfg: dict, device, ctx) -> dict | None:
    log = (lambda m: ctx.log(m)) if ctx else (lambda m: None)
    hw = M.hwbench(device, world, rank) if cfg["bench_hw"] else {}
    torch.manual_seed(cfg["seed"])
    T, mbs, gas = cfg["seq_len"], cfg["mbs"], cfg["grad_accum"]
    st = {"stage": "build", "oom": False}
    snaps, timeline = {}, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        # ---------------- 建模 ----------------
        model = LM(cfg["preset"], cfg["attn"], T).to(device)
        V = model.s.vocab
        fsdp = cfg["strategy"].startswith("fsdp") and world > 1
        if not fsdp:                                                         # DDP 包装时就分配梯度桶，所以在包装前量参数
            snaps["after_build"] = M.snapshot(device)
        model = wrap(model, cfg, device, world)
        if fsdp:                                                             # FSDP：切分之后每卡只剩 1/N
            snaps["after_build"] = M.snapshot(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)                       # 不算建整份模型时的瞬时峰值
        opt = make_opt(model, cfg)
        ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        calls = [0]
        if ddp:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

            def hook(state, bucket):                                         # 每个桶 all-reduce 一次，计数
                calls[0] += 1
                return default_hooks.allreduce_hook(state, bucket)
            model.register_comm_hook(dist.group.WORLD, hook)
        gen = torch.Generator(device=device).manual_seed(cfg["seed"] * 1000 + rank)
        total = cfg["steps"]
        n_batches = (total + cfg["profile_steps"]) * gas + 4
        loader = iter(D.cpu_loader(cfg, V, rank, n_batches)) if cfg["data"] == "cpu_pipeline" else None

        def get_batch():
            if loader is None:
                return D.gpu_random(mbs, T, V, device, gen)
            return next(loader).to(device, non_blocking=cfg["pin_memory"])

        # --8<-- [start:step_timing]
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
        # --8<-- [end:step_timing]

        # ---------------- 热身 ----------------
        for s in range(WARMUP):
            one_step(s, timed=False)
        sync(device)
        if ctx:
            ctx.check()
        # ---------------- 吞吐窗口：不插同步 ----------------
        calls0, buckets = calls[0], st.get("bucket_calls", 1)
        t0 = time.perf_counter()
        losses = [one_step(WARMUP + s, timed=False)["loss"] for s in range(THROUGHPUT)]
        sync(device)
        win_s = time.perf_counter() - t0
        syncs = (calls[0] - calls0) / THROUGHPUT / buckets if ddp else None
        st["bucket_calls"] = buckets
        # ---------------- 逐阶段计时窗口 ----------------
        rows = []
        for s in range(WARMUP + THROUGHPUT, total):
            rows.append(one_step(s, timed=True))
            if ctx:
                ctx.check()
        timeline = [{k: round(v, 2) for k, v in r.items() if k != "loss"} for r in rows[:3]]
        # ---------------- profiler（只在 rank0、CUDA 上） ----------------
        prof = {}
        if cfg["profile_steps"] and device.type == "cuda":
            prof = _profile(one_step, total, cfg["profile_steps"], device, rank, world)
        peak = torch.cuda.max_memory_allocated(device) / M.GB if device.type == "cuda" else None
        peak_res = torch.cuda.max_memory_reserved(device) / M.GB if device.type == "cuda" else None
        st["stage"] = "done"
    except torch.cuda.OutOfMemoryError:
        if world > 1:
            raise                                                            # 多卡里的 OOM 是真失败（避免别的 rank 卡在集合通信里）
        st["oom"] = True
        st["oom_alloc_gb"] = round(torch.cuda.memory_allocated(device) / M.GB, 2)
        st["oom_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / M.GB, 2)
        model = opt = None
        torch.cuda.empty_cache()
        return {"oom": True, "stage": st["stage"], "hw": hw, "snaps": snaps, **{k: st[k] for k in ("oom_alloc_gb", "oom_peak_gb")}}

    loss_vals = [float(x) for x in losses]
    per = {k: statistics.median(r[k] for r in rows) for k in ("data", "fwd", "bwd", "opt")}
    mine = {"rank": rank, "data_ms": per["data"], "peak_gb": peak, "tokens_per_s": THROUGHPUT * gas * mbs * T / win_s}
    gathered = [None] * world
    if world > 1:
        dist.gather_object(mine, gathered if rank == 0 else None, dst=0)
    else:
        gathered = [mine]
    if rank != 0:
        return None
    step_ms = [sum(r[k] for k in ("data", "fwd", "bwd", "opt")) for r in rows]
    if ctx:
        for i, lv in enumerate(loss_vals):
            ctx.metric("train_loss", lv, step=WARMUP + i + 1, tokens_per_s=round(mine["tokens_per_s"], 1))
    return {"oom": False, "hw": hw, "snaps": snaps, "peak_gb": peak, "peak_reserved_gb": peak_res, "win_s": win_s,
            "tokens_per_s_per_gpu": mine["tokens_per_s"], "per_rank": gathered, "phase_ms": per,
            "step_ms": step_ms, "syncs_per_opt_step": syncs, "bucket_calls": st.get("bucket_calls"), "prof": prof,
            "timeline": timeline, "loss_first": loss_vals[0], "loss_last": loss_vals[-1]}


def _profile(one_step, total, n, device, rank, world) -> dict:
    from torch.profiler import ProfilerActivity, profile

    sync(device)
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        t0 = time.perf_counter()
        for s in range(n):
            one_step(total + s, timed=False)
        sync(device)
        wall = (time.perf_counter() - t0) * 1000
    ev = [(e.name, e.time_range.start, e.time_range.end) for e in p.events()
          if e.device_type == torch.autograd.DeviceType.CUDA and e.time_range.end > e.time_range.start]
    return M.kernel_stats(ev, wall, n) if ev else {}


# --8<-- [start:spawn]
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
# --8<-- [end:spawn]


# --8<-- [start:mfu]
def mfu(cfg: dict, tokens_per_s_per_gpu: float, peak_tflops: float, measured_tflops: float | None) -> dict:
    """MFU = 每卡 tokens/s × 每 token 训练 FLOPs ÷ 峰值算力；HFU 把重计算的那一遍前向也算上。"""
    f = flops_per_token(cfg["preset"], cfg["seq_len"])
    achieved = tokens_per_s_per_gpu * f["model"] / 1e12
    hw = achieved + (tokens_per_s_per_gpu * f["recompute"] / 1e12 if cfg["ckpt"] == "full" else 0.0)
    return {"mfu": round(achieved / peak_tflops, 4), "mfu_dense_only": round(tokens_per_s_per_gpu * f["dense"] / 1e12 / peak_tflops, 4),
            "hfu": round(hw / peak_tflops, 4), "model_tflops_per_gpu": round(achieved, 2), "peak_tflops_used": peak_tflops,
            "mfu_vs_measured_peak": round(achieved / measured_tflops, 4) if measured_tflops else None,
            "attn_flops_share": round(f["attn"] / f["model"], 4)}
# --8<-- [end:mfu]


def counts(cfg):
    return param_counts(cfg["preset"])
