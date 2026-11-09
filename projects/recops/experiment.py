"""recops 执行入口。suite=r05：Embedding 微基准 / 表规模网格 / 优化器 / 数据管线 / 端到端 DLRM。
suite=r06：两种打分 × 四种实现 × 长度 × 头维度的耗时、显存、误差，以及一个 HSTU 层的端到端占比。"""

from __future__ import annotations

import math

import torch

from . import attn as A
from . import bench as BN
from . import embops as E


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ints(s) -> list[int]:
    return [int(float(x)) for x in str(s).split(",") if x.strip()]


def run(ctx, cfg: dict) -> None:
    ctx.stage("prepare")
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    hw = BN.hw_info(device)
    hw.update({"ref_copy_gbs": BN.ref_copy_gbs(device, cfg), "ref_matmul_tflops": BN.ref_matmul_tflops(device, cfg)})
    ctx.info("data", {"kind": "微基准（随机输入）", "suite": cfg["suite"], **{k: hw[k] for k in hw if k in ("gpu", "sms", "l2_mb")}})
    ctx.log(f"硬件：{hw.get('gpu', hw['device'])}；拷贝带宽实测 {hw['ref_copy_gbs']} GB/s，bf16 矩阵乘实测 {hw['ref_matmul_tflops']} TFLOPS")
    ctx.stage("evaluate")
    (run_r05 if cfg["suite"] == "r05" else run_r06)(ctx, cfg, device, hw)


# ------------------------------------------------------------------ R05
def run_r05(ctx, cfg, device, hw) -> None:
    parts = ["emb", "optim", "loader", "e2e"] if cfg["parts"] == "auto" else [p.strip() for p in cfg["parts"].split(",")]
    timer = BN.Timer(device, cfg)
    out = {"hw": hw}
    if "emb" in parts:
        emb = {}
        for bag in ints(cfg["bag_lens"]):
            for dist in ("uniform", "zipf"):
                r = E.emb_bench(cfg, device, timer, cfg["emb_rows"], cfg["emb_dim"], cfg["emb_dtype"], bag,
                                dist if cfg["id_dist"] == "uniform" or dist == "zipf" else dist)
                emb[f"bag{bag}_{dist}"] = r
                if r.get("eff_gbs") is not None:
                    ctx.metric("emb_eff_gbs", r["eff_gbs"], step=bag, split=dist)
                ctx.check()
        def ratio(a, b):
            return round(a / b, 3) if a and b else None
        emb["ratio"] = {"bag50_over_bag1_gbs": ratio(emb.get("bag50_uniform", {}).get("eff_gbs"), emb.get("bag1_uniform", {}).get("eff_gbs")),
                        "zipf_over_uniform_ms": ratio(emb.get("bag10_zipf", {}).get("median_ms"), emb.get("bag10_uniform", {}).get("median_ms"))}
        out["emb"] = emb
        b10 = emb.get("bag10_uniform", {})
        ctx.trace("emb", {"note": "EmbeddingBag 查表：10M × 64 fp32 表，16384 个样本 × 每个 10 个 id（均匀）",
                          "table_gb": b10.get("table_gb"), "bytes_read_mb": round(cfg["emb_batch"] * 10 * cfg["emb_dim"] * 4 / 1e6, 1),
                          "median_ms": b10.get("median_ms"), "eff_gbs": b10.get("eff_gbs"), "floor_ms": b10.get("floor_ms"),
                          "ref_copy_gbs": hw["ref_copy_gbs"]})
    if "grid" in parts:
        out["grid"] = E.emb_grid(cfg, device, timer)
    if "optim" in parts:
        out["optim"] = E.optim_bench(cfg, device, timer, cfg["emb_rows"], cfg["emb_dim"])
    if "loader" in parts or "e2e" in parts:
        raw = E.RawShard(200_000, cfg["n_tables"], cfg["multi_hot_len"], cfg["seed"])
        idx = torch.arange(3).numpy()
        ctx.trace("feature", {"note": "3 条原始样本 → 处理后：数值 log1p、类别和多值按表行数取模",
                              "raw_cat_head": raw.cat[idx, :3].tolist(), "cat_after": (raw.cat[idx, :3] % cfg["e2e_rows"]).tolist(),
                              "raw_dense_head": [[round(float(x), 3) for x in r] for r in raw.dense[idx, :3]],
                              "batch_shapes": {"dense": [cfg["batch_size"], 13], "cat": [cfg["batch_size"], cfg["n_tables"]],
                                               "mh": [cfg["batch_size"], 2, cfg["multi_hot_len"]]}})
        if "loader" in parts:
            out["loader"] = E.loader_bench(raw, cfg, device)
            ctx.check()
        if "e2e" in parts:
            out["e2e"] = E.e2e_bench(raw, cfg, device)
            for k, v in out["e2e"]["stage_ms"].items():
                ctx.metric("stage_ms", v, step=None, split=k)
            ctx.log(f"端到端：{out['e2e']['samples_per_s']:,.0f} 样本/秒；逐阶段 ms {out['e2e']['stage_ms']}")
    disp = {}
    if "emb" in out:
        disp["查表有效带宽 GB/s（bag 10 均匀）"] = out["emb"].get("bag10_uniform", {}).get("eff_gbs")
    if "optim" in out and out["optim"].get("dense_over_sparse"):
        disp["稠密 Adam ÷ SparseAdam 每步耗时"] = out["optim"]["dense_over_sparse"]
    if "e2e" in out:
        disp.update({"端到端 样本/秒": out["e2e"]["samples_per_s"], "等数据占比": out["e2e"]["stage_share"]["wait_data"],
                     "优化器占比": out["e2e"]["stage_share"]["optimizer"]})
    if "loader" in out:
        disp["数据管线 样本/秒"] = out["loader"]["samples_per_s"]
    ctx.summary({**out, "headline": {k: v for k, v in disp.items()}, "display": disp,
                 "display_pct": ["等数据占比", "优化器占比"]})


# ------------------------------------------------------------------ R06
DTS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def run_r06(ctx, cfg, device, hw) -> None:
    timer = BN.Timer(device, cfg)
    dt = DTS[cfg["attn_dtype"]]
    B, H = cfg["attn_batch"], cfg["attn_heads"]
    impl = cfg["attn_impl"]
    kinds = [k.strip() for k in cfg["attn_kinds"].split(",")]
    lens = ints(cfg["seq_lens"]) + ([cfg["ragged_len"]] if cfg["ragged_len"] else [])
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    rab = (torch.randn(cfg["rab_buckets"], generator=g) * 0.1).to(device)
    res, worst = {}, 0.0
    for kind in kinds:
        if impl == "sdpa" and kind != "softmax":
            continue
        for D in ints(cfg["head_dims"]):
            for L in lens:
                key = f"{kind}_L{L}_D{D}"
                q, k, v = (torch.randn(B, H, L, D, generator=g).to(device, dt) for _ in range(3))
                r = {"oom": False}
                try:
                    fn = lambda: A.run_impl(impl, q, k, v, kind, rab, cfg)  # noqa: E731
                    o = fn()
                    if cfg["check_ref"]:
                        ref = A.naive(q.float(), k.float(), v.float(), kind, rab)
                        err = float((o.float() - ref).norm() / ref.norm().clamp(min=1e-12))
                        r["rel_err"] = round(err, 6)
                        worst = max(worst, err)
                    r["peak_delta_gb"] = BN.peak_delta(fn, device)
                    t = timer.time(fn)
                    fl = A.attn_flops(B, H, L, D)
                    nb = A.attn_bytes(B, H, L, D, q.element_size(), impl)
                    rl = BN.roofline(fl, nb, cfg["hw_peak_tflops"], cfg["hw_hbm_gbs"])
                    r.update(t)
                    r.update({"tflops": round(fl / (t["median_ms"] / 1000) / 1e12, 3), "ai": rl["ai"], "bound_tflops": rl["bound_tflops"],
                              "roofline_frac": round(fl / (t["median_ms"] / 1000) / 1e12 / rl["bound_tflops"], 4)})
                except torch.cuda.OutOfMemoryError:
                    r.update({"oom": True, "median_ms": None})
                except Exception as e:  # noqa: BLE001
                    r.update({"error": f"{type(e).__name__}: {str(e)[:120]}", "median_ms": None})
                finally:
                    q = k = v = o = None
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                res[key] = r
                if r.get("median_ms") is not None:
                    ctx.metric("attn_ms", r["median_ms"], step=L, split=f"{kind}_D{D}")
                ctx.check()
    layer = hstu_layer(cfg, device, timer, rab, dt)
    first = next(iter(res.values()))
    ctx.trace("kernel", {"note": "一次调用的形状、grid 和每个 program 读写的量（Triton 实现）",
                         "q_shape": [B, H, lens[0], ints(cfg["head_dims"])[0]], "impl": impl,
                         "grid": [math.ceil(lens[0] / cfg["block_m"]), B * H], "block": [cfg["block_m"], cfg["block_n"]],
                         "first_result": first})
    disp = {}
    for kind in kinds:
        for L in ints(cfg["seq_lens"]):
            r = res.get(f"{kind}_L{L}_D64", {})
            if r.get("median_ms") is not None:
                disp[f"{kind} L={L} ms"] = r["median_ms"]
    errors = sorted({r["error"] for r in res.values() if r.get("error")})
    if errors:
        ctx.warn("attn_error", f"有 {sum(1 for r in res.values() if r.get('error'))} 个组合没跑成：{errors[0]}")
    disp["最大相对误差"] = round(worst, 6)
    disp["HSTU 层里 attention 占比"] = layer.get("attn_share")
    ctx.summary({"attn": res, "layer": layer, "hw": hw, "check": {"max_rel_err": round(worst, 6), "errors": errors},
                 "headline": {"max_rel_err": round(worst, 6)}, "display": disp, "display_pct": ["HSTU 层里 attention 占比"]})


def hstu_layer(cfg, device, timer, rab, dt) -> dict:
    """一个 HSTU 层（f1 → pointwise attention → f2），前向，看 attention 占整层的比例。"""
    d, L, H = cfg["layer_d_model"], cfg["layer_len"], cfg["attn_heads"]
    B = cfg["attn_batch"]
    D = d // H
    if D not in (16, 32, 64, 128) and cfg["attn_impl"] == "triton":
        return {}
    try:
        x = torch.randn(B, L, d, device=device, dtype=dt)
        f1, f2 = torch.nn.Linear(d, 4 * d).to(device, dt), torch.nn.Linear(d, d).to(device, dt)
        impl = "naive" if cfg["attn_impl"] == "sdpa" else cfg["attn_impl"]

        def proj():
            u, v, q, k = torch.nn.functional.silu(f1(x)).view(B, L, 4, H, D).permute(2, 0, 3, 1, 4)
            return u, v.contiguous(), q.contiguous(), k.contiguous()

        u, v, q, k = proj()
        t_attn = timer.time(lambda: A.run_impl(impl, q, k, v, "pointwise", rab, cfg))["median_ms"]

        def full():
            u, v, q, k = proj()
            o = A.run_impl(impl, q, k, v, "pointwise", rab, cfg)
            return f2(o.transpose(1, 2).reshape(B, L, d) * u.transpose(1, 2).reshape(B, L, d))

        t_full = timer.time(full)["median_ms"]
        return {"impl": impl, "attn_ms": t_attn, "layer_ms": t_full, "attn_share": round(t_attn / t_full, 4), "L": L, "d_model": d}
    except torch.cuda.OutOfMemoryError:
        return {"oom": True}
    except Exception as e:  # noqa: BLE001  （例如 CPU 上没有 Triton）
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
