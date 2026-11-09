"""trainperf 执行入口：硬件信息 → 启动 N 个 rank 训练 → rank0 汇总吞吐、MFU、显存构成、步时分解、profiler、通信计数 → summary。
单卡 OOM（7B 全参）在这里记录成"成功的运行"：OOM 发生在哪个阶段本身就是测量结果。"""

from __future__ import annotations

import os
import statistics

import torch

from . import measure as M
from .model import PRESETS, param_counts
from .train import STAGES, launch, mfu


def run(ctx, cfg: dict) -> None:
    ctx.stage("prepare")
    world = cfg["world_size"] if cfg["strategy"] != "none" else 1
    pc = param_counts(cfg["preset"])
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if world > 1 and torch.cuda.is_available() and n_gpus < world:
        raise RuntimeError(f"要 {world} 张卡，CUDA_VISIBLE_DEVICES 里只有 {n_gpus} 张")
    est = M.estimate(pc["params_total"], cfg["strategy"], world, cfg["precision"])
    ctx.info("data", {"kind": "随机 token（性能测量）" if cfg["data"] == "gpu_random" else "合成文本 + CPU 规范化与分词",
                      "seq_len": cfg["seq_len"], "mbs": cfg["mbs"], "world_size": world})
    ctx.info("model", {"arch": f"类 Qwen2.5（{cfg['preset']}）", "params": {"total": pc["params_total"], "trainable": pc["params_total"]},
                       "train_mode": "full", "dtype": "fp32 主参数 + bf16 autocast" if cfg["precision"] == "bf16" else "fp32",
                       "strategy": cfg["strategy"], "world_size": world, "device": gpu or "cpu"})
    ctx.log(f"{cfg['preset']}：{pc['params_total'] / 1e9:.2f}B 参数；{cfg['strategy']} × {world}；静态显存估算 {est['static_gb']:.1f} GB/卡")
    ctx.stage("train")
    r = launch(cfg, ctx)
    ctx.stage("evaluate")
    hw = {"gpu_name": gpu, "n_gpus": n_gpus, "cpu_count": os.cpu_count(), "peak_tflops_nominal": cfg["peak_tflops"], **r.get("hw", {})}
    common = {"model": {"preset": cfg["preset"], **pc, **PRESETS[cfg["preset"]]},
              "parallel": {"strategy": cfg["strategy"], "world_size": world, "mbs": cfg["mbs"], "seq_len": cfg["seq_len"],
                           "grad_accum": cfg["grad_accum"], "tokens_per_step_global": cfg["mbs"] * cfg["seq_len"] * cfg["grad_accum"] * world},
              "hw": hw}
    if r.get("oom"):
        idx = STAGES.index(r["stage"]) if r["stage"] in STAGES else len(STAGES)
        need = est["static_gb"] + 0.5 * pc["params_total"] * 2 / 1e9            # 静态 16Ψ + 粗估激活（约一份 bf16 权重）
        mem = {"oom": True, "oom_stage": r["stage"], "oom_stage_idx": idx, "oom_alloc_gb": r.get("oom_alloc_gb"),
               "peak_allocated_gb": r.get("oom_peak_gb"), "oom_need_gb_est": round(need, 1), "est": est,
               "device_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
               "method": "OOM：记录出错的阶段和当时的显存；需要量 = 16 字节/参数 + 激活粗估"}
        ctx.summary({**common, "memory": mem, "headline": {"oom_stage": r["stage"]},
                     "display": {"OOM 阶段序号（0 建模 / 1 前向 / 2 反向 / 3 优化器）": idx, "OOM 时已分配 GB": r.get("oom_alloc_gb") or 0,
                                 "估算需要 GB": round(need, 1)}})
        ctx.log(f"OOM：发生在 {r['stage']}，当时已分配 {r.get('oom_alloc_gb')} GB；估算需要 {need:.0f} GB")
        return
    tps = r["tokens_per_s_per_gpu"]
    m = mfu(cfg, tps, cfg["peak_tflops"], hw.get("matmul_tflops_measured"))
    ph = r["phase_ms"]
    step = sum(ph.values())
    time_ = {"data_ms": round(ph["data"], 2), "fwd_ms": round(ph["fwd"], 2), "bwd_ms": round(ph["bwd"], 2), "opt_ms": round(ph["opt"], 2),
             "step_ms": round(step, 2), "data_ms_max_rank": round(max(x["data_ms"] for x in r["per_rank"]), 2)}
    for k in ("data", "fwd", "bwd", "opt"):
        time_[f"{k}_frac"] = round(ph[k] / step, 4) if step else None
    peak = r["peak_gb"]
    bd = M.breakdown(r["snaps"], peak) if peak is not None else None
    mem = {"oom": False, "oom_stage": "none", "oom_stage_idx": len(STAGES), "peak_allocated_gb": round(peak, 2) if peak else None,
           "peak_reserved_gb": round(r["peak_reserved_gb"], 2) if r.get("peak_reserved_gb") else None,
           "device_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
           "per_rank_peak_gb": [round(x["peak_gb"], 2) if x["peak_gb"] else None for x in r["per_rank"]], "est": est,
           "method": "rank0 实测快照：建模（FSDP 为切分）后 = 参数；首个反向后增量 = 梯度；首次 step 后增量 = 优化器状态；"
                     "激活与其他 = 训练期间峰值 − 前三项（含 logits、通信缓冲、FSDP 的整份参数、碎片）"}
    if bd:
        mem["breakdown"] = bd
        mem["activations_frac"] = round(bd.get("activations_other_gb", 0) / peak, 4) if peak else None
    thr = {"tokens_per_s_per_gpu": round(tps, 1), "tokens_per_s_global": round(tps * world, 1),
           "step_ms_median": round(statistics.median(r["step_ms"]), 2), "step_ms_p90": round(sorted(r["step_ms"])[int(0.9 * (len(r["step_ms"]) - 1))], 2),
           "step_ms_window": round(r["win_s"] * 1000 / 15, 2)}
    comm = {"syncs_per_opt_step": r["syncs_per_opt_step"], "buckets": r.get("bucket_calls"),
            "grad_bytes_per_sync_gb": round(pc["params_total"] * 4 / 1e9, 2) if cfg["strategy"] == "ddp" else None}
    ctx.trace("step_timeline", {"note": "rank0 逐阶段计时窗口的前 3 步（每段前后同步，ms）", "steps": r["timeline"]})
    ctx.trace("memory_timeline", {"note": "rank0 显存快照（GB）", **{k: round(v, 3) for k, v in r["snaps"].items()},
                                  "peak": round(peak, 3) if peak else None, "estimate": est})
    disp = {"MFU": m["mfu"], "每卡 tokens/s": round(tps, 0), "峰值显存 GB": mem["peak_allocated_gb"] or 0,
            "步时 ms": thr["step_ms_median"], "等数据占比": time_["data_frac"]}
    if r.get("prof"):
        disp["GPU 空闲占比"] = r["prof"].get("gpu_idle_frac")
    ctx.summary({**common, "throughput": thr, "mfu": m, "memory": mem, "time": time_, "prof": r.get("prof") or {}, "comm": comm,
                 "loss": {"first": round(r["loss_first"], 4), "last": round(r["loss_last"], 4)},
                 "headline": {"mfu": m["mfu"], "tokens_per_s_per_gpu": round(tps, 1), "peak_allocated_gb": mem["peak_allocated_gb"]},
                 "display": {k: v for k, v in disp.items() if v is not None}, "display_pct": ["MFU", "等数据占比", "GPU 空闲占比"]})
    ctx.log(f"每卡 {tps:,.0f} tokens/s，MFU {m['mfu']:.1%}，步时 {thr['step_ms_median']:.0f} ms（数据 {ph['data']:.0f} / 前向 {ph['fwd']:.0f} / "
            f"反向 {ph['bwd']:.0f} / 优化器 {ph['opt']:.0f}），峰值 {mem['peak_allocated_gb']} GB")
