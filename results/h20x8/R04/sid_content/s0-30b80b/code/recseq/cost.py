"""成本：解析估算（FLOPs、推理计算量、手写注意力的显存）和实测探针（耗时、峰值显存、OOM、jagged 对补齐）。"""

from __future__ import annotations

import time

import numpy as np
import torch

from .models import build

# 手写注意力每个 (b, h, i, j) 元素在反向时要保存的字节数（bf16）：
#   SASRec manual：打分 s 2 + softmax 输出 2 + 因果 mask 广播 ≈ 2（dropout mask 等）≈ 6
#   HSTU manual：  打分 s 2 + 位置偏置 2 + 时间偏置 2 + 时间桶下标 8（int64）+ SiLU 输入 / 输出 2 + mask 1 ≈ 15（下标是大头）
BYTES_PER_ELEM = {"sasrec": 6, "hstu": 15}


# --8<-- [start:estimate]
def estimate(cfg: dict, V: int, L: int) -> dict:
    """每 token 前向乘加（MAC）：线性层 SASRec 12d²（注意力投影 4d² + MLP 8d²）、HSTU 5d²（f1 4d² + f2 d²），每层；
    注意力 QKᵀ 和 AV 在因果下平均每个 token 看 L/2 个 key，合计 L·d，每层。注意力占比 = L / (L + 12d) 或 L / (L + 5d)。
    推理（只算最后一个位置的打分，对全部 V 个候选）：
      DIN    ≈ 2·L·V·d（每个候选都要对 L 个历史打分、加权）
      SASRec ≈ n_layer·(12d²·L + L²·d) + V·d（整段过一遍编码器，打分只是一次 [d] × [d, V]）"""
    d, nl, m = cfg["d_model"], cfg["n_layer"], cfg["model"]
    lin = {"sasrec": 12, "hstu": 5}.get(m, 0) * d * d
    attn = L * d if m in ("sasrec", "hstu") else 0
    if m == "din":
        infer = 2 * L * V * d
    elif m == "mean":
        infer = L * d + d * d + V * d
    else:
        infer = nl * (lin * L + L * L * d) + V * d
    return {"linear_per_token": nl * lin, "attn_per_token": nl * attn,
            "attn_share": round(attn / (attn + lin), 4) if attn + lin else 0.0, "infer_macs_per_user": int(infer)}


def est_attn_gb(cfg: dict, L: int, B: int = 1) -> float | None:
    """手写注意力反向保存的 [B, H, L, L] 张量：n_layer · B · H · L² · 每元素字节。"""
    beta = BYTES_PER_ELEM.get(cfg["model"])
    return round(cfg["n_layer"] * B * cfg["n_head"] * L * L * beta / 1e9, 3) if beta else None
# --8<-- [end:estimate]


def _fake_batch(V: int, L: int, B: int, device, lengths: list[int] | None = None) -> dict:
    g = torch.Generator().manual_seed(0)
    x = torch.randint(1, V, (B, L), generator=g)
    if lengths is not None:
        for i, n in enumerate(lengths):
            x[i, : L - n] = 0
    ts = torch.cumsum(torch.randint(1, 3600, (B, L), generator=g), 1)
    nxt = torch.cat([ts[:, 1:], ts[:, -1:] + 60], 1)
    y = torch.where(x != 0, torch.randint(1, V, (B, L), generator=g), 0)
    return {k: v.to(device) for k, v in {"x": x, "ts": ts, "ts_next": nxt, "y": y}.items()}


def _fwd_bwd(model, b, device, cfg):
    from .train import autocast

    with autocast(device, cfg):
        logits = model(b)
        loss = logits.float().logsumexp(-1).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)


def _time(fn, device, reps: int = 5) -> float:
    for _ in range(2):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) * 1000 / reps


# --8<-- [start:probe]
def probe(cfg: dict, V: int, L: int, impl: str, device, B: int = 1) -> dict:
    """随机输入、B 条长度 L 的序列，前向 + 反向的平均毫秒和峰值显存（CUDA 上）；放不下记 oom。"""
    c = {**cfg, "attn_impl_eff": impl, "max_len_eff": max(L, cfg["max_len_eff"])}
    out = {"ms": None, "peak_gb": None, "est_gb": est_attn_gb(c, L, B) if impl == "manual" else None, "oom": False}
    try:
        model = build(V, c).to(device)
        b = _fake_batch(V, L, B, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        out["ms"] = round(_time(lambda: _fwd_bwd(model, b, device, c), device), 3)
        if device.type == "cuda":
            out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except torch.cuda.OutOfMemoryError:
        out["oom"] = True
    finally:
        model = b = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def jagged_probe(cfg: dict, V: int, lengths: list[int], device) -> dict:
    """同一批用户：补齐到最长后整批前向 + 反向，对比逐条不补齐分别算再求和。"""
    L = max(lengths)
    c = {**cfg, "max_len_eff": max(L, cfg["max_len_eff"])}
    try:
        model = build(V, c).to(device)
        padded = _fake_batch(V, L, len(lengths), device, lengths)
        singles = [_fake_batch(V, n, 1, device) for n in lengths]
        ms_p = _time(lambda: _fwd_bwd(model, padded, device, c), device, reps=3)
        ms_j = _time(lambda: [_fwd_bwd(model, s, device, c) for s in singles], device, reps=3)
        return {"lengths": lengths, "ms_padded": round(ms_p, 2), "ms_jagged": round(ms_j, 2), "speedup": round(ms_p / ms_j, 2)}
    except torch.cuda.OutOfMemoryError:
        return {"lengths": lengths, "ms_padded": None, "ms_jagged": None, "speedup": None, "oom": True}
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
# --8<-- [end:probe]


def sdpa_backend(device) -> str:
    """SDPA 在这台机器上会选哪个后端（只是可用性，实际选择由 PyTorch 按输入决定）。"""
    if device.type != "cuda":
        return f"{device.type}（无 flash）"
    b = torch.backends.cuda
    return "flash" if b.flash_sdp_enabled() else ("mem_efficient" if b.mem_efficient_sdp_enabled() else "math")


def lengths_sample(N: np.ndarray, k: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(N, size=min(k, len(N)), replace=False))
