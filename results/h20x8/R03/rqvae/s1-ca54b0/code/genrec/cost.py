"""成本基准：千万级物品的全量打分（[U, d] × [d, N] + topk）对比 beam 解码的每用户耗时，和物品表的显存。"""

from __future__ import annotations

import time

import torch


# --8<-- [start:cost]
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
# --8<-- [end:cost]
