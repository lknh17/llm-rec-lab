"""在 GPU 上测 paged split-KV decode：扫 num_splits，对比 SDPA 和 flash_attn_with_kvcache（如果装了），和模型预测对照。

运行（公司 GPU，加锁）：
    uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.triton_kernels.bench_split_kv --hw H100
输出：stages/s03_kernels/experiments/split_kv_bench_<hw>.md
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tabulate import tabulate

from stages.s03_kernels.code.split_kv_decode import DECODE_HW, split_kv_model
from stages.s03_kernels.triton_kernels.split_kv_decode import contiguous_to_paged, split_kv_decode
from tools.bench import benchmark

EXP = Path(__file__).resolve().parents[1] / "experiments"
# Qwen2.5-7B 单层 attention
H, HKV, D, PAGE = 28, 4, 128, 16


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hw", default="H100", choices=sorted(DECODE_HW))
    args = p.parse_args()
    hw = DECODE_HW[args.hw]
    dev = torch.device("cuda")
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError:
        flash_attn_with_kvcache = None

    rows = []
    for b, t in [(1, 4096), (1, 10240), (1, 32768), (8, 10240), (32, 4096), (32, 10240)]:
        q = torch.randn(b, H, D, device=dev, dtype=torch.float16)
        k = torch.randn(b, HKV, t, D, device=dev, dtype=torch.float16)
        v = torch.randn(b, HKV, t, D, device=dev, dtype=torch.float16)
        kv_bytes = 2 * k.numel() * 2
        k_pool, v_pool, table = contiguous_to_paged(k, v, PAGE)
        lens = torch.full((b,), t, device=dev, dtype=torch.int32)

        def add(name, fn, model_us="-"):
            us = benchmark(fn, dev, warmup=10, iters=50).median_ms * 1e3
            gbs = kv_bytes / (us * 1e-6) / 1e9
            rows.append([b, t, name, f"{us:.1f}", f"{gbs:.0f}", f"{gbs / hw.bw_gbs:.0%}", model_us])

        # SDPA：q 当成长度 1 的序列；enable_gqa 免复制 KV
        add("SDPA (enable_gqa)", lambda: F.scaled_dot_product_attention(q[:, :, None], k, v, enable_gqa=True))
        if flash_attn_with_kvcache is not None:
            # flash_attn 的布局是 [B, T, Hkv, D]；num_splits=0 表示让库自己用启发式选
            kc, vc = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
            add("flash_attn_with_kvcache (auto split)",
                lambda: flash_attn_with_kvcache(q[:, None], kc, vc, cache_seqlens=lens, num_splits=0))
        for s in [1, 4, 16, 32, 64]:
            m = split_kv_model(b, H, HKV, t, D, s, hw)["total_us"]
            add(f"triton paged S={s}", lambda: split_kv_decode(q, k_pool, v_pool, table, lens, s), f"{m:.1f}")
        add("triton paged S=auto", lambda: split_kv_decode(q, k_pool, v_pool, table, lens, None))

    table_md = tabulate(rows, headers=["B", "T", "kernel", "μs", "有效 GB/s", "占 HBM 峰值", "模型预测 μs"],
                        tablefmt="github")
    report = (f"# split-KV decode benchmark（{torch.cuda.get_device_name()}，roofline 按 {hw.name}）\n\n"
              f"Qwen2.5-7B 单层：H={H}、Hkv={HKV}、D={D}、page={PAGE}、FP16。有效 GB/s = 必须读的 KV 字节 / 耗时。\n\n"
              f"{table_md}\n")
    print(report)
    EXP.mkdir(exist_ok=True)
    (EXP / f"split_kv_bench_{args.hw}.md").write_text(report)


if __name__ == "__main__":
    main()
