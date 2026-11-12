"""S2 核心实验：整模型 prefill / decode 的实测 vs roofline 预测。

用随机权重即可（性能和权重数值无关）。KV cache 直接预填到 context 长度，只测 decode 一步。

Mac（验证方法）：uv run python -m stages.s02_perf.code.decode_bench
GPU（真实结论）：uv run python -m stages.s02_perf.code.decode_bench --preset llama3_8b --hw H100 --context 4096 \
                    --batches 1 4 16 32 64 128
输出：experiments/decode_bench_<preset>_<hw>.md 和 .png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tabulate import tabulate

from stages.s01_transformer.code.model import GPT, PRESETS, GPTConfig, KVCache
from tools.bench import benchmark
from tools.device import describe, get_device
from tools.roofline import HARDWARE, ModelShape, decode, max_batch, prefill

EXP = Path(__file__).resolve().parents[1] / "experiments"

# Mac 上能跑、又足够大（约 0.5B，FP16 约 1GB）的配置：让"读权重"的时间显著超过固定开销
MAC_DEMO = GPTConfig(vocab_size=32000, n_layer=8, n_head=16, n_kv_head=4, d_model=2048, ffn_hidden=5632,
                     max_seq_len=4096, tie_embeddings=False)


# --8<-- [start:build]
def build(cfg: GPTConfig, device, dtype) -> GPT:
    # 全局默认 dtype：之后新建的浮点张量（包括 nn.Linear 的权重）直接是 FP16/BF16
    torch.set_default_dtype(dtype)
    # with torch.device(...)：块内新建的张量直接分配在目标设备上，不经过 CPU
    # 对比"CPU 上 FP32 建好再 .to(device, dtype)"：8B 模型要先占 32 GB 内存、再拷一次
    with torch.device(device):
        # .eval() 切到推理模式（关 dropout 等）；它不关梯度，关梯度靠 main 上的 no_grad
        model = GPT(cfg).eval()
    # 恢复默认，避免影响后面创建的 FP32 张量（比如统计用的计时数组）
    torch.set_default_dtype(torch.float32)
    return model
# --8<-- [end:build]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="mac_demo", choices=["mac_demo", *PRESETS])
    p.add_argument("--hw", default="M4Pro", choices=sorted(HARDWARE))
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--prefill-lens", nargs="+", type=int, default=[128, 512, 2048])
    args = p.parse_args()

    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    cfg = MAC_DEMO if args.preset == "mac_demo" else PRESETS[args.preset]
    shape = ModelShape.from_gpt_config(args.preset, cfg)
    hw = HARDWARE[args.hw]
    model = build(cfg, device, dtype)
    print(f"{describe(device)}；{args.preset}：{shape.total_params / 1e9:.2f}B 参数，"
          f"KV {shape.kv_bytes_per_token() / 1024:.0f} KB/token；roofline 按 {hw.name}")

    # --8<-- [start:prefill_loop]
    # ---- prefill：B=1，不同 prompt 长度
    pf_rows = []
    for t in args.prefill_lens:
        # 随机 token id 就够了：耗时只取决于形状，不取决于内容
        idx = torch.randint(0, cfg.vocab_size, (1, t), device=device)  # [1, T]
        # 一次完整前向，输出 logits [1, T, V]；没有 cache，就是 prefill
        # 不传 cache 也就不写 KV，比真实 prefill 少写一点，相对权重读取可忽略
        res = benchmark(lambda: model(idx), device, warmup=3, iters=10)
        # 同一形状的理论值：FLOPs 和字节都来自 tools/roofline.py
        est = prefill(shape, 1, t, hw)
        # 用理论 FLOPs 除以实测秒数，得到"实际跑到的算力"；再除以峰值就是 MFU
        # mac_demo、T=512 时约 0.45 TFLOPs / 76 ms ≈ 5.9 TFLOPS
        achieved_tflops = est.flops / (res.median_ms * 1e-3) / 1e12
        # 表格一行：长度、理论 ms、实测 ms、实测 TFLOPS、占峰值比例、预测瓶颈
        # f"{x:.0%}" 把 0.79 格式化成 "79%"；est.bound 形如 "compute-bound（算力瓶颈）"，split("（")[0] 只留英文部分
        pf_rows.append([t, f"{est.time_s * 1e3:.2f}", f"{res.median_ms:.2f}", f"{achieved_tflops:.2f}",
                        f"{achieved_tflops / hw.peak_tflops:.0%}", est.bound.split("（")[0]])
    # --8<-- [end:prefill_loop]

    # --8<-- [start:decode_loop]
    # --8<-- [start:decode_setup]
    # ---- decode：固定 context，扫 batch
    dc_rows, curve = [], []
    # 先用显存公式算出最多能放几条 context 长度的请求，超出的 batch 直接跳过，免得 OOM
    limit = max_batch(shape, args.context, hw)
    for b in args.batches:
        if b > limit:
            print(f"B={b} 超出显存上限（约 {limit}），跳过")
            continue
        # 预分配 [L, B, Hkv, S_max, D] 的 K、V 两块；S_max 多留 1 个位置给本步新 token
        # 内容是 0 不要紧：attention 读多少字节只看长度，不看数值
        # mac_demo 每 token 16 KB，B=64、context=1024 时共 1.07 GB
        cache = KVCache(cfg, b, args.context + 1, device, dtype)
        # 每条请求只喂 1 个新 token，线性层都退化成 [B, C] @ [C, N] 的瘦矩阵乘
        tok = torch.randint(0, cfg.vocab_size, (b, 1), device=device)  # [B, 1]
    # --8<-- [end:decode_setup]

    # --8<-- [start:decode_step]
        # 闭包 step 捕获外层的 cache 和 tok，benchmark 反复调用它
        def step():
            # 前向会把 length 加 1；每次调用前重置，保证测的永远是"第 context+1 个位置"这同一步
            # 如果不重置，length 会一路涨过 S_max 而越界
            cache.length = args.context  # 每次都从"已有 context 个历史"开始，只测 1 步 decode
            # 每层：写 1 个新 K/V，再读全部 context+1 个 K/V 做 attention
            model(tok, cache)

        # decode 一步只有几 ms，iters 比 prefill 多几次，中位数更稳
        res = benchmark(step, device, warmup=3, iters=15)
    # --8<-- [end:decode_step]

    # --8<-- [start:decode_metrics]
        est = decode(shape, b, args.context, hw)
        # MBU = roofline 认为必须读的字节 / 实测秒数 / 峰值带宽，评价 decode 引擎的核心指标
        # 例：B=1 时约 1.0 GB / 6.28 ms ≈ 159 GB/s，占 273 GB/s 的 58%
        mbu = est.bytes / (res.median_ms * 1e-3) / (hw.bandwidth_gbs * 1e9)
        # 吞吐 = 每步生成 B 个 token ÷ 每步秒数
        dc_rows.append([b, f"{est.time_s * 1e3:.2f}", f"{res.median_ms:.2f}", f"{b / (res.median_ms * 1e-3):,.0f}",
                        f"{est.intensity:.1f}", f"{mbu:.0%}"])
        # 画图用：(batch, 理论吞吐, 实测吞吐)
        curve.append((b, b / est.time_s, b / (res.median_ms * 1e-3)))
        # 及时释放这一档的 KV，下一档 batch 更大，避免两份同时占显存
        del cache
    # --8<-- [end:decode_metrics]
    # --8<-- [end:decode_loop]

    pf_table = tabulate(pf_rows, headers=["prompt 长度", "roofline ms", "实测 ms", "实测 TFLOPS", "占峰值", "预测瓶颈"], tablefmt="github")
    dc_table = tabulate(dc_rows, headers=["batch", "roofline ms/步", "实测 ms/步", "实测 tok/s", "算术强度", "MBU"], tablefmt="github")
    tag = f"{args.preset}_{args.hw}"
    report = f"""# decode/prefill benchmark：{tag}

- 设备：{describe(device)}，dtype={dtype}
- 模型：{args.preset}，{shape.total_params / 1e9:.2f}B 参数，权重 {shape.total_params * 2 / 1e9:.2f} GB，KV {shape.kv_bytes_per_token() / 1024:.0f} KB/token
- roofline 硬件参数：{hw.name}，{hw.peak_tflops} TFLOPS，{hw.bandwidth_gbs} GB/s，拐点 {hw.ridge_point:.0f} FLOP/B

## prefill（batch=1）

{pf_table}

## decode（context={args.context}）

{dc_table}

MBU（Model Bandwidth Utilization）= roofline 认为必须读的字节数 / 实测耗时 / 峰值带宽。decode 优化得好的引擎能到 70-85%。
"""
    print(report)
    EXP.mkdir(exist_ok=True)
    (EXP / f"decode_bench_{tag}.md").write_text(report)

    bs, pred, meas = zip(*curve)
    plt.figure(figsize=(6, 4))
    plt.loglog(bs, pred, "o--", label="roofline")
    plt.loglog(bs, meas, "s-", label="measured")
    plt.xlabel("batch size")
    plt.ylabel("decode throughput (tok/s)")
    plt.title(f"{tag}, context={args.context}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(EXP / f"decode_bench_{tag}.png", dpi=120)


if __name__ == "__main__":
    main()
