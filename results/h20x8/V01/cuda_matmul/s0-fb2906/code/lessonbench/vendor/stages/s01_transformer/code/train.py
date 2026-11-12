"""S1 第 4 课：训练一个字节级的小 GPT。

语料：本机 Python 标准库的源码（离线、确定、约 20MB）。模型学会"写 Python 代码"，生成结果肉眼可判断好坏。
产物：
- data/checkpoints/code_gpt.pt：后续课程（采样、KV cache、量化、推理引擎）都用这个模型
- experiments/train_log.csv、experiments/loss_curve.png、experiments/train_report.md

运行：uv run python -m stages.s01_transformer.code.train [--steps 1000]（M4 Pro 上约 4 分钟）
"""

import argparse
import csv
import sysconfig
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from stages.s01_transformer.code.model import GPT, GPTConfig, count_params
from tools.device import describe, get_device, synchronize

ROOT = Path(__file__).resolve().parents[3]
CKPT = ROOT / "data" / "checkpoints" / "code_gpt.pt"
EXP = Path(__file__).resolve().parents[1] / "experiments"

MODEL_CFG = GPTConfig(vocab_size=256, n_layer=6, n_head=6, n_kv_head=2, d_model=384, max_seq_len=512)


# --8<-- [start:data]
# --8<-- [start:data_load]
def load_corpus(max_bytes: int = 20_000_000) -> tuple[torch.Tensor, torch.Tensor]:
    """读标准库源码，按文件切 95/5 的训练/验证集（按文件切，避免同一文件同时出现在两边造成泄漏）。"""
    # 当前 Python 解释器自带的标准库目录；sorted 保证每台机器、每次运行的文件顺序一致
    files = sorted(Path(sysconfig.get_paths()["stdlib"]).rglob("*.py"))
    # chunks：每个文件一段 bytes；total：累计字节数，到 max_bytes 就停
    chunks, total = [], 0
    for f in files:
        # 跳过测试代码（大量重复的断言样板）和第三方包，只留"正常的库代码"
        if "test" in f.parts or "site-packages" in f.parts:
            continue
        b = f.read_bytes()
        chunks.append(b)
        total += len(b)
        if total > max_bytes:
            break
    # 切分点按"文件个数"算，不按字节：前 95% 的文件训练，后 5% 验证，同一个文件不会两边都有
    split = int(len(chunks) * 0.95)
    # 文件之间用换行拼起来，再把整段 bytes 直接当成 token 序列：一个字节就是一个 token id（0-255）
    # torch.frombuffer 不拷贝，把 bytearray 当成 uint8 张量；.long() 转 int64（8 倍内存，约 160 MB）
    # 必须是 int64：nn.Embedding 的下标和 cross_entropy 的标签都要求 long 类型
    to_tensor = lambda cs: torch.frombuffer(bytearray(b"\n".join(cs)), dtype=torch.uint8).long()
    # 两个一维张量：训练集约 19M 个 token，验证集约 1M 个
    return to_tensor(chunks[:split]), to_tensor(chunks[split:])
# --8<-- [end:data_load]


# --8<-- [start:data_batch]
def get_batch(data: torch.Tensor, batch: int, seq_len: int, device, gen: torch.Generator):
    # 随机抽 B 个起点；上界减 seq_len+1 是给 y 多留 1 个字节，保证 y 的最后一位不越界
    # generator=gen：用独立的随机数生成器，抽到的批次可复现，且不受模型初始化等其他随机调用的影响
    ix = torch.randint(len(data) - seq_len - 1, (batch,), generator=gen)  # [B]
    # 每个起点切一段长 T 的窗口，stack 把 B 个 [T] 叠成 [B, T]
    x = torch.stack([data[i:i + seq_len] for i in ix])  # [B, T]
    # y 和 x 是同一段数据整体右移一位：y[b, t] = x[b, t+1]，即位置 t 的"正确答案"（下一个字节）
    # 一条样本因此同时提供 T 个"预测下一个字节"的训练信号（因果 mask 保证位置 t 看不到 t+1）
    y = torch.stack([data[i + 1:i + 1 + seq_len] for i in ix])  # [B, T]
    # 数据在 CPU 上切好再拷到设备：每步 2 × 32 × 256 × 8 字节 = 128 KB，相对每步 260 ms 可忽略
    return x.to(device), y.to(device)
# --8<-- [end:data_batch]
# --8<-- [end:data]


# --8<-- [start:flops]
def flops_per_token(cfg: GPTConfig, seq_len: int) -> float:
    """训练每个 token 的 FLOPs ≈ 6N（矩阵乘）+ 12·L·T·C（attention 的 QK^T 和 AV，前向+反向）。"""
    # N 只数"参与矩阵乘的参数"。count_params 的 total 含 embedding（V·C），共享权重时先把它减掉
    # 本课模型 tie_embeddings=True；不共享时这一行不减，下一行又加一次，会多算 2·V·C（高估）
    n = count_params(cfg)["total"] - cfg.vocab_size * cfg.d_model * (1 if cfg.tie_embeddings else 0)
    # 再把 lm_head 加回：它的矩阵乘要算（embedding 查表不算）
    # 每个 token 都要做 [C] @ [C, V]，本课 V=256、C=384，约 0.1M 参数
    n += cfg.vocab_size * cfg.d_model
    # 6N：前向 2N + 反向 4N。本课 N≈9.54M，6N≈57.2M
    # 12·L·T·C：每层 QK^T 和 AV 前向 4·T·C、反向 8·T·C；本课 12×6×256×384≈7.1M
    # 合计约 64.3 MFLOPs/token；乘以每步 8192 个 token，一步约 0.53 TFLOP
    return 6 * n + 12 * cfg.n_layer * seq_len * cfg.d_model
# --8<-- [end:flops]


@torch.no_grad()
def evaluate(model, data, batch, seq_len, device, iters=20) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(123)
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch, seq_len, device, gen)
        losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(0)
    train_data, val_data = load_corpus()
    # --8<-- [start:train_loop]
    # --8<-- [start:train_setup]
    # .to(device) 把全部参数搬到 MPS/CUDA 上；9.54M 参数 FP32 约 38 MB
    model = GPT(MODEL_CFG).to(device)
    # AdamW 给每个参数再存两份状态（一阶矩 m、二阶矩 v），FP32 下优化器状态 = 2 × 38 MB
    # betas=(0.9, 0.95)：beta2 比默认 0.999 小，v 只记约最近 20 步，梯度尺度变了能更快跟上
    # weight_decay=0.1：和梯度解耦的权重衰减（AdamW 的 "W"），每步把参数往 0 拉一点
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    # 学习率：线性 warmup 50 步 + 余弦衰减到 10%。warmup 避免初期 Adam 二阶矩估计不准导致的大步长发散
    # LambdaLR 每次 sched.step() 后把学习率设为 lr × lambda(step)，lambda 是两个因子的乘积：
    # 第一个因子 min(1, (s+1)/50)：前 50 步从 0.02 线性涨到 1
    # 第二个因子 0.1 + 0.9 × (1 + cos(π·s/steps)) / 2：从 1 平滑降到 0.1（3.1416 就是 π）
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(s / args.steps * 3.1416)).item()))
    )
    # 训练取批次用的随机数生成器，和评估用的（seed 123）分开，两边互不干扰
    gen = torch.Generator().manual_seed(0)
    # 每步 32 × 256 = 8192 个 token；乘 fpt 就是一步的有效 FLOPs
    tokens_per_step = args.batch * args.seq_len
    fpt = flops_per_token(MODEL_CFG, args.seq_len)
    # --8<-- [end:train_setup]

    print(f"设备 {describe(device)}；参数量 {model.num_params() / 1e6:.1f}M；训练集 {len(train_data) / 1e6:.1f}M 字节")
    log, t_start = [], time.perf_counter()
    # --8<-- [start:train_eval]
    # 多跑 1 次（range 到 steps+1）只是为了在最后一步也做一次评估，然后直接退出
    for step in range(args.steps + 1):
        # 每 100 步在验证集上评估一次：20 个 batch 的平均 loss，固定随机种子，前后可比
        # evaluate 带 @torch.no_grad()，只做前向，代价约为训练一步的 1/3
        # 装饰器写法等价于把整个函数体包在 with torch.no_grad() 里；函数内还会 model.eval()，结束时切回 model.train()
        # 注意：这些评估落在下面的计时区间里，所以第 5 步报的吞吐会被略微低估（估算约 7%）
        if step % 100 == 0:
            val = evaluate(model, val_data, args.batch, args.seq_len, device)
            log.append({"step": step, "val_loss": val})
            print(f"step {step:4d} | val loss {val:.3f}（随机猜测为 ln256 = 5.545）")
        if step == args.steps:
            break
        # --8<-- [end:train_eval]
        # --8<-- [start:train_step]
        # 前向：x [B, T] 的 token id -> 每个位置对下一个字节的打分 [B, T, V] = [32, 256, 256]
        x, y = get_batch(train_data, args.batch, args.seq_len, device, gen)
        logits = model(x)  # [B, T, V]
        # cross_entropy 要求输入 [N, V] 和标签 [N]，所以把 B、T 两维摊平成 N = B·T = 8192
        # flatten(0, 1) 合并第 0、1 维：[B, T, V] -> [B·T, V]；y.flatten() 是 [B·T]
        # 结果是 8192 个位置的平均 -log p(正确字节)，一个标量；初始时约等于 ln 256 = 5.545
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        # PyTorch 的梯度默认是累加的，每步开始前必须清掉
        # set_to_none=True：直接把 .grad 置为 None 而不是写 0，省一次清零 kernel 和对应的显存
        opt.zero_grad(set_to_none=True)
        # 反向：沿计算图求出每个参数的 .grad，约 4N FLOPs/token，是前向的两倍
        loss.backward()
        # 梯度裁剪：防止偶发的大梯度把训练带飞
        # 先求所有参数梯度拼起来的全局 L2 范数，超过 1.0 就整体等比例缩小到 1.0（方向不变）
        # 函数名末尾的 _ 表示原地修改：直接改写各参数的 .grad，不返回新张量
        # 代价：求范数读一遍 9.54M 个梯度，缩放再读写一遍，约 0.1 GB 访存，相对整步可忽略
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # AdamW 更新：每个参数读 p、g、m、v，写 p、m、v，FP32 下约 28 字节/参数，一步约 270 MB 访存
        opt.step()
        # 按 lambda 更新下一步的学习率
        sched.step()
        # --8<-- [end:train_step]
        # --8<-- [start:train_timing]
        # GPU/MPS 是异步执行的：Python 发完 kernel 就往下走，不等它跑完
        # synchronize 阻塞到设备上所有已发出的 kernel 都完成，计时起点才准确
        # 前 20 步含编译/预热，从这里开始计时
        if step == 20:
            synchronize(device)
            t_measure, step_measure = time.perf_counter(), step
        # .item() 把 GPU 上的标量拷回 CPU，本身就会等 GPU 算完，相当于每步一次同步
        # 这里无所谓；大规模训练里每步 .item() 会让 CPU 无法提前发下一步的 kernel，通常隔 N 步才取一次
        log.append({"step": step, "train_loss": loss.item()})
        # --8<-- [end:train_timing]
    # --8<-- [end:train_loop]

    synchronize(device)
    elapsed = time.perf_counter() - t_measure
    steps_timed = args.steps - 1 - step_measure
    tok_s = steps_timed * tokens_per_step / elapsed
    tflops = tok_s * fpt / 1e12

    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": MODEL_CFG, "state_dict": model.state_dict()}, CKPT)

    EXP.mkdir(exist_ok=True)
    with open(EXP / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(log)
    tr = [(r["step"], r["train_loss"]) for r in log if "train_loss" in r]
    va = [(r["step"], r["val_loss"]) for r in log if "val_loss" in r]
    plt.figure(figsize=(7, 4))
    plt.plot(*zip(*tr), alpha=0.5, label="train")
    plt.plot(*zip(*va), "o-", label="val")
    plt.axhline(5.545, ls="--", c="gray", label="random (ln 256)")
    plt.xlabel("step")
    plt.ylabel("cross-entropy (nats/byte)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXP / "loss_curve.png", dpi=120)

    from stages.s01_transformer.code.sampling import generate

    model.eval()
    prompt = torch.tensor(list(b"def fibonacci(n):\n"), device=device)[None]
    sample = bytes(generate(model, prompt, 200, temperature=0.8, top_k=40)[0].tolist()).decode("utf-8", "replace")

    report = f"""# 训练报告

- 设备：{describe(device)}（FP32）
- 模型：{MODEL_CFG}
- 参数量：{model.num_params() / 1e6:.2f}M
- 训练：{args.steps} 步 × {tokens_per_step} token/步 = {args.steps * tokens_per_step / 1e6:.1f}M token
- 最终验证 loss：{va[-1][1]:.3f} nats/byte（= {va[-1][1] / 0.693:.2f} bits/byte）
- 吞吐：{tok_s:,.0f} token/s；每 token {fpt / 1e6:.1f} MFLOPs => 实际 {tflops:.2f} TFLOPS

## 采样（temperature=0.8, top_k=40）

```python
{sample}
```
"""
    (EXP / "train_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
