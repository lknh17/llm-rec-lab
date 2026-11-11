"""仅权重量化（weight-only quantization）：INT8 按输出通道、INT4 按组（group-wise）。

decode 是访存受限：每步要把全部权重读一遍，耗时 ≈ 权重字节数 / 带宽。
权重从 BF16(2B) 压到 INT8(1B) / INT4(0.5B)，decode 理论提速约 2x / 3.5x（扣掉 scale 的开销）。
计算仍然是 BF16：kernel 读入低比特权重，在寄存器里反量化后做乘法 —— 省的是带宽，不是算力。
所以 prefill（算力受限）用 weight-only 量化几乎不提速，要提速得用 W8A8（FP8/INT8 激活也量化，走低精度 Tensor Core）。

这里是"模拟量化"：存 int8 张量 + scale，前向时反量化成浮点再乘。数值和真实 kernel 一致，速度不代表真实 kernel。

    uv run python -m stages.s05_inference.code.quant
"""

# 上面 docstring 里 INT4 的 "3.5x" 是保守估计；按 g=128 的有效位宽 4.125 bit 算，理论上限是 16 / 4.125 ≈ 3.9x（第 3 课 2.4 节）

from pathlib import Path

import torch
from tabulate import tabulate
from torch import nn

EXP = Path(__file__).resolve().parents[1] / "experiments"


# --8<-- [start:quantize]
# --8<-- [start:quantize_fn]
def quantize(w: torch.Tensor, bits: int, group_size: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """对称量化。w: [out, in]。group_size=None 表示每个输出通道一个 scale。
    返回 q: [out, in]（int8 存储，INT4 的取值范围是 [-8, 7]），scale: [out, n_groups, 1]"""
    # nn.Linear 的权重是 [out, in]：每一行是一个输出通道，和输入 x 的 in 维做点积
    out_f, in_f = w.shape
    # group_size=None 即按通道：整行 in 个权重是一组
    g = group_size or in_f
    # 每行必须能整切成若干组；d_model=384 时 g=256 就切不开（第 6 步 A）
    assert in_f % g == 0
    # 对称量化正半轴的最大整数 2^(bits-1) - 1
    qmax = 2 ** (bits - 1) - 1  # INT8: 127，INT4: 7
    # .float() 先转 FP32 再算，避免 BF16 下求最大值、相除带来额外舍入
    # reshape 只换形状不搬数据：每行 in 个权重切成 in/g 组，每组是 g 个连续的输入通道
    wg = w.float().reshape(out_f, in_f // g, g)  # [out, n_groups, g]
    # 每组一个步长：abs().amax(dim=-1) 沿最后一维（组内 g 个数）取最大绝对值
    # keepdim=True 保留被归约的那一维（长度变成 1），后面才能和 wg 广播相除
    # clamp(min=1e-8) 防止全零的组除以 0；每组的最大值映射到 qmax
    # 一个离群值只会撑大它所在那一组的步长，这就是分组比按通道准的原因
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax  # [out, n_groups, 1]
    # 除以步长 -> 四舍五入（RTN）-> 截断到整数范围；最大误差是半个步长 scale/2
    # 下界 -qmax-1（INT4 是 -8）是补码多出的那个负值；对称量化下 |w/scale| <= qmax，实际用不到它
    # INT4 也先存成 int8（一个字节一个值），真实部署两两打包，存储按 storage_bytes 计算
    q = (wg / scale).round().clamp(-qmax - 1, qmax).to(torch.int8)  # [out, n_groups, g]
    # q 拉平回 [out, in]；scale 保持 [out, n_groups, 1]
    # 7B 的 [3584, 18944] 矩阵 g=128 时：INT4 权重 34 MB，FP16 scale 约 1 MB（3%）
    return q.reshape(out_f, in_f), scale
# --8<-- [end:quantize_fn]


# --8<-- [start:dequantize_fn]
def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_f, in_f = q.shape
    # 组数从 scale 的第 1 维读出，不需要另外传 group_size
    n_groups = scale.size(1)
    # q 转浮点后按组展开成 [out, n_groups, g]（-1 让 PyTorch 自动推算 g = in / n_groups）
    # 乘 [out, n_groups, 1] 的 scale：广播到组内每个数，再拉平回 [out, in]
    return (q.float().reshape(out_f, n_groups, -1) * scale).reshape(out_f, in_f)
# --8<-- [end:dequantize_fn]
# --8<-- [end:quantize]


# --8<-- [start:quant_linear]
# --8<-- [start:quant_linear_init]
class QuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int, group_size: int | None):
        super().__init__()
        # 这个模型的 Linear 都不带偏置；带偏置的话偏置保持浮点即可，这里省掉
        assert linear.bias is None
        # .data 取出不带梯度追踪的原始权重；量化离线做一次，之后只存结果
        q, scale = quantize(linear.weight.data, bits, group_size)
        # register_buffer：把张量登记成模块状态（随 .to(device) 移动、进 state_dict），但不是可训练参数
        # 原来的浮点权重不再保留，模块里只有整数和 scale
        self.register_buffer("q", q)  # [out, in] int8
        # scale 存成原权重的 dtype；storage_bytes 里按 FP16 每个 2 字节计
        self.register_buffer("scale", scale.to(linear.weight.dtype))  # [out, n_groups, 1]
        self.bits, self.group_size = bits, group_size
# --8<-- [end:quant_linear_init]

# --8<-- [start:quant_linear_forward]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每次前向都反量化出完整的 [out, in] 浮点权重：这就是模拟量化（fake quant），数值对、速度不对
        # 真实 kernel 在寄存器里做这一步
        w = dequantize(self.q, self.scale.float()).to(x.dtype)  # [out, in]
        # x: [..., in] @ [in, out] -> [..., out]，和 nn.Linear 的 x @ W^T 相同
        return x @ w.T

    def storage_bytes(self) -> int:
        """真实部署的存储：INT4 两个值打包进 1 字节；scale 用 FP16。"""
        # 权重 numel·bits/8 字节，scale 每个 2 字节
        # 例：[1024, 384] 的 INT4 g32 = 196608 + 12288·2 字节 ≈ 0.21 MB，有效位宽 4.5 bit
        return self.q.numel() * self.bits // 8 + self.scale.numel() * 2
# --8<-- [end:quant_linear_forward]
# --8<-- [end:quant_linear]


def quantize_model(model: nn.Module, bits: int, group_size: int | None) -> nn.Module:
    """把 Transformer block 里的所有 Linear 换成 QuantLinear（embedding / lm_head 保持原精度）。"""
    # 只遍历 model.blocks，所以 embedding 和 lm_head 天然不在替换范围内
    for block in model.blocks:
        # modules() 递归列出 block 及其所有子模块；要替换子模块，得拿到它的父模块和属性名
        for parent in block.modules():
            # 先 list() 固定下来：循环里会 setattr 改子模块，边遍历边改会出错
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    # setattr 用同名属性替换，原模型的 forward 代码不用改一行
                    setattr(parent, name, QuantLinear(child, bits, group_size))
    return model


def linear_bytes(model: nn.Module, base_bytes: int = 2) -> int:
    total = 0
    for mod in model.blocks.modules():
        if isinstance(mod, QuantLinear):
            total += mod.storage_bytes()
        elif isinstance(mod, nn.Linear):
            total += mod.weight.numel() * base_bytes
    return total


def main() -> None:
    import copy

    from stages.s01_transformer.code.train import evaluate, load_corpus
    from stages.s05_inference.code.common import load_code_gpt
    from tools.device import get_device
    from tools.roofline import HARDWARE

    device = get_device()
    base = load_code_gpt(device)
    _, val = load_corpus()
    base_loss = evaluate(base, val, batch=16, seq_len=256, device=device, iters=20)
    base_bytes = linear_bytes(base)

    rows = [["未量化（按 BF16 计字节）", "-", f"{base_bytes / 2**20:.2f}", "1.00x", f"{base_loss:.4f}", "-"]]
    configs = [("INT8 按通道", 8, None), ("INT4 按通道", 4, None), ("INT4 group=128", 4, 128), ("INT4 group=32", 4, 32)]
    for name, bits, g in configs:
        m = quantize_model(copy.deepcopy(base), bits, g)
        loss = evaluate(m, val, batch=16, seq_len=256, device=device, iters=20)
        b = linear_bytes(m)
        rows.append([name, g or "通道", f"{b / 2**20:.2f}", f"{base_bytes / b:.2f}x", f"{loss:.4f}", f"{loss - base_loss:+.4f}"])

    # 真实模型的 decode 收益（访存受限，batch=1，只算权重读取）
    hw = HARDWARE["H20"]
    n = 7.6e9  # Qwen2.5-7B
    est = []
    for name, bpp in [("BF16", 2), ("INT8", 1 + 2 / 4096), ("INT4 g128", 0.5 + 2 / 128)]:
        t = n * bpp / (hw.bandwidth_gbs * 1e9) * 1e3
        est.append([name, f"{bpp:.3f}", f"{n * bpp / 1e9:.1f}", f"{t:.2f}"])

    text = "# 仅权重量化实验\n\n"
    text += "## code_gpt（9.5M，验证集 loss 单位 nats/byte）\n\n"
    text += tabulate(rows, ["方案", "group", "Linear 权重 MB", "压缩比", "val loss", "Δloss"], tablefmt="github")
    text += ("\n\n小模型对量化更敏感（每个权重承载的信息更多）；7B 以上模型 INT8 几乎无损，INT4 g128（GPTQ/AWQ）通常掉点 <1%。\n")
    text += f"\n## Qwen2.5-7B 在 {hw.name} 上 batch=1 decode 的理论下限（{hw.bandwidth_gbs:.0f} GB/s，只算权重读取）\n\n"
    text += tabulate(est, ["精度", "字节/参数（含 scale）", "权重 GB", "每 token ms"], tablefmt="github")
    text += "\n\nbatch 增大后 decode 逐渐变成算力受限，weight-only 量化的收益随之消失（还多了反量化开销）——这时要上 FP8 W8A8。\n"
    EXP.mkdir(exist_ok=True)
    (EXP / "quant.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
