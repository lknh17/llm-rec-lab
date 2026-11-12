"""S3 第 8 课：W4A16 量化 GEMM 的参考实现、字节/算力模型和 Mac 实验。

1. pack_int4 / unpack_int4：两个 4 bit 权重放进一个 uint8，布局和 triton_kernels/w4a16_gemm.py 一致。
2. w4a16_matmul_ref：先整体反量化再 GEMM（数值标准答案，但多一遍 FP16 权重的写和读）。
   w4a16_matmul_blocked：按 K 分块"读 int4 -> 解包 -> 乘 scale -> 累加"，和 Triton kernel 主循环一一对应。
3. w4a16_model：给定 M、N、K 和硬件，估算 FP16 GEMM 与 W4A16 GEMM 的耗时和提速，找出提速消失的 M。
4. fp8_quant：FP8（E4M3/E5M2）的 per-tensor / per-token / per-channel / per-block scaling 误差对比。
5. main()：Mac 实测误差、打包正确性和"先反量化再 GEMM"的代价，写入 experiments/w4a16.md。

量化本身（对称、按组 scale）直接复用 S5 的 stages/s05_inference/code/quant.py。
记号：x [M, K] 是激活，W [N, K] 是 nn.Linear 的权重（out × in），y = x @ W^T [M, N]，g = group_size。

运行（会测耗时，要加锁）：
    uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.code.w4a16_ref
"""

import math
from pathlib import Path

import torch
from tabulate import tabulate

from stages.s05_inference.code.quant import dequantize, quantize
from tools.roofline import HARDWARE, Hardware

OUT = Path(__file__).resolve().parents[1] / "experiments" / "w4a16.md"


# --8<-- [start:pack]
def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """q: [N, K] int8，取值 [-8, 7]（quantize(w, bits=4, ...) 的输出）-> packed: [K/2, N] uint8。

    布局按 GEMM 的 B 操作数 W^T [K, N] 来排：沿 K 每两个相邻的 4 bit 放进一个字节，
    低 4 位是偶数行 k=2j，高 4 位是奇数行 k=2j+1。N 维连续，kernel 读一行时访存可以合并。
    """
    n, k = q.shape
    assert k % 2 == 0
    # +8 把有符号的 [-8, 7] 平移成无符号的 [0, 15]，这样一个 4 bit 半字节就能装下，不用处理符号位
    # .T 转成 [K, N]；uint8 的位运算不会有符号扩展的问题
    u = (q.to(torch.int16) + 8).to(torch.uint8).T  # [K, N]
    # u[0::2] 是偶数行 [K/2, N]，u[1::2] 是奇数行 [K/2, N]；奇数行左移 4 位放进高半字节
    return (u[0::2] | (u[1::2] << 4)).contiguous()  # [K/2, N]


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """packed: [K/2, N] uint8 -> q: [N, K] int8，pack_int4 的逆。"""
    # & 0xF 取低 4 位（偶数行），>> 4 取高 4 位（奇数行）
    lo = packed & 0xF  # [K/2, N]
    hi = packed >> 4  # [K/2, N]
    # stack 成 [K/2, 2, N] 再展平成 [K, N]：行顺序恢复为 0, 1, 2, 3, ...
    u = torch.stack([lo, hi], dim=1).reshape(-1, packed.size(1))  # [K, N]
    # 减 8 回到有符号，再转回 [N, K]
    return (u.to(torch.int16) - 8).to(torch.int8).T.contiguous()  # [N, K]


def scales_for_kernel(scale: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
    """quant.py 的 scale [N, n_groups, 1] -> kernel 用的 [n_groups, N]：同一组的 N 个 scale 连续，按行读一次。"""
    return scale.squeeze(-1).T.to(dtype).contiguous()  # [n_groups, N]
# --8<-- [end:pack]


# --8<-- [start:ref_gemm]
def w4a16_matmul_ref(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """先把整个权重反量化成 FP16，再做一次普通 GEMM。x: [M, K] -> [M, N]。

    数值上是标准答案；但性能上是反例：多写一遍、多读一遍 2·K·N 字节的 FP16 权重，
    decode 时（M 很小）几乎把 W4 省下的带宽全部还回去。
    """
    k = packed.size(0) * 2
    # 解包成 [K, N] 的有符号整数（这里在显存里真的造出一个 [K, N] 张量）
    qt = unpack_int4(packed).T  # [K, N]
    # 第 k 行属于第 k // g 组；repeat_interleave 把 [n_groups, N] 的 scale 展开成 [K, N]
    s = scales.repeat_interleave(group_size, dim=0)[:k]  # [K, N]
    # 反量化：w = q · scale。这一步写出完整的 FP16 权重，K=N=4096 时是 32 MB
    wt = (qt.to(scales.dtype) * s).to(x.dtype)  # [K, N]
    return x @ wt  # [M, N]
# --8<-- [end:ref_gemm]


# --8<-- [start:blocked_gemm]
def w4a16_matmul_blocked(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group_size: int,
                         block_k: int = 128) -> torch.Tensor:
    """按 K 分块的融合写法（Triton kernel 主循环的 PyTorch 版）。x: [M, K] -> [M, N]。

    每次只解包、反量化一个 [BK, N] 的小块，马上和 x 的 [M, BK] 相乘累加；
    在 kernel 里这个小块只存在于寄存器，FP16 权重从不写回显存。
    """
    m = x.size(0)
    k, n = packed.size(0) * 2, packed.size(1)
    # FP32 累加器，和 Tensor Core 的 FP32 累加对应
    acc = torch.zeros(m, n, dtype=torch.float32, device=x.device)  # [M, N]
    for k0 in range(0, k, block_k):
        # 读 int4：BK 行权重只占 BK/2 行字节
        blk = packed[k0 // 2 : (k0 + block_k) // 2]  # [BK/2, N] uint8
        # 解包：低、高半字节交错成 BK 行，减 8 回到有符号
        u = torch.stack([blk & 0xF, blk >> 4], dim=1).reshape(-1, n)  # [BK, N]
        q = u.to(torch.float32) - 8.0  # [BK, N]
        # 乘 scale：本块每一行所在的组号是 (k0 + r) // g
        rows = torch.arange(k0, k0 + block_k, device=x.device) // group_size  # [BK]
        w = q * scales[rows].float()  # [BK, N]
        # 喂给矩阵乘：[M, BK] @ [BK, N]，累加到 FP32
        acc += x[:, k0 : k0 + block_k].float() @ w
    return acc.to(x.dtype)  # [M, N]
# --8<-- [end:blocked_gemm]


# --8<-- [start:model]
# CUDA Core（非 Tensor Core）FP32 吞吐，TFLOPS（厂商标称）。反量化的移位、与、转换、乘 scale 都在这里做
CUDA_CORE_TFLOPS = {"H100": 67.0, "H20": 44.0, "A100": 19.5}


def w4a16_model(m: int, n: int, k: int, hw_name: str, group_size: int = 128, dequant_ops: int = 4) -> dict:
    """一次 [M, K] @ [K, N] 的 FP16 GEMM 和 W4A16 GEMM 的理想耗时（μs）。"""
    hw: Hardware = HARDWARE[hw_name]
    flops = 2 * m * n * k
    t_mma = flops / (hw.peak_tflops * 1e12)
    # FP16：权重 2 字节/个；激活 x 读一遍、输出 y 写一遍
    bytes_fp16 = 2 * k * n + 2 * m * k + 2 * m * n
    # W4A16：权重 0.5 字节/个，加上每 g 个权重一个 FP16 scale（g=128 时每个权重摊 0.0156 字节）
    bytes_w4 = 0.5 * k * n + 2 * (k // group_size) * n + 2 * m * k + 2 * m * n
    # 反量化次数：每个 M 方向的块（BM 行）都要把整块权重解包一次。BM 按 kernel 常用的 16..128 取
    bm = min(128, max(16, 2 ** math.ceil(math.log2(max(m, 1)))))
    deq = dequant_ops * k * n * math.ceil(m / bm)
    t_deq = deq / (CUDA_CORE_TFLOPS[hw_name] * 1e12)
    # 访存和计算理想重叠取 max；反量化和 MMA 抢同一批 warp 的发射槽，按串行算（偏保守）
    t16 = max(t_mma, bytes_fp16 / (hw.bandwidth_gbs * 1e9))
    t4 = max(t_mma + t_deq, bytes_w4 / (hw.bandwidth_gbs * 1e9))
    return {"t_fp16_us": t16 * 1e6, "t_w4_us": t4 * 1e6, "speedup": t16 / t4,
            "intensity_fp16": flops / bytes_fp16, "intensity_w4": flops / bytes_w4}
# --8<-- [end:model]


# --8<-- [start:fp8]
FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}


def fp8_quant(x: torch.Tensor, mode: str, dtype=torch.float8_e4m3fn, block: int = 128) -> torch.Tensor:
    """把 x [R, C] 量化成 FP8 再反量化回 FP32（模拟），返回反量化后的张量，用来量误差。

    mode：tensor（整个张量一个 scale）、row（每行一个：激活的 per-token / 权重的 per-output-channel）、
    block1d（每行每 128 列一个：DeepSeek-V3 激活的 1x128）、block2d（每 128x128 一个：DeepSeek-V3 权重）。
    """
    r, c = x.shape
    x = x.float()
    if mode == "tensor":
        amax = x.abs().amax().expand(r, c)
    elif mode == "row":
        # keepdim=True 保留被归约的维度为 1：[R, 1]，再广播回 [R, C]
        amax = x.abs().amax(dim=1, keepdim=True).expand(r, c)
    elif mode == "block1d":
        # [R, C] -> [R, C/128, 128]，每个 128 段求最大值，再展回 [R, C]
        amax = x.view(r, c // block, block).abs().amax(-1, keepdim=True).expand(r, c // block, block).reshape(r, c)
    elif mode == "block2d":
        # [R/128, 128, C/128, 128]：第 1、3 维是块内坐标，对它们一起求最大值
        xb = x.view(r // block, block, c // block, block)
        amax = xb.abs().amax(dim=(1, 3), keepdim=True).expand_as(xb).reshape(r, c)
    else:
        raise ValueError(mode)
    # scale 让块内最大值正好映射到 FP8 的最大可表示值（E4M3 是 448）
    scale = amax.clamp(min=1e-12) / FP8_MAX[dtype]
    # .to(float8) 做舍入；再乘回 scale 得到"反量化后"的值
    return (x / scale).to(dtype).float() * scale
# --8<-- [end:fp8]


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """相对误差 ||a - b|| / ||b||（Frobenius 范数）。"""
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def heavy_tailed_weight(n: int, k: int, seed: int = 0) -> torch.Tensor:
    """模拟真实 LLM 权重：大部分接近高斯，少数元素很大（Student-t 分布，自由度 4）。"""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, k, generator=g)
    # 自由度 4 的卡方变量 = 4 个标准正态的平方和；t = z / sqrt(chi² / 4)
    chi = (torch.randn(4, n, k, generator=g) ** 2).sum(0)
    return 0.02 * z / torch.sqrt(chi / 4)


def outlier_activation(m: int, k: int, n_outlier: int = 8, mag: float = 50.0, seed: int = 0) -> torch.Tensor:
    """模拟 LLM 激活：少数几个通道（列）系统性地大几十倍（LLM.int8() 论文观察到的现象）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g)
    cols = torch.randperm(k, generator=g)[:n_outlier]
    x[:, cols] *= mag
    return x


def main() -> None:
    from tools.bench import benchmark
    from tools.device import describe, get_device

    device = get_device()
    torch.manual_seed(0)
    n = k = 4096

    # ---- 误差：不同 group size 的 INT4，和 INT8 按通道
    w = heavy_tailed_weight(n, k)
    x = torch.randn(16, k)
    y_ref = x @ w.T
    rows_err = []
    for name, bits, g in [("INT8 按通道", 8, None), ("INT4 按通道", 4, None), ("INT4 g=256", 4, 256),
                          ("INT4 g=128", 4, 128), ("INT4 g=64", 4, 64), ("INT4 g=32", 4, 32)]:
        q, s = quantize(w, bits, g)
        w_hat = dequantize(q, s)
        bpp = bits / 8 + 2 / (g or k)
        rows_err.append([name, f"{bpp:.4f}", f"{2 / bpp:.2f}x", f"{rel_err(w_hat, w):.4f}",
                         f"{rel_err(x @ w_hat.T, y_ref):.4f}"])

    # ---- 打包正确性 + 分块融合写法和"先反量化"写法一致
    q, s = quantize(w, 4, 128)
    packed = pack_int4(q)
    sk = scales_for_kernel(s, torch.float32)
    roundtrip = torch.equal(unpack_int4(packed), q)
    y_a = w4a16_matmul_ref(x, packed, sk, 128)
    y_b = w4a16_matmul_blocked(x, packed, sk, 128)
    y_c = x @ dequantize(q, s).T
    pack_info = (f"- 打包后字节：{packed.numel() / 2**20:.1f} MB（INT8 存储 {q.numel() / 2**20:.1f} MB，"
                 f"FP16 {2 * w.numel() / 2**20:.1f} MB）\n"
                 f"- unpack(pack(q)) == q：{roundtrip}\n"
                 f"- 先反量化 vs quant.py 的 dequantize：最大差 {(y_a - y_c).abs().max().item():.2e}\n"
                 f"- 分块融合 vs 先反量化：最大差 {(y_b - y_a).abs().max().item():.2e}\n")

    # ---- FP8 scaling 粒度：离群通道放大 50 倍（常见）、2 万倍、10 万倍（逼近 E4M3 的动态范围）
    wf = heavy_tailed_weight(n, k, seed=1)
    rows_fp8 = []
    for mag in [50.0, 20000.0, 100000.0]:
        xa = outlier_activation(256, k, mag=mag)
        # 只看普通通道的误差：离群通道自己总是量化得很准，被"挤坏"的是其余 4088 个普通通道
        normal = xa.abs().mean(0) < 5  # [K] 布尔
        yf = xa @ wf.T
        for name, xm, wm, dt in [("E4M3 per-tensor / per-tensor", "tensor", "tensor", torch.float8_e4m3fn),
                                 ("E5M2 per-tensor / per-tensor", "tensor", "tensor", torch.float8_e5m2),
                                 ("E4M3 per-token / per-channel", "row", "row", torch.float8_e4m3fn),
                                 ("E4M3 1x128 / 128x128（DeepSeek-V3）", "block1d", "block2d", torch.float8_e4m3fn)]:
            xq, wq = fp8_quant(xa, xm, dt), fp8_quant(wf, wm, dt)
            rows_fp8.append([f"x{mag:g}", name, f"{rel_err(xq[:, normal], xa[:, normal]):.4f}",
                             f"{rel_err(wq, wf):.4f}", f"{rel_err(xq @ wq.T, yf):.4f}"])

    # ---- 字节/算力模型：Qwen2.5-7B 的 gate_up 投影（K=3584，N=2·18944），扫 M
    rows_model = []
    for m in [1, 4, 16, 32, 64, 128, 256, 512]:
        row = [m]
        for hw in ["H100", "H20"]:
            r = w4a16_model(m, 2 * 18944, 3584, hw)
            row += [f"{r['t_fp16_us']:.1f}", f"{r['t_w4_us']:.1f}", f"{r['speedup']:.2f}x"]
        rows_model.append(row)

    # ---- Mac 实测：先反量化再 GEMM 的代价
    rows_mac = []
    wd = w.to(device, torch.float16)
    wt = wd.T.contiguous()  # [K, N]
    pk, sk16 = packed.to(device), scales_for_kernel(s).to(device)
    for m in [1, 16]:
        xm = torch.randn(m, k, device=device, dtype=torch.float16)
        t_fp16 = benchmark(lambda: xm @ wt, device, warmup=5, iters=30).median_ms
        t_deq = benchmark(lambda: w4a16_matmul_ref(xm[:0], pk, sk16, 128), device, warmup=5, iters=30).median_ms
        t_ref = benchmark(lambda: w4a16_matmul_ref(xm, pk, sk16, 128), device, warmup=5, iters=30).median_ms
        t_blk = benchmark(lambda: w4a16_matmul_blocked(xm, pk, sk16, 128, 512), device, warmup=3, iters=10).median_ms
        rows_mac.append([m, f"{t_fp16:.3f}", f"{t_deq:.3f}", f"{t_ref:.3f}", f"{t_ref / t_fp16:.1f}x",
                         f"{t_blk:.3f}", f"{32 / t_fp16:.0f}"])

    report = (
        "# W4A16 量化 GEMM 实验\n\n"
        "## 误差：权重 [4096, 4096]（Student-t 重尾，模拟真实权重），x [16, 4096]\n\n"
        + tabulate(rows_err, headers=["方案", "字节/权重（含 FP16 scale）", "相对 FP16 压缩", "权重相对误差",
                                      "输出相对误差"], tablefmt="github")
        + "\n\n## 打包与两种 GEMM 写法的一致性（INT4 g=128）\n\n" + pack_info
        + "\n## FP8 scaling 粒度：激活 [256, 4096]（8 个离群通道），权重 [4096, 4096]\n\n"
        + tabulate(rows_fp8, headers=["离群通道放大", "方案（激活 / 权重）", "普通通道激活相对误差", "权重相对误差",
                                      "输出相对误差"], tablefmt="github")
        + "\n\n## 字节/算力模型：Qwen2.5-7B gate_up（K=3584，N=37888），INT4 g=128（预期，未实测，需要 GPU）\n\n"
        + "假设：访存与 MMA 理想重叠；反量化每个权重 4 次 CUDA Core 运算、每个 BM 行块做一次，和 MMA 串行。\n\n"
        + tabulate(rows_model, headers=["M", "H100 FP16 μs", "H100 W4 μs", "H100 提速",
                                        "H20 FP16 μs", "H20 W4 μs", "H20 提速"], tablefmt="github")
        + f"\n\n## Mac 实测（{describe(device)}，N=K=4096，FP16 权重 32 MB）\n\n"
        + tabulate(rows_mac, headers=["M", "FP16 matmul ms", "只反量化 ms", "先反量化再 GEMM ms", "相对 FP16",
                                      "分块融合（PyTorch 循环，BK=512）ms", "FP16 有效 GB/s"], tablefmt="github")
        + "\n"
    )
    print(report)
    OUT.write_text(report)
    print(f"已保存到 {OUT}")


if __name__ == "__main__":
    main()
