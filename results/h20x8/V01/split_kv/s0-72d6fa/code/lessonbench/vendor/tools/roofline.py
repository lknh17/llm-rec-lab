"""Roofline 估算器：给定计算量和访存量，估算理论耗时和瓶颈类型。

S0/S1：单个矩阵乘、粗粒度 decode。
S2：整模型的 prefill / decode / KV 预算 / 最大并发，与实测对照（stages/s02_perf）。

用法示例：
    uv run python -m tools.roofline --hw H100 --matmul 4096 4096 4096
    uv run python -m tools.roofline --hw H20 --decode-params 7e9 --batch 1
    uv run python -m tools.roofline --hw H100 --model llama3_8b --context 4096 --batches 1 8 32 128
"""

import argparse
from dataclasses import dataclass


# --8<-- [start:hardware]
# --8<-- [start:hardware_class]
# @dataclass 自动生成 __init__，按字段顺序传参：Hardware("H100 SXM", 989, 3350, 80)
# frozen=True 表示创建后字段不可修改，防止某处代码不小心把全局的硬件参数改掉
@dataclass(frozen=True)
class Hardware:
    name: str
    # 稠密（非稀疏）矩阵乘峰值，单位 TFLOPS
    # 只有矩阵乘能用到这个数；norm、softmax 等走 CUDA Core，峰值低约 15 倍
    peak_tflops: float  # BF16/FP16 dense tensor core 峰值
    # HBM 带宽，单位 GB/s（1e9 字节/秒）；decode、逐元素算子的耗时由它决定
    bandwidth_gbs: float  # 显存带宽 GB/s
    # 显存容量，决定"权重 + KV cache"能放多少（见 max_batch）
    memory_gb: float

    # @property 让方法像字段一样访问：hw.ridge_point，不用写括号
    @property
    def ridge_point(self) -> float:
        """拐点算术强度（FLOP/Byte）：低于它是带宽瓶颈，高于它是算力瓶颈。"""
        # 单位换算：TFLOPS 乘 1e12 得 FLOP/s，GB/s 乘 1e9 得 B/s，相除得 FLOP/B
        # 含义：每从 HBM 读 1 字节，至少要做这么多次运算才能把算力喂饱
        # H100：989e12 / 3.35e12 ≈ 295；H20：148e12 / 4.0e12 ≈ 37
        return self.peak_tflops * 1e12 / (self.bandwidth_gbs * 1e9)
# --8<-- [end:hardware_class]


# --8<-- [start:hardware_table]
# 参数：名字、BF16 稠密峰值 TFLOPS、带宽 GB/s、显存 GB（均为厂商标称值）
# 用标称值算出的是理论下限耗时；实测可达值约为标称算力的 60-80%、标称带宽的 80-90%
HARDWARE = {
    # H800 是 H100 的出口版：单卡算力、带宽相同，只砍了 NVLink 卡间带宽（影响多卡，不影响单卡 roofline）
    "H100": Hardware("H100 SXM", 989, 3350, 80),
    "H800": Hardware("H800 SXM", 989, 3350, 80),
    # H20：算力只有 H800 的 15%，带宽反而高 20%，拐点 37，是"为 decode 设计"的卡
    "H20": Hardware("H20", 148, 4000, 96),
    # 拐点 153
    "A100": Hardware("A100 80GB SXM", 312, 2039, 80),
    "A800": Hardware("A800 80GB SXM", 312, 2039, 80),
    # L 系列用 GDDR6 而不是 HBM，带宽只有 864 GB/s，decode 明显吃亏
    "L40S": Hardware("L40S", 362, 864, 48),
    "L20": Hardware("L20", 119, 864, 48),
    "4090": Hardware("RTX 4090", 165, 1008, 24),
    # 本机：算力用 S0 实测的 FP16 矩阵乘可达值，带宽用标称值。仅用于验证估算方法，不可外推到 NVIDIA
    # 拐点 7.4e12 / 273e9 ≈ 27
    "M4Pro": Hardware("Apple M4 Pro (MPS)", 7.4, 273, 48),
}
# --8<-- [end:hardware_table]
# --8<-- [end:hardware]


# --8<-- [start:estimate]
# --8<-- [start:estimate_inputs]
# 一次估算只需要两个数：这件事要做多少 FLOPs、至少要搬多少字节；再加上在哪张卡上做
# 所有上层函数（matmul、prefill、decode）的工作都只是"算出这两个数"，然后交给它
@dataclass
class Estimate:
    # 浮点运算次数；一次乘加记 2 FLOPs
    flops: float
    # 必须经过 HBM 的字节数（读 + 写），按理想最少访存算
    bytes: float
    hw: Hardware

    @property
    def intensity(self) -> float:
        # 算术强度，单位 FLOP/B；和 hw.ridge_point 比大小就知道落在 roofline 的哪一段
        return self.flops / self.bytes
# --8<-- [end:estimate_inputs]

# --8<-- [start:estimate_times]
    # 假设算力全部跑满时的耗时：FLOPs / (TFLOPS × 1e12)
    # 例：33.5 MFLOPs 在 H100 上 ≈ 0.03 μs
    @property
    def compute_time_s(self) -> float:
        return self.flops / (self.hw.peak_tflops * 1e12)

    # 假设带宽全部跑满时的耗时：字节 / (GB/s × 1e9)
    # 例：33.5 MB 在 H100 上 ≈ 10 μs
    @property
    def memory_time_s(self) -> float:
        return self.bytes / (self.hw.bandwidth_gbs * 1e9)

    @property
    def time_s(self) -> float:
        # 理想情况下计算和访存完全重叠，耗时取两者较大者
        # 所以这是一个下界：真实 kernel 重叠不完美、还有启动开销（几 μs），实测只会更慢
        # 上界是两者相加（完全不重叠）；强度远离拐点时两者差别很小
        return max(self.compute_time_s, self.memory_time_s)

    # 谁的时间长谁就是瓶颈；这等价于比较 intensity 和 ridge_point
    @property
    def bound(self) -> str:
        return "compute-bound（算力瓶颈）" if self.compute_time_s >= self.memory_time_s else "memory-bound（带宽瓶颈）"
# --8<-- [end:estimate_times]
# --8<-- [end:estimate]

    def report(self) -> str:
        return (
            f"[{self.hw.name}] FLOPs={self.flops:.3e}, Bytes={self.bytes:.3e}, "
            f"算术强度={self.intensity:.1f} FLOP/B (拐点 {self.hw.ridge_point:.0f})\n"
            f"  计算时间 {self.compute_time_s * 1e3:.4f} ms | 访存时间 {self.memory_time_s * 1e3:.4f} ms "
            f"=> 理论耗时 {self.time_s * 1e3:.4f} ms, {self.bound}"
        )


# --8<-- [start:matmul]
def matmul(m: int, n: int, k: int, hw: Hardware, dtype_bytes: int = 2) -> Estimate:
    """[M,K] @ [K,N]：FLOPs = 2MNK；最少访存 = 读 A、读 B、写 C。"""
    # 输出 C 有 M·N 个元素，每个是长度 K 的点积：K 次乘 + K 次加
    # 例：M=N=K=4096 时 2·4096³ ≈ 137 GFLOPs
    flops = 2 * m * n * k
    # A [M, K] 读一遍、B [K, N] 读一遍、C [M, N] 写一遍；dtype_bytes=2 对应 FP16/BF16
    # 这是"理想最少访存"：真实 kernel 分块计算，A、B 会被重复读，实际字节更多
    # decode 时 M=batch 很小：字节 ≈ 2·K·N（全是权重），强度 ≈ 2MNK / 2KN = M
    nbytes = dtype_bytes * (m * k + k * n + m * n)
    return Estimate(flops, nbytes, hw)
# --8<-- [end:matmul]


def decode_step(n_params: float, batch: int, hw: Hardware, dtype_bytes: int = 2, kv_bytes: float = 0.0) -> Estimate:
    """一个 decode step（不含 attention 细节的粗估）：
    计算 ≈ 2 * N * batch；访存 ≈ 全部权重读一遍 + 所有请求的 KV cache 读一遍。"""
    flops = 2 * n_params * batch
    nbytes = n_params * dtype_bytes + kv_bytes
    return Estimate(flops, nbytes, hw)


# ---------------------------------------------------------------- 整模型估算（S2）


@dataclass(frozen=True)
class ModelShape:
    """只保留性能估算需要的结构参数。与 stages/s01_transformer/code/model.py 的 GPTConfig 一一对应。"""

    name: str
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    ffn_hidden: int
    vocab_size: int
    tie_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    # --8<-- [start:model_shape]
    @property
    def matmul_params(self) -> float:
        """参与矩阵乘的参数（每层 Q/K/V/O + MLP，加上 lm_head）。embedding 查表不算 FLOPs。"""
        # kv：K、V 投影的输出维度 = KV 头数 × head_dim；GQA 时比 c 小
        c, kv = self.d_model, self.n_kv_head * self.head_dim
        # Q、O 投影各 c×c；K、V 投影各 c×kv；SwiGLU MLP 的 gate、up、down 三个 c×ffn
        per_layer = 2 * c * c + 2 * c * kv + 3 * c * self.ffn_hidden
        return self.n_layer * per_layer + self.vocab_size * c

    @property
    def total_params(self) -> float:
        embed = self.vocab_size * self.d_model
        return self.matmul_params + (0 if self.tie_embeddings else embed)

    def kv_bytes_per_token(self, kv_dtype_bytes: float = 2) -> float:
        # K 和 V 两份 × 层数 × KV 头数 × head_dim × 每元素字节；Llama-3-8B：2·32·8·128·2 = 128 KB
        return 2 * self.n_layer * self.n_kv_head * self.head_dim * kv_dtype_bytes
    # --8<-- [end:model_shape]

    @classmethod
    def from_gpt_config(cls, name: str, cfg) -> "ModelShape":
        return cls(name, cfg.n_layer, cfg.d_model, cfg.n_head, cfg.n_kv_head, cfg.ffn_hidden, cfg.vocab_size, cfg.tie_embeddings)


MODELS = {
    "llama3_8b": ModelShape("Llama-3-8B", 32, 4096, 32, 8, 14336, 128256),
    "llama3_70b": ModelShape("Llama-3-70B", 80, 8192, 64, 8, 28672, 128256),
    "qwen2_5_7b": ModelShape("Qwen2.5-7B", 28, 3584, 28, 4, 18944, 152064),
    "qwen2_5_1_5b": ModelShape("Qwen2.5-1.5B", 28, 1536, 12, 2, 8960, 151936, tie_embeddings=True),
    # 生成式推荐示例：HSTU/OneRec 量级的中等模型，MHA，语义 ID 小词表；用于 10K 用户序列的成本分析（S6/S7）
    "genrec_1b": ModelShape("GenRec-1B (示例)", 24, 2048, 16, 16, 5632, 8192),
}


# --8<-- [start:prefill_decode]
# --8<-- [start:prefill_fn]
# w_bytes / kv_bytes：权重和 KV 每个元素的字节数，BF16 是 2，FP8 是 1，INT4 是 0.5
def prefill(m: ModelShape, batch: int, seq_len: int, hw: Hardware, w_bytes: float = 2, kv_bytes: float = 2) -> Estimate:
    """prefill B 条长度为 T 的 prompt。
    FLOPs = 2·N_matmul·B·T（线性层）+ 2·L·B·T²·C（QK^T 与 AV，因果 mask 省一半：4·T²·C/2）
    Bytes ≈ 读一遍权重 + 写出 KV cache（激活值在 L2/片上流转，粗估忽略）"""
    # 一次前向要处理的 token 总数；线性层对每个 token 都做一遍 [C] @ [C, N]
    tokens = batch * seq_len
    # 第一项：每个参数对每个 token 做 1 次乘加 = 2 FLOPs，和 T 线性
    # 第二项：attention 的 QK^T、AV，和 T 平方（** 是乘方）；Llama-3-8B 4K 时约占 7%
    # 例：Llama-3-8B 1×4096：2 × 7.50e9 × 4096 ≈ 61.5T，加 2 × 32 × 4096² × 4096 ≈ 4.4T，共 65.9 TFLOPs
    # ÷ 989 TFLOPS = H100 上 66.6 ms
    flops = 2 * m.matmul_params * tokens + 2 * m.n_layer * batch * seq_len**2 * m.d_model
    # 权重只读一遍，被 B·T 个 token 共享 -> 强度在几千量级，远超拐点 -> 算力瓶颈
    # KV 是写出：Llama-3-8B 每 token 128 KB，4K 个 token 写 0.5 GB
    nbytes = m.total_params * w_bytes + tokens * m.kv_bytes_per_token(kv_bytes)
    return Estimate(flops, nbytes, hw)
# --8<-- [end:prefill_fn]


# --8<-- [start:decode_fn]
def decode(m: ModelShape, batch: int, context: int, hw: Hardware, w_bytes: float = 2, kv_bytes: float = 2) -> Estimate:
    """一个 decode step：B 条请求各生成 1 个 token，每条已有 context 个历史 token。
    FLOPs = 2·N_matmul·B + 4·L·B·S·C（新 token 对 S 个历史做 QK^T 和 AV）
    Bytes = 权重读一遍（所有请求共享）+ 每条请求读自己的 KV（不共享！）"""
    # 每条请求只有 1 个新 token：线性层 2·N·B；attention 是 1 个 query 对 S 个 key，没有 T²
    flops = 2 * m.matmul_params * batch + 4 * m.n_layer * batch * context * m.d_model
    # 第一项和 batch 无关：Llama-3-8B BF16 是 16 GB，B=1 时就是全部开销
    # 第二项随 B·S 线性增长：B=104（H100 显存上限）、S=4K 时是 104 × 4096 × 128 KB ≈ 56 GB，反过来压倒权重
    # 所以 B 增大时强度 ≈ 2NB / (2N + B·S·kv) 会饱和（Llama-3-8B 约 25），到不了拐点 295
    nbytes = m.total_params * w_bytes + batch * context * m.kv_bytes_per_token(kv_bytes)
    return Estimate(flops, nbytes, hw)
# --8<-- [end:decode_fn]
# --8<-- [end:prefill_decode]


# --8<-- [start:max_batch]
def max_batch(m: ModelShape, context: int, hw: Hardware, w_bytes: float = 2, kv_bytes: float = 2,
              mem_util: float = 0.9, n_gpus: int = 1) -> int:
    """显存能装下多少条 context 长度的请求：(总显存·利用率 - 权重) / 每条请求的 KV。"""
    # mem_util=0.9：留 10% 给激活、CUDA 上下文和碎片，对应 vLLM 的 --gpu-memory-utilization
    # n_gpus：TP 切分时权重和 KV 都均摊到多卡，这里直接把显存相加
    # 例：Qwen2.5-7B 在 H800 上 80·0.9 - 15.2 ≈ 56.8 GB 可用于 KV（1 GB 按 1e9 字节，和 memory_gb 一致）
    free = hw.memory_gb * 1e9 * mem_util * n_gpus - m.total_params * w_bytes
    # 每条请求要预留整段 context 的 KV：Qwen2.5-7B 4K 时 4096 × 56 KB ≈ 0.23 GB，56.8 / 0.235 ≈ 241 条
    # // 是整除；权重放不下时 free 为负，用 max(0, ...) 兜底
    return max(0, int(free // (context * m.kv_bytes_per_token(kv_bytes))))
# --8<-- [end:max_batch]


def main() -> None:
    p = argparse.ArgumentParser(description="Roofline 估算器")
    p.add_argument("--hw", default="H100", choices=sorted(HARDWARE))
    p.add_argument("--matmul", nargs=3, type=int, metavar=("M", "N", "K"))
    p.add_argument("--decode-params", type=float, help="模型参数量，如 7e9")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    args = p.parse_args()
    hw = HARDWARE[args.hw]

    if args.matmul:
        print(matmul(*args.matmul, hw).report())
    if args.decode_params:
        est = decode_step(args.decode_params, args.batch, hw)
        print(est.report())
        print(f"  => decode 吞吐上限约 {args.batch / est.time_s:.0f} tokens/s（batch={args.batch}）")
    if args.model:
        m = MODELS[args.model]
        print(f"{m.name}: {m.total_params / 1e9:.2f}B 参数，KV {m.kv_bytes_per_token() / 1024:.0f} KB/token，"
              f"{args.context} 上下文最多 {max_batch(m, args.context, hw)} 条并发（{hw.name}）")
        pf = prefill(m, 1, args.context, hw)
        print(f"prefill 1×{args.context}: {pf.time_s * 1e3:.1f} ms，{pf.bound}，算术强度 {pf.intensity:.0f}")
        for b in args.batches:
            d = decode(m, b, args.context, hw)
            print(f"decode B={b:4d}: 每步 {d.time_s * 1e3:.2f} ms，总吞吐 {b / d.time_s:8.0f} tok/s，"
                  f"算术强度 {d.intensity:6.1f}，{d.bound}")


if __name__ == "__main__":
    main()
