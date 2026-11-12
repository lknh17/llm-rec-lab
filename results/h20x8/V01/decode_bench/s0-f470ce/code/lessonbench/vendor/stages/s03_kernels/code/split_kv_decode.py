"""S3 第 7 课：decode attention 的 split-KV（FlashDecoding）参考实现 + 访存/并行度模型 + Mac 实测。

1. split_kv_decode：把 KV 沿序列切成 S 段，每段独立算局部 (o, lse)，最后用 log-sum-exp 合并。
2. paged_split_kv_decode：KV 存在块池（block pool）里，按 block table 间接寻址，逻辑和 Triton kernel 一一对应。
3. split_kv_model：给定 B、H、Hkv、T、S 和硬件，估算并发程序数、每个程序读多少 KV、合并开销和总耗时。
4. main()：Mac 上实测不同 S 的耗时，并输出 H100 / H20 上的模型预测，写入 experiments/split_kv_decode.md。

记号：B=batch，H=query 头数，Hkv=KV 头数，G=H/Hkv（每个 KV 头服务几个 Q 头），T=已有上下文长度，
D=head_dim，S=num_splits（KV 切几段），P=page_size（每个物理块放几个 token）。

运行（会测耗时，要加锁）：
    uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.code.split_kv_decode
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tabulate import tabulate

OUT = Path(__file__).resolve().parents[1] / "experiments" / "split_kv_decode.md"


# --8<-- [start:sdpa_decode]
def sdpa_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """官方实现做对照。q: [B, H, D]，k/v: [B, Hkv, T, D] -> [B, H, D]。"""
    h, hkv = q.size(1), k.size(1)
    # GQA：把每个 KV 头复制 G 份给它的 Q 头组（和 S1 第 1 课同样的 repeat_interleave 约定）
    # 这里只为了拿到"标准答案"，真正的 decode kernel 不复制，见 split_kv_decode 里的分组写法
    k = k.repeat_interleave(h // hkv, dim=1)  # [B, H, T, D]
    v = v.repeat_interleave(h // hkv, dim=1)  # [B, H, T, D]
    # q[:, :, None] 在第 2 维插入长度为 1 的"query 序列"维：decode 每条请求每个头只有 1 个新 query
    # decode 时新 token 能看到全部历史，所以不需要因果 mask（is_causal 默认 False）
    return F.scaled_dot_product_attention(q[:, :, None], k, v)[:, :, 0]  # [B, H, D]
# --8<-- [end:sdpa_decode]


# --8<-- [start:split_kv]
# --8<-- [start:split_kv_partial]
def split_kv_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_splits: int) -> torch.Tensor:
    """split-KV decode（连续 KV）。q: [B, H, D]，k/v: [B, Hkv, T, D] -> [B, H, D]。

    S 段在这里用一次批量 einsum 同时算（PyTorch 里"并行"就是多一个批维），
    GPU kernel 里则是 S 个独立的 program 各算一段。
    """
    b, h, d = q.shape
    hkv, t = k.size(1), k.size(2)
    g = h // hkv
    s = num_splits
    # 每段的长度：向上取整，最后一段可能不满（用 mask 处理）
    chunk = math.ceil(t / s)
    pad = chunk * s - t

    # GQA 分组：同一个 KV 头的 G 个 query 放在一起，这一组只读一遍这个 KV 头
    # 先乘 1/sqrt(D)：只缩放 [B, H, D] 的 q，比缩放分数便宜
    qg = q.float().view(b, hkv, g, d) * d**-0.5  # [B, Hkv, G, D]
    # F.pad 的参数从最后一维往前写：(D 左, D 右, T 左, T 右)，这里只在 T 的末尾补 pad 个 0
    kf = F.pad(k.float(), (0, 0, 0, pad)).view(b, hkv, s, chunk, d)  # [B, Hkv, S, n, D]
    vf = F.pad(v.float(), (0, 0, 0, pad)).view(b, hkv, s, chunk, d)  # [B, Hkv, S, n, D]

    # 每段内的分数：einsum 的下标就是形状。对每个 (b, kv 头, 段, 组内 q 头) 做 [D]·[n, D]^T
    scores = torch.einsum("bhgd,bhsnd->bhsgn", qg, kf)  # [B, Hkv, S, G, n]
    # 补出来的位置（全局下标 >= T）不能参与 softmax：填 -inf
    pos = torch.arange(s * chunk, device=q.device).view(s, chunk)  # [S, n]
    # [S, n] 中间插一维成 [S, 1, n]，广播到 [B, Hkv, S, G, n]
    scores = scores.masked_fill((pos >= t)[:, None, :], float("-inf"))
# --8<-- [end:split_kv_partial]

# --8<-- [start:split_kv_local]
    # 每段各自做一次"局部 softmax"：只看本段的最大值 m 和指数和 l
    # amax 沿指定维取最大值（只返回值，不像 max(dim) 还返回下标）
    m = scores.amax(dim=-1)  # [B, Hkv, S, G]
    # T 很短、S 很大时最后几段可能全是补位（m = -inf）。exp(-inf - (-inf)) 是 NaN，先把 m 换成 0
    # torch.where(条件, a, b)：条件为真处取 a，否则取 b，逐元素进行
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p = torch.exp(scores - m_safe[..., None])  # [B, Hkv, S, G, n]
    l = p.sum(dim=-1)  # [B, Hkv, S, G]
    # 局部输出已经按本段的 l 归一化；clamp 防止空段 0/0
    o = torch.einsum("bhsgn,bhsnd->bhsgd", p, vf) / l.clamp_min(1e-30)[..., None]  # [B, Hkv, S, G, D]
    # lse = log Σ exp(score) = m + log l：一个数概括"这段的 softmax 分母有多大"，空段记 -inf
    lse = torch.where(l > 0, m_safe + torch.log(l.clamp_min(1e-30)), torch.full_like(l, float("-inf")))
    # 合并 S 段（dim=2），再把 [Hkv, G] 拼回 H 个头
    out = merge_partials(o, lse, dim=2)  # [B, Hkv, G, D]
    return out.reshape(b, h, d).to(q.dtype)  # [B, H, D]
# --8<-- [end:split_kv_local]
# --8<-- [end:split_kv]


# --8<-- [start:merge]
def merge_partials(o: torch.Tensor, lse: torch.Tensor, dim: int) -> torch.Tensor:
    """把沿 dim 排开的 S 段局部结果合并成一个。o: [..., S, ..., D]，lse: 去掉 D 的同形状。

    第 s 段的局部输出 o_s 是按它自己的分母 e^{lse_s} 归一化的。整体的分母是 Σ e^{lse_s}，
    所以整体输出 = Σ o_s · e^{lse_s} / Σ e^{lse_s}。和 S7 cascade attention 的合并公式完全相同。
    """
    # 减去最大的 lse 再取 exp，防止上溢：和 online softmax 减最大值是同一个技巧
    lse_max = lse.amax(dim=dim, keepdim=True)  # [..., 1, ...]
    # 每段的权重 = 它的分母占比（未归一化）；空段 lse=-inf，权重是 0
    w = torch.exp(lse - lse_max)  # [..., S, ...]
    # 加权平均：分子是 Σ w_s·o_s，分母是 Σ w_s；[..., None] 让 w 广播到 D 维
    return (o * w[..., None]).sum(dim=dim) / w.sum(dim=dim)[..., None]
# --8<-- [end:merge]


# --8<-- [start:paged]
# --8<-- [start:paged_build]
def build_paged_cache(k: torch.Tensor, v: torch.Tensor, seq_lens: list[int], page_size: int,
                      seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把连续的 k/v [B, Hkv, T, D] 拷进一个块池，物理块号随机打乱，模拟 vLLM 运行一段时间后的碎片化。

    返回 k_pool/v_pool: [num_blocks, P, Hkv, D]（vLLM 的布局：块 -> 块内槽位 -> 头 -> D），
    block_table: [B, max_pages] int32，block_table[b, i] = 第 b 条序列第 i 个逻辑块所在的物理块号。
    """
    b, hkv, _, d = k.shape
    # 每条序列需要的页数，向上取整；最后一页可能只用了一部分槽位
    pages = [math.ceil(n / page_size) for n in seq_lens]
    num_blocks = sum(pages)
    # 随机打乱物理块号：同一条序列的相邻逻辑块，物理上不相邻
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_blocks, generator=g).tolist()
    k_pool = torch.zeros(num_blocks, page_size, hkv, d, dtype=k.dtype, device=k.device)
    v_pool = torch.zeros_like(k_pool)
    # 没用到的表项填 0：kernel 靠 seq_lens 判断边界，永远不会读到它们
    block_table = torch.zeros(b, max(pages), dtype=torch.int32, device=k.device)
    nxt = 0
    for i, n in enumerate(seq_lens):
        for page in range(pages[i]):
            phys = perm[nxt]
            nxt += 1
            block_table[i, page] = phys
            lo, hi = page * page_size, min((page + 1) * page_size, n)
            # [Hkv, n_tok, D] -> [n_tok, Hkv, D]，写进物理块的前 n_tok 个槽位
            k_pool[phys, : hi - lo] = k[i, :, lo:hi].transpose(0, 1)
            v_pool[phys, : hi - lo] = v[i, :, lo:hi].transpose(0, 1)
    return k_pool, v_pool, block_table
# --8<-- [end:paged_build]


# --8<-- [start:paged_decode]
def paged_split_kv_decode(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                          block_table: torch.Tensor, seq_lens: list[int], num_splits: int) -> torch.Tensor:
    """paged 版 split-KV decode，每条序列长度可以不同。q: [B, H, D] -> [B, H, D]。

    循环结构和 triton_kernels/split_kv_decode.py 的 partial kernel 一一对应：
    外两层 (b, split) 就是 kernel 的 grid，内层"按页取 KV"就是 kernel 里的 for 循环。
    """
    b, h, d = q.shape
    page_size, hkv = k_pool.size(1), k_pool.size(2)
    g = h // hkv
    out = torch.empty(b, h, d, dtype=q.dtype, device=q.device)
    for i in range(b):
        n = seq_lens[i]
        # 按"页"切段而不是按 token：每段是整数个物理块，kernel 里每次循环正好读一个块
        n_pages = math.ceil(n / page_size)
        pages_per_split = math.ceil(n_pages / num_splits)
        qg = q[i].float().view(hkv, g, d) * d**-0.5  # [Hkv, G, D]
        o_parts, lse_parts = [], []
        for s in range(num_splits):
            p0 = s * pages_per_split
            p1 = min(p0 + pages_per_split, n_pages)
            # 这一段负责的 token 下标 [t0, t1)；最后一页按真实长度截断
            t0, t1 = p0 * page_size, min(p1 * page_size, n)
            pos = torch.arange(t0, max(t0, t1), device=q.device)  # [n_s]
            # 间接寻址的核心两行：逻辑位置 -> (物理块号, 块内槽位)
            phys = block_table[i, pos // page_size].long()  # [n_s]
            slot = pos % page_size  # [n_s]
            # 高级索引：两个 [n_s] 的下标同时作用于前两维，取出 [n_s, Hkv, D]
            kb = k_pool[phys, slot].float().transpose(0, 1)  # [Hkv, n_s, D]
            vb = v_pool[phys, slot].float().transpose(0, 1)  # [Hkv, n_s, D]
            o_s, lse_s = _partial(qg, kb, vb)  # [Hkv, G, D], [Hkv, G]
            o_parts.append(o_s)
            lse_parts.append(lse_s)
        # 堆成 [S, Hkv, G, D] 和 [S, Hkv, G]，沿 dim=0 合并
        o_all = merge_partials(torch.stack(o_parts), torch.stack(lse_parts), dim=0)  # [Hkv, G, D]
        out[i] = o_all.reshape(h, d).to(q.dtype)
    return out
# --8<-- [end:paged_decode]


def _partial(qg: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """一段 KV 上的局部 attention。qg: [Hkv, G, D]（已缩放），k/v: [Hkv, n, D] -> (o [Hkv, G, D], lse [Hkv, G])。"""
    if k.size(1) == 0:
        # 空段：S 大于页数时会出现。o 填 0、lse 填 -inf，合并时权重自动为 0
        return (torch.zeros_like(qg), torch.full(qg.shape[:-1], float("-inf"), device=qg.device))
    scores = qg @ k.transpose(-1, -2)  # [Hkv, G, n]
    lse = torch.logsumexp(scores, dim=-1)  # [Hkv, G]
    return torch.softmax(scores, dim=-1) @ v, lse  # [Hkv, G, D], [Hkv, G]
# --8<-- [end:paged]


# --8<-- [start:gather]
def gather_paged_kv(pool: torch.Tensor, block_table: torch.Tensor, t: int) -> torch.Tensor:
    """等长序列的批量 gather：块池 [N, P, Hkv, D] -> 连续 [B, Hkv, T, D]。

    PyTorch 没法"边寻址边算"，只能先拷出一份连续 KV 再算，多了一遍完整的 KV 写和读；
    paged attention kernel 在循环里直接按 block table 读物理块，没有这次拷贝。
    """
    page_size = pool.size(1)
    pos = torch.arange(t, device=pool.device)  # [T]
    # block_table[:, pos // P] 对每条序列取出 T 个位置各自的物理块号
    phys = block_table[:, pos // page_size].long()  # [B, T]
    # 物理块号 [B, T] 和槽位 [T]（广播成 [B, T]）一起索引，得到 [B, T, Hkv, D]
    return pool[phys, pos % page_size].transpose(1, 2)  # [B, Hkv, T, D]
# --8<-- [end:gather]


# --8<-- [start:model]
# --8<-- [start:model_hw]
@dataclass(frozen=True)
class DecodeHW:
    name: str
    n_sm: int
    # HBM 标称带宽 GB/s
    bw_gbs: float
    # 一个 program 能拉动的带宽（Little 定律）：在途字节 / 访存延迟
    # 假设每个 program 同时有约 32 KB 的 K/V 加载在路上（2 级流水：算当前块时，下一块的 K、V
    # 各一块 [64, 128] FP16 = 16 KB 已发出，共 32 KB），
    # HBM 延迟约 0.8 μs：32 KB / 0.8 μs ≈ 40 GB/s。这是估算假设，要用 GPU 上扫 S 的实测校准
    # triton_kernels/split_kv_decode.py 每次只读一页（PAGE=16，K、V 各 4 KB），在途量更小，实测可能低于这个值
    prog_bw_gbs: float = 40.0
    # 一次 kernel 启动 + 尾部的固定开销（μs），CUDA Graph 下约 2-4 μs
    launch_us: float = 3.0


DECODE_HW = {
    "H100": DecodeHW("H100 SXM", 132, 3350),
    "H20": DecodeHW("H20", 78, 4000),
}
# --8<-- [end:model_hw]


# --8<-- [start:model_body]
def split_kv_model(b: int, h: int, hkv: int, t: int, d: int, s: int, hw: DecodeHW, kv_bytes: int = 2) -> dict:
    """一层 decode attention 的 split-KV 估算（每个 program 负责一个 (b, kv 头, 段)，组内 G 个 q 头一起算）。"""
    # 并发 program 数：decode 时没有 Q 块可分，只能在 B、KV 头、段三个维度上找并行
    programs = b * hkv * s
    # 必须读的 KV 字节：K 和 V 各 [B, Hkv, T, D]。GQA 分组后每个 KV 头只读一遍
    kv_total = 2 * b * hkv * t * d * kv_bytes
    per_prog = kv_total / programs
    # 能达到的带宽：program 不够多时，受"并发数 × 每个 program 的带宽"限制；够多时被 HBM 封顶
    achieved_gbs = min(hw.bw_gbs, programs * hw.prog_bw_gbs)
    t_partial_us = hw.launch_us + kv_total / (achieved_gbs * 1e9) * 1e6
    # 合并开销：partial kernel 写出 S 份 FP32 的 (o, lse)，reduce kernel 读回来再写最终输出
    merge_bytes = 2 * b * h * s * (d + 1) * 4 + b * h * d * kv_bytes if s > 1 else 0
    # S=1 时不需要 reduce kernel；S>1 多一次启动
    t_merge_us = (hw.launch_us + merge_bytes / (hw.bw_gbs * 1e9) * 1e6) if s > 1 else 0.0
    total_us = t_partial_us + t_merge_us
    return {
        "programs": programs,
        "kv_mb": kv_total / 1e6,
        "per_prog_kb": per_prog / 1e3,
        "achieved_gbs": achieved_gbs,
        "merge_kb": merge_bytes / 1e3,
        "total_us": total_us,
        # 有效带宽 = 必须读的 KV / 总耗时（含启动和合并），这才是 decode attention 的 MBU
        "eff_gbs": kv_total / (total_us * 1e-6) / 1e9,
    }


def best_split(b: int, h: int, hkv: int, t: int, d: int, hw: DecodeHW, candidates=(1, 2, 4, 8, 16, 32, 64, 128)) -> int:
    """在候选 S 里挑模型预测耗时最短的。每段至少要有 1 页（这里按 16 个 token 算），否则没有意义。"""
    ok = [s for s in candidates if t / s >= 16]
    return min(ok, key=lambda s: split_kv_model(b, h, hkv, t, d, s, hw)["total_us"])
# --8<-- [end:model_body]
# --8<-- [end:model]


def main() -> None:
    from tools.bench import benchmark
    from tools.device import describe, get_device

    device = get_device()
    torch.manual_seed(0)
    fp16 = torch.float16
    # 内容理解 LLM（Qwen2.5-7B）一层 attention 的形状：28 个 Q 头、4 个 KV 头、head_dim 128
    H, HKV, D = 28, 4, 128
    rows_model = []
    for hw_name, hw in DECODE_HW.items():
        for b, t in [(1, 4096), (1, 10000), (1, 32768), (32, 4096), (32, 10000)]:
            s_best = best_split(b, H, HKV, t, D, hw)
            base = split_kv_model(b, H, HKV, t, D, 1, hw)
            best = split_kv_model(b, H, HKV, t, D, s_best, hw)
            rows_model.append([hw_name, b, t, base["programs"], f"{base['kv_mb']:.1f}", f"{base['total_us']:.1f}",
                               f"{base['eff_gbs']:.0f}", s_best, best["programs"], f"{best['per_prog_kb']:.0f}",
                               f"{best['total_us']:.1f}", f"{best['eff_gbs']:.0f}",
                               f"{base['total_us'] / best['total_us']:.1f}x"])

    # 同一个配置（H100、B=1、T=10000）下逐个 S 的模型曲线
    hw = DECODE_HW["H100"]
    rows_curve = []
    for s in [1, 2, 4, 8, 16, 32, 64, 128]:
        r = split_kv_model(1, H, HKV, 10000, D, s, hw)
        rows_curve.append([s, r["programs"], f"{r['per_prog_kb']:.0f}", f"{r['achieved_gbs']:.0f}",
                           f"{r['merge_kb']:.0f}", f"{r['total_us']:.1f}", f"{r['eff_gbs']:.0f}"])

    # Mac 实测：B=1 和 B=8，T=10000，扫 S
    rows_mac = []
    for b in [1, 8]:
        t = 10000
        q = torch.randn(b, H, D, device=device, dtype=fp16)
        k = torch.randn(b, HKV, t, D, device=device, dtype=fp16)
        v = torch.randn(b, HKV, t, D, device=device, dtype=fp16)
        kv_mb = 2 * k.numel() * 2 / 1e6
        # 带宽参照：只把 K、V 各读一遍（求和），这是任何 decode attention 在这台机器上的下限
        floor_ms = benchmark(lambda: (k.sum(), v.sum()), device, warmup=5, iters=30).median_ms
        rows_mac.append([b, "只读一遍 K、V（下限参照）", f"{kv_mb:.1f}", f"{floor_ms:.3f}", f"{kv_mb / floor_ms:.0f}"])
        gqa_ms = benchmark(lambda: F.scaled_dot_product_attention(q[:, :, None], k, v, enable_gqa=True),
                           device, warmup=5, iters=30).median_ms
        rows_mac.append([b, "SDPA（enable_gqa，不复制 KV）", f"{kv_mb:.1f}", f"{gqa_ms:.3f}", f"{kv_mb / gqa_ms:.0f}"])
        sdpa_ms = benchmark(lambda: sdpa_decode(q, k, v), device, warmup=5, iters=30).median_ms
        rows_mac.append([b, "SDPA（KV 先 repeat 成 H 头）", f"{kv_mb:.1f}", f"{sdpa_ms:.3f}", f"{kv_mb / sdpa_ms:.0f}"])
        for s in [1, 4, 16, 64]:
            ms = benchmark(lambda: split_kv_decode(q, k, v, s), device, warmup=5, iters=30).median_ms
            rows_mac.append([b, f"split_kv_decode S={s}", f"{kv_mb:.1f}", f"{ms:.3f}", f"{kv_mb / ms:.0f}"])

    # 旋钮 B：paged 的代价。PyTorch 参考实现只能先 gather 成连续张量；kernel 不需要这一步
    rows_paged = []
    b, t = 8, 10000
    k = torch.randn(b, HKV, t, D, device=device, dtype=fp16)
    v = torch.randn(b, HKV, t, D, device=device, dtype=fp16)
    q = torch.randn(b, H, D, device=device, dtype=fp16)
    kv_mb = 2 * k.numel() * 2 / 1e6
    contig_ms = benchmark(lambda: split_kv_decode(q, k, v, 16), device, warmup=5, iters=30).median_ms
    rows_paged.append(["连续 KV", "-", f"{contig_ms:.3f}", "-"])
    for page in [16, 128]:
        k_pool, v_pool, table = build_paged_cache(k, v, [t] * b, page)

        def paged_then_split():
            kc = gather_paged_kv(k_pool, table, t)
            vc = gather_paged_kv(v_pool, table, t)
            return split_kv_decode(q, kc, vc, 16)

        gather_ms = benchmark(lambda: gather_paged_kv(k_pool, table, t), device, warmup=5, iters=30).median_ms
        ms = benchmark(paged_then_split, device, warmup=5, iters=30).median_ms
        rows_paged.append([f"块池 + gather 再算（P={page}）", f"{2 * gather_ms:.3f}", f"{ms:.3f}", f"{ms / contig_ms:.2f}x"])

    report = (
        "# split-KV decode 实验\n\n"
        f"形状：Qwen2.5-7B 单层 attention，H={H}、Hkv={HKV}、D={D}，FP16。\n\n"
        "## 访存/并行度模型（预期，未实测，需要 GPU）\n\n"
        f"假设：每个 program 约 {hw.prog_bw_gbs:.0f} GB/s（32 KB 在途 / 0.8 μs 延迟），每次 kernel 启动 {hw.launch_us:.0f} μs。\n\n"
        + tabulate(rows_model, headers=["卡", "B", "T", "S=1 program 数", "KV MB", "S=1 μs", "S=1 有效 GB/s",
                                        "最佳 S", "program 数", "每 program KB", "最佳 μs", "有效 GB/s", "提速"],
                   tablefmt="github")
        + "\n\n## H100、B=1、T=10000 时逐个 S 的模型曲线（预期，未实测，需要 GPU）\n\n"
        + tabulate(rows_curve, headers=["S", "program 数", "每 program KB", "可达带宽 GB/s", "合并读写 KB",
                                        "总 μs", "有效 GB/s"], tablefmt="github")
        + f"\n\n## Mac 实测（{describe(device)}，T=10000）\n\n"
        + tabulate(rows_mac, headers=["B", "写法", "KV MB", "ms", "有效 GB/s"], tablefmt="github")
        + "\n\n## 旋钮 B：paged KV 在 PyTorch 参考实现里的代价（Mac，B=8，T=10000，S=16）\n\n"
        + tabulate(rows_paged, headers=["写法", "gather K+V ms", "总 ms", "相对连续"], tablefmt="github")
        + "\n"
    )
    print(report)
    OUT.write_text(report)
    print(f"已保存到 {OUT}")


if __name__ == "__main__":
    main()
