"""S3 第 9 课：MoE 层的参考实现（路由 -> 重排 -> grouped GEMM -> 反重排合并）+ moe_align_block_size + 实验。

1. route：top-k gating，得到每个 token 的 k 个专家编号和权重。
2. moe_naive：逐 token、逐专家地算，最慢但一眼能看出对不对，当标准答案。
3. moe_grouped_ref：按专家排序（permute）-> 每个专家一段变长 M 的 GEMM -> 按逆置换放回（unpermute）并按 gate 权重合并。
4. moe_align_block_size_ref：vLLM fused_moe 的预处理，把每个专家的 token 数补齐到 BLOCK_M 的倍数，产出
   sorted_token_ids、expert_ids（每个块属于哪个专家）、num_tokens_post_padded。
5. fused_moe_ref：按 4 的输出"每块一个专家"地算，和 triton_kernels/moe_grouped_gemm.py 一一对应。
6. main()：路由偏斜度 vs padding 浪费（纯计算）、逐专家循环 vs padding 成批 bmm 的耗时（Mac，加锁）、EP 通信估算。

记号：T=token 数，C=hidden（d_model），I=专家 FFN 中间维度，E=专家数，k=top_k。
专家权重沿用 vLLM 的布局：w1 [E, 2I, C]（gate 和 up 拼在一起），w2 [E, C, I]。

运行（会测耗时，要加锁）：
    uv run python -m tools.exclusive -- uv run python -m stages.s03_kernels.code.moe_ref
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from tabulate import tabulate

OUT = Path(__file__).resolve().parents[1] / "experiments" / "moe.md"


# --8<-- [start:route]
def route(x: torch.Tensor, w_gate: torch.Tensor, top_k: int, renormalize: bool = True):
    """top-k gating。x: [T, C]，w_gate: [E, C] -> (topk_weights [T, k] FP32, topk_ids [T, k] int64)。"""
    # 路由打分就是一个小 Linear：[T, C] @ [C, E]，E=128 时只占 MoE 层 FLOPs 的千分之几
    logits = x.float() @ w_gate.float().T  # [T, E]
    probs = logits.softmax(dim=-1)  # [T, E]
    # topk 沿最后一维取最大的 k 个，返回 (值, 下标)；下标就是被选中的专家编号
    topk_w, topk_ids = probs.topk(top_k, dim=-1)  # [T, k], [T, k]
    if renormalize:
        # Mixtral、Qwen-MoE 的做法：只在被选中的 k 个专家之间重新归一化，权重和为 1
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    return topk_w, topk_ids
# --8<-- [end:route]


# --8<-- [start:expert_ffn]
def expert_ffn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """一个专家的 SwiGLU FFN。x: [n, C]，w1: [2I, C]，w2: [C, I] -> [n, C]。"""
    # 一次 GEMM 同时算出 gate 和 up：[n, C] @ [C, 2I]
    h = x @ w1.T  # [n, 2I]
    # chunk(2, dim=-1) 沿最后一维对半切成 gate、up 两块 [n, I]
    gate, up = h.chunk(2, dim=-1)
    # SiLU(gate) ⊙ up，再投回 C 维
    return (F.silu(gate) * up) @ w2.T  # [n, C]
# --8<-- [end:expert_ffn]


# --8<-- [start:naive]
def moe_naive(x, topk_w, topk_ids, w1, w2):
    """逐 token、逐个被选中的专家地算：T·k 次小矩阵乘。只用来当标准答案。"""
    out = torch.zeros_like(x, dtype=torch.float32)  # [T, C]
    for t in range(x.size(0)):
        for j in range(topk_ids.size(1)):
            e = topk_ids[t, j].item()
            # x[t : t + 1] 保留 batch 维：[1, C]
            out[t] += topk_w[t, j] * expert_ffn(x[t : t + 1].float(), w1[e].float(), w2[e].float())[0]
    return out.to(x.dtype)
# --8<-- [end:naive]


# --8<-- [start:permute]
def moe_permute(topk_ids: torch.Tensor, num_experts: int):
    """按专家排序。topk_ids: [T, k] -> (order [T·k], counts [E], offsets [E+1])。

    把 T 个 token 的 k 个选择摊平成 T·k 个"(token, 槽位)"，flat 下标 i 对应 token i // k 的第 i % k 个选择。
    order[j] = 排序后第 j 行来自哪个 flat 下标；同一专家的行连续，专家 e 占 [offsets[e], offsets[e+1])。
    """
    flat = topk_ids.flatten()  # [T·k]
    # stable=True：同一专家内部保持 token 原来的顺序（结果可复现，也让相邻 token 的访存更连续）
    order = torch.sort(flat, stable=True).indices  # [T·k]
    # bincount 数每个专家被选了几次；minlength 保证没人选的专家也有一个 0
    counts = torch.bincount(flat, minlength=num_experts)  # [E]
    # 前缀和得到每个专家那一段的起点；前面补 0，长度 E+1
    offsets = F.pad(counts.cumsum(0), (1, 0))  # [E+1]
    return order, counts, offsets
# --8<-- [end:permute]


# --8<-- [start:grouped]
def moe_grouped_ref(x, topk_w, topk_ids, w1, w2):
    """permute -> 每个专家一段变长 GEMM -> unpermute + 加权合并。x: [T, C] -> [T, C]。"""
    t, k = topk_ids.shape
    e_num = w1.size(0)
    order, counts, offsets = moe_permute(topk_ids, e_num)
    # 第一步 permute：按排序后的顺序把 token 复制出来；一个 token 被选 k 次就出现 k 次
    # order // k 把 flat 下标换回 token 下标
    xs = x[order // k]  # [T·k, C]
    ys = torch.empty_like(xs)  # [T·k, C]
    # 第二步 grouped GEMM：每个专家处理自己那一段，段长 counts[e] 各不相同（变长 M）
    # 这里用 Python 循环逐个发 GEMM —— 正是第 1 步说的"E 次启动、小 M 吃不满"的写法
    for e in range(e_num):
        lo, hi = offsets[e].item(), offsets[e + 1].item()
        if hi > lo:
            ys[lo:hi] = expert_ffn(xs[lo:hi], w1[e], w2[e])
    # 第三步 unpermute：排序后的第 j 行放回 flat 位置 order[j]（散射写入就是逆置换）
    y_flat = torch.empty_like(ys)
    y_flat[order] = ys  # [T·k, C]，现在按 (token, 槽位) 排列
    # 第四步合并：[T, k, C] 乘 gate 权重 [T, k, 1]，沿 k 求和
    return (y_flat.view(t, k, -1).float() * topk_w[..., None]).sum(dim=1).to(x.dtype)  # [T, C]
# --8<-- [end:grouped]


# --8<-- [start:align]
def moe_align_block_size_ref(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    """vLLM moe_align_block_size 的 PyTorch 版。

    返回：
      sorted_token_ids [P]：按专家排好的 flat 下标，每个专家的段补齐到 block_m 的倍数，补位填哨兵 T·k
      expert_ids [P / block_m]：第 b 个块属于哪个专家（kernel 用它找权重）
      num_tokens_post_padded：P，有效的块数 × block_m
    """
    flat = topk_ids.flatten()  # [T·k]
    n_valid = flat.numel()
    order, counts, offsets = moe_permute(topk_ids, num_experts)
    # 每个专家补齐后的长度：向上取整到 block_m 的倍数；没人选的专家是 0，不占块
    padded = (counts + block_m - 1) // block_m * block_m  # [E]
    padded_offsets = F.pad(padded.cumsum(0), (1, 0))  # [E+1]
    total = padded_offsets[-1].item()
    # 先全填哨兵：kernel 看到 id >= T·k 就知道是补位，不读也不写
    sorted_ids = torch.full((total,), n_valid, dtype=torch.int32, device=flat.device)  # [P]
    # 排序后第 j 行属于专家 se[j]，是这个专家内部的第 (j - offsets[se]) 个
    se = flat[order]  # [T·k]
    rank = torch.arange(n_valid, device=flat.device) - offsets[se]
    # 放到补齐后的位置：专家段起点 + 段内名次
    sorted_ids[padded_offsets[se] + rank] = order.to(torch.int32)
    # 每个块一个专家号：专家 e 占 padded[e] / block_m 个块
    expert_ids = torch.repeat_interleave(torch.arange(num_experts, device=flat.device), padded // block_m)
    return sorted_ids, expert_ids.to(torch.int32), total
# --8<-- [end:align]


# --8<-- [start:fused_ref]
def block_gemm_ref(a, w, sorted_ids, expert_ids, block_m, top_k, n_valid, weights=None):
    """模拟 fused_moe kernel 的一次 grouped GEMM：每个 [block_m] 行块只乘它那个专家的权重。

    a: [T, K]（第一层，行下标 = id // top_k）或 [T·k, K]（第二层，top_k=1）；w: [E, N, K] -> c: [T·k 或 T, N]
    """
    n_blocks = expert_ids.numel()
    ids = sorted_ids.view(n_blocks, block_m).long()  # [nb, BM]
    valid = ids < n_valid  # [nb, BM]，False 是补位行
    # 补位行随便指向第 0 行（kernel 里用 mask 不读），结果最后不写回
    rows = torch.where(valid, ids // top_k, torch.zeros_like(ids))  # [nb, BM]
    a_blk = a[rows]  # [nb, BM, K]
    # 每个块取它专家的权重 [N, K]。PyTorch 只能真的复制出来；kernel 里只是指针偏移 expert·stride_e
    w_blk = w[expert_ids.long()]  # [nb, N, K]
    c_blk = a_blk @ w_blk.transpose(1, 2)  # [nb, BM, N]
    if weights is not None:
        # 第二层顺便乘 gate 权重（vLLM 的 MUL_ROUTED_WEIGHT），省一次单独的逐元素 kernel
        c_blk = c_blk * torch.where(valid, weights.flatten()[ids.clamp(max=n_valid - 1)], 0.0)[..., None]
    out = torch.zeros(n_valid, w.size(1), dtype=a.dtype, device=a.device)
    # 只写回有效行：按 flat 下标散射
    out[ids[valid]] = c_blk[valid].to(a.dtype)
    return out  # [T·k, N]


def fused_moe_ref(x, topk_w, topk_ids, w1, w2, block_m: int = 16):
    """vLLM fused_moe 的计算顺序：align -> GEMM1（按块查专家）-> SiLU·up -> GEMM2（乘 gate 权重）-> 沿 k 求和。"""
    t, k = topk_ids.shape
    n_valid = t * k
    sorted_ids, expert_ids, _ = moe_align_block_size_ref(topk_ids, w1.size(0), block_m)
    h = block_gemm_ref(x, w1, sorted_ids, expert_ids, block_m, k, n_valid)  # [T·k, 2I]
    gate, up = h.chunk(2, dim=-1)
    act = F.silu(gate) * up  # [T·k, I]
    # 第二层的 A 已经是每个 (token, 槽位) 一行，所以 top_k=1
    y = block_gemm_ref(act, w2, sorted_ids, expert_ids, block_m, 1, n_valid, weights=topk_w.to(act.dtype))
    return y.view(t, k, -1).sum(dim=1)  # [T, C]
# --8<-- [end:fused_ref]


# --8<-- [start:padded_bmm]
def moe_padded_bmm(x, topk_w, topk_ids, w1, w2):
    """按最大负载 padding：把每个专家的 token 都补到 max_count 行，一次 bmm 算完所有专家。"""
    t, k = topk_ids.shape
    e_num, _, c = w1.shape
    order, counts, offsets = moe_permute(topk_ids, e_num)
    cap = counts.max().item()
    # 每个排序后的行在 [E, cap] 网格里的位置：(专家号, 段内名次)
    se = topk_ids.flatten()[order]  # [T·k]
    rank = torch.arange(t * k, device=x.device) - offsets[se]
    buf = torch.zeros(e_num, cap, c, dtype=x.dtype, device=x.device)  # [E, cap, C]，大部分可能是 0
    buf[se, rank] = x[order // k]
    # 一次批量矩阵乘：[E, cap, C] @ [E, C, 2I]；补的 0 行照样参与计算，这就是浪费
    h = buf @ w1.transpose(1, 2)  # [E, cap, 2I]
    gate, up = h.chunk(2, dim=-1)
    y = (F.silu(gate) * up) @ w2.transpose(1, 2)  # [E, cap, C]
    y_flat = torch.empty(t * k, c, dtype=x.dtype, device=x.device)
    y_flat[order] = y[se, rank]
    return (y_flat.view(t, k, c).float() * topk_w[..., None]).sum(dim=1).to(x.dtype)
# --8<-- [end:padded_bmm]


# --8<-- [start:waste]
def zipf_routing(num_tokens: int, num_experts: int, top_k: int, alpha: float, seed: int = 0) -> torch.Tensor:
    """模拟偏斜的路由：专家 e 被选中的概率 ∝ 1/(e+1)^alpha。alpha=0 是均匀分布。返回 topk_ids [T, k]。"""
    g = torch.Generator().manual_seed(seed)
    p = 1.0 / torch.arange(1, num_experts + 1, dtype=torch.float64) ** alpha  # [E]
    # 每个 token 不放回地抽 k 个不同的专家（top-k 不会选同一个专家两次）
    return torch.multinomial(p.expand(num_tokens, -1), top_k, replacement=False, generator=g)  # [T, k]


def padding_waste(topk_ids: torch.Tensor, num_experts: int, block_m: int) -> dict:
    """三种组织方式各要算多少行：真实 T·k、补齐到 BLOCK_M、补齐到最大负载。"""
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)
    real = counts.sum().item()
    aligned = ((counts + block_m - 1) // block_m * block_m).sum().item()
    padded_max = num_experts * counts.max().item()
    return {
        "real": real,
        "aligned": aligned,
        "padded_max": padded_max,
        # 浪费比例 = 补出来的行 / 总共算的行
        "waste_aligned": 1 - real / aligned,
        "waste_max": 1 - real / padded_max,
        # 最忙专家 / 平均负载：EP 时最忙的那张卡决定这一层的耗时
        "max_over_mean": counts.max().item() / (real / num_experts),
        "active": int((counts > 0).sum().item()),
    }
# --8<-- [end:waste]


# --8<-- [start:a2a]
def ep_all_to_all_bytes(tokens: int, hidden: int, top_k: int, dispatch_bytes: float = 2, combine_bytes: float = 2) -> float:
    """专家并行一层 MoE 的 all-to-all 字节数（每张卡发出的量，不考虑同卡命中和按节点去重）。

    dispatch：每个 token 的隐藏向量发给 k 个专家所在的卡；combine：k 个专家的输出发回来。共 2 次。
    """
    return tokens * hidden * top_k * (dispatch_bytes + combine_bytes)
# --8<-- [end:a2a]


def main() -> None:
    from tools.bench import benchmark
    from tools.device import describe, get_device

    # ---- 实验 1：路由偏斜 vs padding 浪费（纯计算）。E=128、k=8 是 Qwen3-MoE 的配置
    E, K_TOP = 128, 8
    rows_waste = []
    for t in [16, 256, 4096]:
        for alpha in [0.0, 0.5, 1.0, 1.5]:
            ids = zipf_routing(t, E, K_TOP, alpha)
            r16 = padding_waste(ids, E, 16)
            r64 = padding_waste(ids, E, 64)
            rows_waste.append([t, alpha, r16["active"], f"{r16['max_over_mean']:.1f}", r16["real"],
                               f"{r16['waste_aligned']:.0%}", f"{r64['waste_aligned']:.0%}", f"{r16['waste_max']:.0%}"])

    # ---- 实验 2：Mac 实测逐专家循环 vs padding 成批 bmm
    device = get_device()
    torch.manual_seed(0)
    C, I_FF = 1024, 512
    rows_time = []
    for e_num, top_k in [(8, 2), (64, 8)]:
        w1 = torch.randn(e_num, 2 * I_FF, C, device=device, dtype=torch.float16) * 0.02
        w2 = torch.randn(e_num, C, I_FF, device=device, dtype=torch.float16) * 0.02
        for t in [64, 2048]:
            x = torch.randn(t, C, device=device, dtype=torch.float16)
            for alpha in [0.0, 1.2]:
                ids = zipf_routing(t, e_num, top_k, alpha).to(device)
                tw = torch.full(ids.shape, 1.0 / top_k, device=device)
                waste = padding_waste(ids.cpu(), e_num, 16)
                t_loop = benchmark(lambda: moe_grouped_ref(x, tw, ids, w1, w2), device, warmup=3, iters=10).median_ms
                t_pad = benchmark(lambda: moe_padded_bmm(x, tw, ids, w1, w2), device, warmup=3, iters=10).median_ms
                rows_time.append([f"E={e_num}, k={top_k}", t, alpha, waste["active"], f"{waste['waste_max']:.0%}",
                                  f"{t_loop:.2f}", f"{t_pad:.2f}", f"{t_loop / t_pad:.2f}x"])

    # ---- 实验 3：EP all-to-all 估算（预期，推导）
    rows_a2a = []
    for name, hidden, top_k, d_b, c_b in [("DeepSeek-V3（FP8 dispatch / BF16 combine）", 7168, 8, 1, 2),
                                          ("Qwen3-30B-A3B（BF16）", 2048, 8, 2, 2)]:
        for tokens in [128, 4096]:
            b = ep_all_to_all_bytes(tokens, hidden, top_k, d_b, c_b)
            rows_a2a.append([name, tokens, f"{b / tokens / 1e3:.0f}", f"{b / 1e6:.1f}",
                             f"{b / 400e9 * 1e6:.0f}", f"{b / 50e9 * 1e6:.0f}"])

    report = (
        "# MoE grouped GEMM 实验\n\n"
        f"## 实验 1：路由偏斜 vs padding 浪费（E={E}，k={K_TOP}，纯计算）\n\n"
        "alpha 是 Zipf 指数，0 为均匀；对齐到 BLOCK_M 是 vLLM fused_moe 的做法，补到最大负载是朴素 bmm 的做法。\n\n"
        + tabulate(rows_waste, headers=["T", "alpha", "被选中专家数", "最大/平均负载", "真实行数 T·k",
                                        "对齐 16 浪费", "对齐 64 浪费", "补到最大负载浪费"], tablefmt="github")
        + f"\n\n## 实验 2：Mac 实测（{describe(device)}，C={C}，I={I_FF}，FP16）\n\n"
        + tabulate(rows_time, headers=["配置", "T", "alpha", "被选中专家数", "补到最大负载浪费",
                                       "逐专家循环 ms", "padding bmm ms", "循环 / bmm"], tablefmt="github")
        + "\n\n## 实验 3：EP all-to-all 每层字节数（预期，推导；不含同卡命中和按节点去重）\n\n"
        + tabulate(rows_a2a, headers=["模型", "每卡 token 数", "每 token KB", "每层 MB", "NVLink 400 GB/s μs",
                                      "IB 50 GB/s μs"], tablefmt="github")
        + "\n"
    )
    print(report)
    OUT.write_text(report)
    print(f"已保存到 {OUT}")


if __name__ == "__main__":
    main()
