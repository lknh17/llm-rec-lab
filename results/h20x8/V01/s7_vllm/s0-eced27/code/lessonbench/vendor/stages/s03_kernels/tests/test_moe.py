import pytest
import torch

from stages.s03_kernels.code.moe_ref import (
    ep_all_to_all_bytes,
    fused_moe_ref,
    moe_align_block_size_ref,
    moe_grouped_ref,
    moe_naive,
    moe_padded_bmm,
    moe_permute,
    padding_waste,
    route,
    zipf_routing,
)


def _setup(t=37, c=32, i=24, e=6, k=2, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(t, c)
    w_gate = torch.randn(e, c)
    w1 = torch.randn(e, 2 * i, c) * 0.1
    w2 = torch.randn(e, c, i) * 0.1
    topk_w, topk_ids = route(x, w_gate, k)
    return x, topk_w, topk_ids, w1, w2


# --8<-- [start:verify]
@pytest.mark.parametrize("t,e,k", [(1, 4, 1), (37, 6, 2), (64, 16, 4)])
def test_grouped_and_fused_match_naive(t, e, k):
    x, topk_w, topk_ids, w1, w2 = _setup(t=t, e=e, k=k)
    ref = moe_naive(x, topk_w, topk_ids, w1, w2)
    torch.testing.assert_close(moe_grouped_ref(x, topk_w, topk_ids, w1, w2), ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(moe_padded_bmm(x, topk_w, topk_ids, w1, w2), ref, atol=1e-5, rtol=1e-4)
    for block_m in (4, 16):
        torch.testing.assert_close(fused_moe_ref(x, topk_w, topk_ids, w1, w2, block_m), ref, atol=1e-5, rtol=1e-4)


def test_align_block_size_layout():
    # 3 个 token、top-2、4 个专家：专家 3 没人选，专家 0 被选 3 次
    topk_ids = torch.tensor([[0, 2], [0, 1], [2, 0]])
    sorted_ids, expert_ids, total = moe_align_block_size_ref(topk_ids, 4, block_m=2)
    sentinel = 6  # T·k
    # 专家 0 的 flat 下标 0, 2, 5 补到 4 行；专家 1 是 3，补到 2 行；专家 2 是 1, 4
    assert sorted_ids.tolist() == [0, 2, 5, sentinel, 3, sentinel, 1, 4]
    assert expert_ids.tolist() == [0, 0, 1, 2]
    assert total == 8
# --8<-- [end:verify]


def test_permute_is_invertible():
    _, _, topk_ids, _, _ = _setup(t=50, e=8, k=3)
    order, counts, offsets = moe_permute(topk_ids, 8)
    flat = topk_ids.flatten()
    # 排序后专家号单调不减，且每个专家的段长等于 counts
    assert torch.all(flat[order][1:] >= flat[order][:-1])
    assert counts.sum() == flat.numel() and offsets[-1] == flat.numel()
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel())
    assert torch.equal(order[inv], torch.arange(order.numel()))


def test_skewed_routing_wastes_more_when_padding_to_max():
    uniform = padding_waste(zipf_routing(4096, 128, 8, 0.0), 128, 16)
    skewed = padding_waste(zipf_routing(4096, 128, 8, 1.5), 128, 16)
    assert skewed["max_over_mean"] > 3 * uniform["max_over_mean"]
    assert skewed["waste_max"] > uniform["waste_max"]
    # 对齐到 BLOCK_M 的浪费有上界：每个被选中的专家最多浪费 BLOCK_M-1 行
    assert skewed["aligned"] - skewed["real"] <= 128 * 15


def test_a2a_bytes_deepseek_v3():
    # 每 token：7168 维 × 8 个专家 ×（1 字节 FP8 发出 + 2 字节 BF16 收回）≈ 172 KB
    assert abs(ep_all_to_all_bytes(1, 7168, 8, 1, 2) - 172_032) < 1
