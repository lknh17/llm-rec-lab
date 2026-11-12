import pytest
import torch

from stages.s03_kernels.code.split_kv_decode import (
    DECODE_HW,
    best_split,
    build_paged_cache,
    gather_paged_kv,
    merge_partials,
    paged_split_kv_decode,
    sdpa_decode,
    split_kv_decode,
    split_kv_model,
)
from stages.s07_capstone.code.cascade_attention import cascade_attention


def _qkv(b, h, hkv, t, d, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, h, d), torch.randn(b, hkv, t, d), torch.randn(b, hkv, t, d)


# --8<-- [start:verify]
@pytest.mark.parametrize("t", [1, 7, 100, 513])
@pytest.mark.parametrize("num_splits", [1, 3, 8, 64])
@pytest.mark.parametrize("h,hkv", [(4, 4), (8, 2)])
def test_split_kv_matches_sdpa(t, num_splits, h, hkv):
    # 覆盖：T=1（只有一个历史 token）、T 不能被 S 整除、S 远大于 T（大量空段）、MHA 和 GQA
    q, k, v = _qkv(2, h, hkv, t, 16)
    torch.testing.assert_close(split_kv_decode(q, k, v, num_splits), sdpa_decode(q, k, v), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("page_size", [4, 16])
@pytest.mark.parametrize("num_splits", [1, 2, 5])
def test_paged_split_kv_matches_sdpa_with_ragged_lengths(page_size, num_splits):
    # 三条序列长度不同、物理块随机打乱：每条单独和它自己长度的 SDPA 对比
    seq_lens = [1, 37, 90]
    q, k, v = _qkv(3, 8, 2, max(seq_lens), 16, seed=1)
    k_pool, v_pool, table = build_paged_cache(k, v, seq_lens, page_size)
    out = paged_split_kv_decode(q, k_pool, v_pool, table, seq_lens, num_splits)
    for i, n in enumerate(seq_lens):
        ref = sdpa_decode(q[i : i + 1], k[i : i + 1, :, :n], v[i : i + 1, :, :n])
        torch.testing.assert_close(out[i : i + 1], ref, atol=1e-5, rtol=1e-4)


def test_merge_is_the_same_formula_as_cascade_attention():
    # S7 的 cascade attention：共享前缀一段 + 私有一段，两段用同一个 lse 合并公式
    torch.manual_seed(2)
    w, h, hkv, p, s, d = 3, 8, 2, 40, 5, 16
    q = torch.randn(w, h, d)
    k_shared, v_shared = torch.randn(hkv, p, d), torch.randn(hkv, p, d)
    k_priv, v_priv = torch.randn(w, hkv, s, d), torch.randn(w, hkv, s, d)
    # 把"共享前缀 + 私有"拼成每个 beam 的完整 KV，按 split=2 的思路切在前缀边界上也应得到同一结果
    k_full = torch.cat([k_shared.expand(w, -1, -1, -1), k_priv], dim=2)
    v_full = torch.cat([v_shared.expand(w, -1, -1, -1), v_priv], dim=2)
    torch.testing.assert_close(cascade_attention(q, k_shared, v_shared, k_priv, v_priv), sdpa_decode(q, k_full, v_full),
                               atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(split_kv_decode(q, k_full, v_full, 2), sdpa_decode(q, k_full, v_full),
                               atol=1e-5, rtol=1e-4)
# --8<-- [end:verify]


def test_merge_ignores_empty_splits():
    o = torch.tensor([[1.0, 2.0], [100.0, 100.0]])  # 第 2 段是空段，o 可以是任何值
    lse = torch.tensor([0.5, float("-inf")])
    torch.testing.assert_close(merge_partials(o, lse, dim=0), torch.tensor([1.0, 2.0]))


def test_split_kv_large_logits_stable():
    q, k, v = _qkv(1, 4, 4, 300, 16)
    out = split_kv_decode(q * 30, k * 30, v, 7)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, sdpa_decode(q * 30, k * 30, v), atol=1e-4, rtol=1e-3)


def test_gather_paged_kv_roundtrip():
    q, k, v = _qkv(2, 4, 2, 50, 8)
    k_pool, _, table = build_paged_cache(k, v, [50, 50], 16)
    torch.testing.assert_close(gather_paged_kv(k_pool, table, 50), k)


def test_model_more_splits_fill_the_gpu():
    hw = DECODE_HW["H100"]
    # Qwen2.5-7B：B=1 时只有 4 个 KV 头 = 4 个 program，远小于 132 个 SM
    one = split_kv_model(1, 28, 4, 10000, 128, 1, hw)
    assert one["programs"] == 4
    best = best_split(1, 28, 4, 10000, 128, hw)
    assert best >= 16
    assert split_kv_model(1, 28, 4, 10000, 128, best, hw)["total_us"] < one["total_us"] / 5
    # batch 大了，B·Hkv 本身就够多，最佳 S 变小
    assert best_split(32, 28, 4, 10000, 128, hw) <= 2
