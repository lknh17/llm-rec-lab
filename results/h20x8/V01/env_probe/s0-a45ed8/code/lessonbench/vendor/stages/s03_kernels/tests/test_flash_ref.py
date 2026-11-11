import pytest
import torch

from stages.s01_transformer.code.attention import naive_attention
from stages.s03_kernels.code.flash_attention_ref import flash_attention_ref, hbm_bytes_flash, hbm_bytes_naive


# --8<-- [start:verify]
# 两个 parametrize 叠加 = 4 × 2 = 8 组用例。T 的取值各有用意：
# 1 = decode 只有一个 query；37 和 200 不是块大小的整数倍，测最后一个残块；64 正好整除
@pytest.mark.parametrize("t", [1, 37, 64, 200])
@pytest.mark.parametrize("causal", [True, False])
def test_flash_ref_matches_naive(t, causal):
    # 固定随机种子，失败时能复现
    torch.manual_seed(0)
    # B=2、H=3、D=16 的小张量，FP32、CPU 上就能跑
    q, k, v = (torch.randn(2, 3, t, 16) for _ in range(3))  # 各 [2, 3, t, 16]
    # block_m=32 ≠ block_n=16：故意让 Q 块和 K/V 块不一样大，检验对角线 mask 的下标逻辑
    # assert_close：逐元素比较，|差| ≤ atol + rtol·|期望值| 才算通过
    torch.testing.assert_close(flash_attention_ref(q, k, v, causal, block_m=32, block_n=16),
                               naive_attention(q, k, v, causal), atol=1e-5, rtol=1e-4)


def test_flash_ref_handles_large_logits_stably():
    # online softmax 必须先减最大值；分数很大时 naive exp 会溢出
    # q、k 各乘 30 后，除以 sqrt(D) 的分数标准差约 30² = 900；exp(900) 远超 FP32 上限（约 exp(88)）
    # 对照的 naive_attention 用的是 torch 的 softmax，它内部也减了最大值，所以能当真值
    q, k, v = (torch.randn(1, 1, 64, 8) * 30 for _ in range(3))  # 各 [1, 1, 64, 8]
    out = flash_attention_ref(q, k, v, block_m=16, block_n=16)
    # isfinite：没有 inf 和 nan
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, naive_attention(q, k, v), atol=1e-4, rtol=1e-3)
# --8<-- [end:verify]


def test_flash_saves_hbm_traffic_at_long_t():
    naive = hbm_bytes_naive(1, 32, 8192, 128)
    flash = hbm_bytes_flash(1, 32, 8192, 128)
    assert naive / flash > 5
    # 更关键的是显存占用：naive 要物化 H·T² 的分数矩阵，flash 只需要 O(T·D)
    assert 32 * 8192 * 8192 * 2 > 4e9
