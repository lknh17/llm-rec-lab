import pytest
import torch

from stages.s03_kernels.code.w4a16_ref import (
    fp8_quant,
    outlier_activation,
    pack_int4,
    rel_err,
    scales_for_kernel,
    unpack_int4,
    w4a16_matmul_blocked,
    w4a16_matmul_ref,
    w4a16_model,
)
from stages.s05_inference.code.quant import dequantize, quantize


# --8<-- [start:verify]
def test_pack_unpack_roundtrip_covers_all_nibbles():
    # 所有 16 个取值都出现在偶数行和奇数行，确认高低半字节和符号处理都对
    q = torch.arange(-8, 8, dtype=torch.int8).repeat(3, 2)  # [3, 32]
    q = torch.cat([q, q.flip(1)])  # [6, 32]
    packed = pack_int4(q)
    assert packed.dtype == torch.uint8 and packed.shape == (16, 6)
    assert torch.equal(unpack_int4(packed), q)


@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("m", [1, 7, 64])
def test_w4a16_gemm_matches_dequantized_fp32(group_size, m):
    torch.manual_seed(0)
    w = torch.randn(96, 256) * 0.02
    x = torch.randn(m, 256)
    q, s = quantize(w, 4, group_size)
    packed, sk = pack_int4(q), scales_for_kernel(s, torch.float32)
    expect = x @ dequantize(q, s).T  # 标准答案：quant.py 反量化后的 FP32 GEMM
    torch.testing.assert_close(w4a16_matmul_ref(x, packed, sk, group_size), expect, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(w4a16_matmul_blocked(x, packed, sk, group_size, block_k=64), expect,
                               atol=1e-5, rtol=1e-4)
# --8<-- [end:verify]


def test_smaller_group_is_more_accurate():
    torch.manual_seed(0)
    w = torch.randn(256, 1024) * 0.02
    w[torch.rand_like(w) < 0.001] *= 20  # 少量离群值
    errs = [rel_err(dequantize(*quantize(w, 4, g)), w) for g in (None, 128, 32)]
    assert errs[0] > errs[1] > errs[2]


def test_model_speedup_near_4x_at_m1_and_vanishes_at_large_m():
    for hw in ["H100", "H20"]:
        small = w4a16_model(1, 37888, 3584, hw)
        assert 3.5 < small["speedup"] < 4.0
        assert w4a16_model(512, 37888, 3584, hw)["speedup"] <= 1.0
    # H20 的拐点低（37），提速比 H100 更早消失
    assert w4a16_model(32, 37888, 3584, "H20")["speedup"] < w4a16_model(32, 37888, 3584, "H100")["speedup"]


def test_fp8_block_scaling_isolates_outlier_channels():
    # 离群通道放大 10 万倍：per-tensor 时普通通道掉进 E4M3 的次正规数区间，精度崩掉；1x128 块把离群值隔离在自己的块里
    x = outlier_activation(128, 1024, mag=1e5)
    normal = x.abs().mean(0) < 5
    e_tensor = rel_err(fp8_quant(x, "tensor")[:, normal], x[:, normal])
    e_block = rel_err(fp8_quant(x, "block1d")[:, normal], x[:, normal])
    assert e_block < e_tensor / 3
    # 128x128 块量化：形状不变，误差在 E4M3 的量级（3 位尾数，约 2^-4）
    w = torch.randn(256, 256)
    assert rel_err(fp8_quant(w, "block2d"), w) < 0.05
