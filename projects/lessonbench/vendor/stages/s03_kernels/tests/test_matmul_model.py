"""CUDA 矩阵乘访存模型的测试：每个数都能手算。"""

import pytest

from stages.s03_kernels.code.matmul_model import (
    VERSIONS,
    global_bytes,
    global_reads_per_output,
    n_blocks,
    roofline_cap_tflops,
    sector_bytes_v1_v2,
)

V = {v.name: v for v in VERSIONS}
S = 4096


def test_reads_per_output():
    # v1/v2：A 一行 + B 一列 = 2K；v3：2K/32；v4：2K/128
    assert global_reads_per_output(V["v1"], S) == 2 * S
    assert global_reads_per_output(V["v2"], S) == 2 * S
    assert global_reads_per_output(V["v3"], S) == S / 16
    assert global_reads_per_output(V["v4"], S) == S / 64


def test_coalescing_ratio_v1_vs_v2():
    # v1 每个乘加 33 字节 sector 流量，v2 只有 5 字节：同样的指令数，搬运量差 6.6 倍
    assert sector_bytes_v1_v2(False) == 33
    assert sector_bytes_v1_v2(True) == 5
    ratio = global_bytes(V["v1"], S, S, S) / global_bytes(V["v2"], S, S, S)
    assert ratio == pytest.approx(6.6, rel=1e-3)


def test_tiled_intensity_matches_bm_bn_formula():
    # 方块 tile 的强度 ≈ 2·BM·BN / (elem·(BM+BN))：FP32 32x32 -> 8，128x128 -> 32；FP16 32x32 -> 16
    assert roofline_cap_tflops(V["v3"], S, S, S)[0] == pytest.approx(8, rel=0.01)
    assert roofline_cap_tflops(V["v4"], S, S, S)[0] == pytest.approx(32, rel=0.02)
    assert roofline_cap_tflops(V["v5"], S, S, S)[0] == pytest.approx(16, rel=0.01)


def test_h100_bounds():
    # v1-v3 和 v5 在无缓存模型下都是带宽瓶颈；v4 的强度 32 已超过 FP32 的拐点 67/3.35 = 20
    assert roofline_cap_tflops(V["v1"], S, S, S)[2] == "带宽"
    assert roofline_cap_tflops(V["v4"], S, S, S)[1:] == (67.0, "算力")
    # v5 用了 Tensor Core 却被带宽卡在 53 TFLOPS，只有 989 的 5%
    intensity, cap, bound = roofline_cap_tflops(V["v5"], S, S, S)
    assert bound == "带宽" and cap == pytest.approx(16 * 3.35, rel=0.01)


def test_block_counts():
    assert n_blocks(V["v1"], S, S) == 128 * 128
    assert n_blocks(V["v4"], S, S) == 32 * 32
    # decode 形状 M=16：v4 只有 32 个 block，132 个 SM 用不满
    assert n_blocks(V["v4"], 16, S) == 32
