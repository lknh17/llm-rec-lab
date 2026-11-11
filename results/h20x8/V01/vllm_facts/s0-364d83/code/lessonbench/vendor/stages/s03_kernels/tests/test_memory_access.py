"""访存模型（合并访存、bank conflict）的测试：全部是可以手算的例子。"""

import pytest

from stages.s03_kernels.code.memory_access import (
    bank_conflict_ways,
    coalescing_efficiency,
    strided_traffic,
    tile_column_words,
    warp_sectors,
)


def test_contiguous_fp32_warp_is_4_sectors():
    # 32 线程 × 4 字节 = 128 字节 = 4 个 32 字节 sector
    addrs = [4 * i for i in range(32)]
    assert warp_sectors(addrs) == 4
    assert coalescing_efficiency(addrs) == pytest.approx(1.0)


def test_misaligned_contiguous_touches_one_extra_sector():
    # 起始地址偏 4 字节：128 字节跨了 5 个 sector
    assert warp_sectors([4 + 4 * i for i in range(32)]) == 5


@pytest.mark.parametrize("stride, sectors", [(1, 4), (2, 8), (4, 16), (8, 32), (64, 32)])
def test_strided_warp_sectors(stride, sectors):
    assert warp_sectors([4 * stride * i for i in range(32)]) == sectors


def test_broadcast_is_one_sector():
    assert warp_sectors([128] * 32) == 1


def test_strided_traffic_saturates_at_sector_size():
    n = 1000
    assert strided_traffic(n, 1) == 4 * n
    assert strided_traffic(n, 4) == 16 * n
    # FP32 步长 >= 8 后每个元素独占 32 字节 sector，不再变差
    assert strided_traffic(n, 8) == strided_traffic(n, 1024) == 32 * n
    # FP16 一个 sector 装 16 个，步长 16 才饱和
    assert strided_traffic(n, 8, elem_bytes=2) == 16 * n
    assert strided_traffic(n, 16, elem_bytes=2) == strided_traffic(n, 64, elem_bytes=2) == 32 * n


def test_bank_conflicts_column_padding_swizzle():
    assert bank_conflict_ways(tile_column_words(32)) == 32
    assert bank_conflict_ways(tile_column_words(33)) == 1
    assert bank_conflict_ways(tile_column_words(32, swizzle=True)) == 1
    # 64 列同样是 32 的倍数，padding 到 65 才能消除
    assert bank_conflict_ways(tile_column_words(64)) == 32
    assert bank_conflict_ways(tile_column_words(65)) == 1


def test_bank_broadcast_and_two_way():
    assert bank_conflict_ways([7] * 32) == 1
    # 步长 2 个 word：16 个 bank 各有 2 个不同 word -> 2 路冲突
    assert bank_conflict_ways([2 * i for i in range(32)]) == 2


def test_v4_regn_pattern_has_4_way_conflict():
    # 第 5 课 v4 的 Bs 读取：线程列号 tc = tid % 16，读 word tc·8 + i（标量读，i 固定）
    # tc = 0, 4, 8, 12 都落在同一个 bank -> 4 路冲突（tid 16-31 读同样的地址，是广播）
    words = [(tid % 16) * 8 for tid in range(32)]
    assert bank_conflict_ways(words) == 4
