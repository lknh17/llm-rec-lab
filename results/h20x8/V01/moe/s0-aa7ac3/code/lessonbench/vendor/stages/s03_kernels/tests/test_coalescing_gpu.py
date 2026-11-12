"""合并访存演示 kernel 的正确性测试：只在有 NVIDIA GPU 且安装了 triton 的机器上运行，Mac 上自动跳过。"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("需要 CUDA GPU", allow_module_level=True)

from stages.s03_kernels.triton_kernels.coalescing import gather_rows, strided_copy  # noqa: E402

DEV = "cuda"


@pytest.mark.parametrize("stride", [1, 2, 7, 32])
@pytest.mark.parametrize("n", [1, 1000, 100_003])
def test_strided_copy(n, stride):
    src = torch.randn((n - 1) * stride + 1, device=DEV)
    torch.testing.assert_close(strided_copy(src, n, stride), src[::stride][:n])


@pytest.mark.parametrize("d", [1, 4, 100, 256])
def test_gather_rows_matches_index_select(d):
    table = torch.randn(5000, d, device=DEV)
    idx = torch.randint(0, 5000, (777,), device=DEV)
    torch.testing.assert_close(gather_rows(table, idx), table[idx])
