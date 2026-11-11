"""Triton kernel 的正确性测试：只在有 NVIDIA GPU 且安装了 triton 的机器上运行，Mac 上自动跳过。"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("需要 CUDA GPU", allow_module_level=True)

from stages.s01_transformer.code.attention import naive_attention  # noqa: E402
from stages.s03_kernels.triton_kernels.flash_attn import flash_attention  # noqa: E402
from stages.s03_kernels.triton_kernels.fused_softmax import softmax  # noqa: E402
from stages.s03_kernels.triton_kernels.matmul import matmul  # noqa: E402
from stages.s03_kernels.triton_kernels.vector_add import add  # noqa: E402

DEV = "cuda"


def test_vector_add():
    x, y = torch.randn(100_003, device=DEV), torch.randn(100_003, device=DEV)
    torch.testing.assert_close(add(x, y), x + y)


@pytest.mark.parametrize("cols", [7, 1000, 4096])
def test_softmax(cols):
    x = torch.randn(33, cols, device=DEV, dtype=torch.float16)
    torch.testing.assert_close(softmax(x), torch.softmax(x.float(), -1).half(), atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("shape", [(128, 128, 128), (257, 511, 129), (1024, 2048, 512)])
def test_matmul(shape):
    m, n, k = shape
    a = torch.randn(m, k, device=DEV, dtype=torch.float16)
    b = torch.randn(k, n, device=DEV, dtype=torch.float16)
    torch.testing.assert_close(matmul(a, b), a @ b, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("t", [1, 63, 128, 1000])
@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention(t, causal):
    q, k, v = (torch.randn(2, 4, t, 64, device=DEV, dtype=torch.float16) for _ in range(3))
    ref = naive_attention(q.float(), k.float(), v.float(), causal).half()
    torch.testing.assert_close(flash_attention(q, k, v, causal), ref, atol=2e-2, rtol=2e-2)
