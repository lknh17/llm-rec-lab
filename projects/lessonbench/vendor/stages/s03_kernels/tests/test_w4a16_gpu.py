"""Triton W4A16 GEMM 的正确性测试：只在有 NVIDIA GPU 且安装了 triton 的机器上运行，Mac 上自动跳过。"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("需要 CUDA GPU", allow_module_level=True)

from stages.s03_kernels.code.w4a16_ref import pack_int4, scales_for_kernel, w4a16_matmul_ref  # noqa: E402
from stages.s03_kernels.triton_kernels.w4a16_gemm import w4a16_matmul  # noqa: E402
from stages.s05_inference.code.quant import dequantize, quantize  # noqa: E402

DEV = "cuda"


@pytest.mark.parametrize("m", [1, 3, 16, 100, 256])
@pytest.mark.parametrize("n,k", [(4096, 4096), (1000, 512), (37888, 3584)])
@pytest.mark.parametrize("group_size", [64, 128])
def test_w4a16_matches_dequantized(m, n, k, group_size):
    torch.manual_seed(0)
    w = torch.randn(n, k, device=DEV) * 0.02
    q, s = quantize(w, 4, group_size)
    packed, sk = pack_int4(q), scales_for_kernel(s)
    x = torch.randn(m, k, device=DEV, dtype=torch.float16)
    # 标准答案：FP32 下 x @ 反量化权重^T
    ref = (x.float() @ dequantize(q, s).T).half()
    out = w4a16_matmul(x, packed, sk, group_size)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w4a16_matmul_ref(x, packed, sk, group_size), ref, atol=2e-2, rtol=2e-2)
