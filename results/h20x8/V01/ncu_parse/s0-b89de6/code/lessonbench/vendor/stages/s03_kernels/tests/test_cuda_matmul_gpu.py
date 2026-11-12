"""CUDA 矩阵乘 v1-v5 的正确性测试：需要 NVIDIA GPU 和 nvcc，Mac 上自动跳过。"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("需要 CUDA GPU", allow_module_level=True)

from stages.s03_kernels.cuda.matmul_ext import matmul, nvcc_available  # noqa: E402

if not nvcc_available():
    pytest.skip("需要 nvcc（CUDA_HOME）", allow_module_level=True)

DEV = "cuda"


# 包含整除和不整除 tile 大小的形状：(257, 511, 129) 覆盖 v1-v4 的边界检查
@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (32, 32, 32), (128, 256, 64), (257, 511, 129), (1024, 1024, 1024)])
def test_fp32_versions_match_torch(version, shape):
    m, n, k = shape
    a = torch.randn(m, k, device=DEV)
    b = torch.randn(k, n, device=DEV)
    torch.backends.cuda.matmul.allow_tf32 = False
    # FP32 累加顺序不同，误差随 K 增长；1e-3 量级足够区分"对"和"错"
    torch.testing.assert_close(matmul(a, b, version), a @ b, atol=1e-3 * k**0.5, rtol=1e-4)


# v5 要求 16 的倍数；(48, 80, 32) 不是 block tile 64 的倍数，覆盖 warp 级越界跳过
@pytest.mark.parametrize("shape", [(16, 16, 16), (48, 80, 32), (256, 256, 256), (1024, 2048, 512)])
def test_wmma_matches_fp32_reference(shape):
    m, n, k = shape
    a = torch.randn(m, k, device=DEV, dtype=torch.float16)
    b = torch.randn(k, n, device=DEV, dtype=torch.float16)
    out = matmul(a, b, "v5")
    assert out.dtype == torch.float32
    # 参考：同样的 FP16 输入，用 FP32 做矩阵乘；Tensor Core 内部 FP32 累加，误差应该很小
    torch.testing.assert_close(out, a.float() @ b.float(), atol=1e-2, rtol=1e-3)


def test_wmma_rejects_unaligned_shapes():
    a = torch.randn(17, 16, device=DEV, dtype=torch.float16)
    b = torch.randn(16, 16, device=DEV, dtype=torch.float16)
    with pytest.raises(RuntimeError):
        matmul(a, b, "v5")
