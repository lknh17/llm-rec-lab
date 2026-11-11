"""Triton fused MoE（grouped GEMM）的正确性测试：只在有 NVIDIA GPU 且安装了 triton 的机器上运行，Mac 上自动跳过。"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("需要 CUDA GPU", allow_module_level=True)

from stages.s03_kernels.code.moe_ref import moe_grouped_ref, route, zipf_routing  # noqa: E402
from stages.s03_kernels.triton_kernels.moe_grouped_gemm import fused_moe  # noqa: E402

DEV = "cuda"


@pytest.mark.parametrize("t", [1, 7, 64, 1000])
@pytest.mark.parametrize("e,k", [(8, 2), (128, 8)])
@pytest.mark.parametrize("block_m", [16, 64])
def test_fused_moe_matches_reference(t, e, k, block_m):
    torch.manual_seed(0)
    c, i = 256, 128
    x = torch.randn(t, c, device=DEV, dtype=torch.float16)
    w_gate = torch.randn(e, c, device=DEV)
    w1 = torch.randn(e, 2 * i, c, device=DEV, dtype=torch.float16) * 0.05
    w2 = torch.randn(e, c, i, device=DEV, dtype=torch.float16) * 0.05
    topk_w, topk_ids = route(x, w_gate, k)
    ref = moe_grouped_ref(x.float(), topk_w, topk_ids, w1.float(), w2.float())
    out = fused_moe(x, w1, w2, topk_w, topk_ids, block_m)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_fused_moe_skewed_routing():
    torch.manual_seed(1)
    t, c, i, e, k = 512, 256, 128, 64, 4
    x = torch.randn(t, c, device=DEV, dtype=torch.float16)
    w1 = torch.randn(e, 2 * i, c, device=DEV, dtype=torch.float16) * 0.05
    w2 = torch.randn(e, c, i, device=DEV, dtype=torch.float16) * 0.05
    topk_ids = zipf_routing(t, e, k, 1.5).to(DEV)
    topk_w = torch.softmax(torch.randn(t, k, device=DEV), -1)
    ref = moe_grouped_ref(x.float(), topk_w, topk_ids, w1.float(), w2.float())
    torch.testing.assert_close(fused_moe(x, w1, w2, topk_w, topk_ids).float(), ref, atol=2e-2, rtol=2e-2)
