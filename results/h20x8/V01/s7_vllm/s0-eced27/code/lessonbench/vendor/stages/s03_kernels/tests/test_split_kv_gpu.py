"""Triton paged split-KV decode 的正确性测试：只在有 NVIDIA GPU 且安装了 triton 的机器上运行，Mac 上自动跳过。"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("需要 CUDA GPU", allow_module_level=True)

from stages.s03_kernels.code.split_kv_decode import build_paged_cache, sdpa_decode  # noqa: E402
from stages.s03_kernels.triton_kernels.split_kv_decode import contiguous_to_paged, split_kv_decode  # noqa: E402

DEV = "cuda"


@pytest.mark.parametrize("h,hkv", [(8, 8), (28, 4), (32, 8)])
@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("num_splits", [1, 3, 16, None])
def test_split_kv_ragged(h, hkv, page, num_splits):
    torch.manual_seed(0)
    seq_lens = [1, 17, 300, 1000]
    b, d = len(seq_lens), 128
    q = torch.randn(b, h, d, device=DEV, dtype=torch.float16)
    k = torch.randn(b, hkv, max(seq_lens), d, device=DEV, dtype=torch.float16)
    v = torch.randn(b, hkv, max(seq_lens), d, device=DEV, dtype=torch.float16)
    k_pool, v_pool, table = build_paged_cache(k, v, seq_lens, page)
    lens = torch.tensor(seq_lens, device=DEV, dtype=torch.int32)
    out = split_kv_decode(q, k_pool, v_pool, table, lens, num_splits)
    for i, n in enumerate(seq_lens):
        ref = sdpa_decode(q[i : i + 1].float(), k[i : i + 1, :, :n].float(), v[i : i + 1, :, :n].float()).half()
        torch.testing.assert_close(out[i : i + 1], ref, atol=2e-3, rtol=2e-3)


def test_contiguous_to_paged_matches():
    torch.manual_seed(1)
    b, h, hkv, t, d = 2, 28, 4, 4096, 128
    q = torch.randn(b, h, d, device=DEV, dtype=torch.float16)
    k = torch.randn(b, hkv, t, d, device=DEV, dtype=torch.float16)
    v = torch.randn(b, hkv, t, d, device=DEV, dtype=torch.float16)
    k_pool, v_pool, table = contiguous_to_paged(k, v, 16)
    lens = torch.full((b,), t, device=DEV, dtype=torch.int32)
    ref = sdpa_decode(q.float(), k.float(), v.float()).half()
    torch.testing.assert_close(split_kv_decode(q, k_pool, v_pool, table, lens, 32), ref, atol=2e-3, rtol=2e-3)
