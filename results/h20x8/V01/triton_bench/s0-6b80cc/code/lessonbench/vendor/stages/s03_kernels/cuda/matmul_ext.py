"""把 matmul_kernels.cu 编译成 Python 可调用的扩展（需要 NVIDIA GPU + nvcc；Mac 上只能读）。

    from stages.s03_kernels.cuda.matmul_ext import matmul
    c = matmul(a, b, "v4")       # a: [M, K] FP32, b: [K, N] FP32 -> [M, N] FP32
    c = matmul(a16, b16, "v5")   # FP16 输入 -> FP32 输出（Tensor Core）

第一次调用会编译（约 1 分钟），结果缓存在 ~/.cache/torch_extensions/，源码不变就不会重编。
"""

import functools
import os
from pathlib import Path

import torch

CU_PATH = Path(__file__).with_name("matmul_kernels.cu")

# --8<-- [start:load_inline]
# load_inline 需要一份 C++ 声明：它据此生成 pybind11 绑定，让 Python 能直接调用 .cu 里定义的主机函数
CPP_DECLS = "\n".join(f"torch::Tensor matmul_{v}(torch::Tensor a, torch::Tensor b);" for v in ["v1", "v2", "v3", "v4", "v5"])

# 版本 -> (扩展里的函数名, 输入 dtype)；v5 是 FP16 输入，其余 FP32
VERSIONS = {
    "v1": ("matmul_v1", torch.float32),
    "v2": ("matmul_v2", torch.float32),
    "v3": ("matmul_v3", torch.float32),
    "v4": ("matmul_v4", torch.float32),
    "v5": ("matmul_v5", torch.float16),
}


def nvcc_available() -> bool:
    """能不能编译：要有 CUDA GPU，且找得到 CUDA 工具链（CUDA_HOME 或 PATH 里的 nvcc）。"""
    from torch.utils.cpp_extension import CUDA_HOME

    return torch.cuda.is_available() and CUDA_HOME is not None


# lru_cache：整个进程只编译、加载一次；想看编译输出就先设环境变量 S3_CUDA_VERBOSE=1
@functools.lru_cache(maxsize=None)
def load():
    # 延迟导入：Mac 上 import 本模块不会触发任何 CUDA 相关的东西
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="s3_cuda_matmul",
        # cpp_sources 前面会自动加 #include <torch/extension.h>；cuda_sources 交给 nvcc 编译
        cpp_sources=CPP_DECLS,
        cuda_sources=CU_PATH.read_text(),
        functions=[name for name, _ in VERSIONS.values()],
        # -O3 开优化；-Xptxas -v 让 ptxas 打印每个 kernel 用了多少寄存器、多少 smem、有没有 spill
        # （verbose 时能看到 "Used 128 registers, 8192 bytes smem"，可以代入 tools/occupancy.py）
        extra_cuda_cflags=["-O3", "-Xptxas", "-v"],
        verbose=os.environ.get("S3_CUDA_VERBOSE") == "1",
    )
# --8<-- [end:load_inline]


def matmul(a: torch.Tensor, b: torch.Tensor, version: str) -> torch.Tensor:
    """a: [M, K] @ b: [K, N] -> [M, N] FP32。输入会被转成该版本要求的 dtype 并变成连续的。"""
    name, dtype = VERSIONS[version]
    fn = getattr(load(), name)
    return fn(a.to(dtype).contiguous(), b.to(dtype).contiguous())
