"""讲义 GPU 验证清单的每一项：在哪个解释器里跑什么命令、需要什么环境、对应哪篇讲义。

解释器：
- main：labkit 主环境（torch / triton 和其他课程一致），用 `uv run --with` 临时叠加讲义脚本要的 tabulate、matplotlib、ninja，
  不改 uv.lock；
- vllm：vllm-env 独立环境（L09 同一套）。
命令里的 {hw} 换成参数 hw；工作目录是 vendor/ 的一份拷贝，所以 `-m stages…` 的导入路径和私有仓库一致。
"""

from __future__ import annotations

S3 = "stages.s03_kernels"

# --8<-- [start:items]
ITEMS: dict[str, dict] = {
    "env_probe": {"title": "环境探测：torch / triton / nvcc / ncu / vLLM 是否可用", "lesson": "—", "steps": [], "needs": []},
    "gpu_check": {"title": "实测矩阵乘峰值和 HBM 带宽（GLOSSARY 的“实测可达”）", "lesson": "S2.1",
                  "steps": [("main", ["-m", "tools.gpu_check"])], "needs": ["cuda"]},
    "decode_bench": {"title": "S2 decode 的 MBU 和 batch 拐点（自写 GPT，llama3_8b 形状）", "lesson": "S2.2",
                     "steps": [("main", ["-m", "stages.s02_perf.code.decode_bench", "--preset", "llama3_8b", "--hw", "{hw}",
                                         "--context", "4096", "--batches", "1", "4", "16", "32", "64", "128"])], "needs": ["cuda"]},
    "s3_tests": {"title": "S3 全部 Triton / CUDA kernel 的正确性测试", "lesson": "S3.1-S3.9",
                 "steps": [("main", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "stages/s03_kernels/tests"])], "needs": ["cuda"]},
    "triton_bench": {"title": "S3 第 1-3 课：Triton kernel 对照 PyTorch", "lesson": "S3.1-S3.3",
                     "steps": [("main", ["-m", f"{S3}.triton_kernels.bench_triton", "--hw", "{hw}"])], "needs": ["cuda"]},
    "coalescing": {"title": "S3 第 4 课：访存合并，步长和随机 gather 的带宽", "lesson": "S3.4",
                   "steps": [("main", ["-m", f"{S3}.triton_kernels.coalescing", "--hw", "{hw}"])], "needs": ["cuda"]},
    "cuda_matmul": {"title": "S3 第 5 课：手写 CUDA matmul v1-v5 对照 cuBLAS", "lesson": "S3.5",
                    "steps": [("main", ["-m", f"{S3}.cuda.bench_matmul", "--hw", "{hw}"])], "needs": ["cuda", "nvcc"]},
    "ncu": {"title": "S3 第 6 课：Nsight Compute 采集 v1-v5 的关键指标并自动诊断", "lesson": "S3.6",
            "steps": [("main", ["-m", f"{S3}.cuda.ncu_profile", "--hw", "{hw}"])], "needs": ["cuda", "nvcc", "ncu"]},
    "ncu_parse": {"title": "S3 第 6 课：解析讲义附带的典型值 CSV（不需要 GPU，冒烟用）", "lesson": "S3.6",
                  "steps": [("main", ["-m", f"{S3}.cuda.ncu_profile", "--parse", "stages/s03_kernels/experiments/ncu_typical_values.csv"])],
                  "needs": []},
    "split_kv": {"title": "S3 第 7 课：split-KV decode attention，校准 Little 定律模型", "lesson": "S3.7",
                 "steps": [("main", ["-m", f"{S3}.triton_kernels.bench_split_kv", "--hw", "{hw}"])], "needs": ["cuda"]},
    "w4a16": {"title": "S3 第 8 课：W4A16 GEMM，提速消失的 M", "lesson": "S3.8",
              "steps": [("main", ["-m", f"{S3}.triton_kernels.bench_w4a16", "--hw", "{hw}"])], "needs": ["cuda"]},
    "moe": {"title": "S3 第 9 课：fused MoE 对照 roofline，BLOCK_M 在 decode / prefill 下的差别", "lesson": "S3.9",
            "steps": [("main", ["-m", f"{S3}.triton_kernels.bench_moe", "--hw", "{hw}"])], "needs": ["cuda"]},
    "vllm_facts": {"title": "S5 第 5 课：从 vLLM 源码核对前缀哈希登记时机、投机草稿 token、接受长度指标名", "lesson": "S5.5",
                   "steps": [("vllm", ["@vllm_facts.py"])], "needs": ["vllm"]},
    "s7_vllm": {"title": "S7 第 3 课：TIGER 模型导出到 vLLM，trie 约束 beam search 对照自研引擎", "lesson": "S7.3",
                "steps": [("main", ["-m", "stages.s06_recsys.code.train_genrec", "--only", "tiger"]),
                          ("main", ["-m", "stages.s07_capstone.code.vllm_genrec.export_hf"]),
                          ("vllm", ["-m", "stages.s07_capstone.code.vllm_genrec.constrained_beam", "--users", "512", "--beam", "20"])],
                "needs": ["cuda", "vllm"]},
}
# --8<-- [end:items]

MAIN_WITH = ["tabulate", "matplotlib", "ninja"]
TEXT_SUFFIXES = {".md", ".csv", ".txt", ".json"}
