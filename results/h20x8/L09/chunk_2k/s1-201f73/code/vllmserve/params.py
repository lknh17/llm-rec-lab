"""推理性能与 vLLM 服务（L09）的全部参数。默认值 = 基线：Qwen2.5-7B-Instruct（bf16）、vLLM、每条输入 4096 token、
输出 50 token（长输入短输出的打标）、256 条同时到达、max_num_seqs 32、每步 token 预算 8192、prefix caching 开、没有共享前缀。"""

from labkit.params import Param, ParamSpace

PARAMS = ParamSpace([
    # ---------- 模型与引擎 ----------
    Param("backend", "choice", "vllm", "推理后端", affects="runtime", choices=("vllm", "hf"),
          choice_notes={"vllm": "vLLM（独立环境 vllm-env，子进程运行）",
                        "hf": "transformers 逐条 generate（只用于 Mac 冒烟，验证流程和测量口径；vLLM 专有的旋钮不生效）"}),
    Param("model", "choice", "qwen2.5-7b", "模型", affects="model", choices=("qwen2.5-7b", "qwen2.5-7b-awq", "qwen2.5-0.5b"),
          choice_notes={"qwen2.5-7b": "Qwen2.5-7B-Instruct，bf16 权重 15.2 GB",
                        "qwen2.5-7b-awq": "Qwen2.5-7B-Instruct-AWQ：权重 INT4（W4A16），激活 fp16，vLLM 用 awq_marlin kernel",
                        "qwen2.5-0.5b": "Qwen2.5-0.5B-Instruct（环境探测、Mac 冒烟、投机解码的草稿模型）"}),
    Param("quantization", "choice", "none", "在线量化（加载 bf16 权重后转换）", affects="model", choices=("none", "fp8"),
          choice_notes={"none": "不量化（AWQ 模型自带量化配置，也选 none）",
                        "fp8": "FP8 W8A8：权重按通道、激活按 token 动态量化，Hopper 上走 FP8 Tensor Core（标称算力是 bf16 的 2 倍）"}),
    Param("gpu_memory_utilization", "float", 0.9, "vLLM 可用的显存比例（权重 + 激活 + KV cache）", affects="runtime", min=0.1, max=0.98,
          cost_note="调低 → KV cache 块变少 → 同时能跑的请求变少"),
    Param("max_num_seqs", "int", 32, "同时在跑的请求数上限（decode batch 上限）", affects="runtime", min=1, max=4096),
    Param("max_num_batched_tokens", "int", 8192, "每个调度步的 token 预算（prefill 按它切块，和 decode 拼在同一步）",
          affects="runtime", min=256, max=262144, cost_note="大 → prefill 一口气做完，TTFT 短，但正在 decode 的请求被卡住；小 → 反过来"),
    Param("enable_prefix_caching", "bool", True, "自动前缀缓存（按 KV 块哈希复用相同前缀）", affects="runtime"),
    Param("spec", "choice", "none", "投机解码", affects="runtime", choices=("none", "ngram", "draft"),
          choice_notes={"none": "不投机", "ngram": "n-gram 提示查找（从输入里找后续 token 当草稿，不需要草稿模型）",
                        "draft": "Qwen2.5-0.5B 当草稿模型"}),
    Param("num_speculative_tokens", "int", 4, "每步草稿 token 数 k", affects="runtime", min=1, max=16),
    # ---------- 负载 ----------
    Param("input_len", "int", 4096, "每条输入的 token 数（含共享前缀和结尾的指令）", affects="data", min=32, max=131072),
    Param("output_len", "int", 50, "每条输出的 token 数（忽略 EOS，固定长度）", affects="data", min=1, max=8192),
    Param("num_prompts", "int", 256, "请求条数", affects="data", min=1, max=100000),
    Param("shared_prefix", "int", 0, "所有请求共享的打标规范长度（token）；0 = 没有共享部分", affects="data", min=0, max=131072),
    Param("prefix_position", "choice", "head", "共享的打标规范放在哪", affects="data", choices=("head", "tail"),
          choice_notes={"head": "规范在前、内容在后（前缀相同，缓存能命中）",
                        "tail": "内容在前、规范在后（每条的开头都不同，前缀缓存命中不了）"}),
    Param("request_rate", "float", 0.0, "到达率（条/s，泊松到达）；0 = 全部同时到达（离线吞吐）", affects="data", min=0.0, max=10000.0),
    Param("seed", "int", 0, "种子（决定内容文本和到达时间）", affects="data", min=0, max=1_000_000),
    # ---------- 测量 ----------
    Param("peak_tflops", "float", 148.0, "bf16 稠密标称算力（TFLOPS），屋顶线估算用；H20 = 148", affects="runtime", min=0.1, max=10000.0),
    Param("hbm_gbps", "float", 4000.0, "显存标称带宽（GB/s），屋顶线估算用；H20 = 4000", affects="runtime", min=1.0, max=100000.0),
    Param("device", "choice", "auto", "设备（hf 后端）", affects="runtime", choices=("auto", "cuda", "mps", "cpu")),
    Param("max_minutes", "int", 30, "超时（分钟）", affects="runtime", min=1, max=1440),
])
