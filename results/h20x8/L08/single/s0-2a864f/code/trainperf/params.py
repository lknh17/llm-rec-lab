"""训练效率与 8 卡并行（L08）的全部参数。默认值 = 基线：Qwen2.5-1.5B 形状、DDP 8 卡、seq 2048、mbs 2、bf16、SDPA。"""

from labkit.params import Param, ParamSpace

PARAMS = ParamSpace([
    # ---------- 模型 ----------
    Param("preset", "choice", "qwen1.5b", "模型形状（随机初始化，只看性能）", affects="model",
          choices=("tiny", "qwen0.5b", "qwen1.5b", "qwen7b"),
          choice_notes={"tiny": "2 层 d=128（冒烟用）", "qwen0.5b": "Qwen2.5-0.5B 形状（4.9 亿参数）",
                        "qwen1.5b": "Qwen2.5-1.5B 形状（15.4 亿参数）", "qwen7b": "Qwen2.5-7B 形状（76 亿参数）"}),
    Param("seq_len", "int", 2048, "序列长度 T", affects="model", min=16, max=131072),
    Param("attn", "choice", "sdpa", "attention 实现", affects="model", choices=("sdpa", "naive"),
          choice_notes={"sdpa": "PyTorch SDPA（flash）", "naive": "显式 [B, H, T, T] 打分 + softmax"}),
    # ---------- 并行与训练 ----------
    Param("strategy", "choice", "ddp", "并行策略", affects="train", choices=("none", "ddp", "fsdp_full", "fsdp_grad_op"),
          choice_notes={"none": "单卡", "ddp": "DDP：每卡一份完整的参数、梯度、优化器状态",
                        "fsdp_full": "FSDP2 FULL_SHARD（≈ ZeRO-3）：参数、梯度、优化器状态都切成 N 份，前向后立刻释放",
                        "fsdp_grad_op": "FSDP2 reshard_after_forward=False（≈ ZeRO-2）：前向后不释放整份参数"}),
    Param("world_size", "int", 8, "卡数", affects="runtime", min=1, max=64),
    Param("backend", "choice", "auto", "通信后端", affects="runtime", choices=("auto", "nccl", "gloo")),
    Param("mbs", "int", 2, "每卡 micro batch（条序列）", affects="train", min=1, max=1024),
    Param("grad_accum", "int", 1, "梯度累积步数", affects="train", min=1, max=1024),
    Param("no_sync", "bool", True, "DDP 梯度累积时，非最后一个 micro step 用 no_sync 跳过 all-reduce", affects="train"),
    Param("precision", "choice", "bf16", "精度", affects="train", choices=("bf16", "fp32"),
          choice_notes={"bf16": "fp32 主参数 + bf16 autocast（FSDP：参数 bf16、梯度归约 fp32）", "fp32": "全部 fp32（不开 TF32）"}),
    Param("ckpt", "choice", "none", "激活重计算", affects="train", choices=("none", "full"),
          choice_notes={"none": "不重计算", "full": "每个 Transformer 层只存输入，反向时重算"}),
    Param("compile", "bool", False, "torch.compile 每个层", affects="train"),
    Param("adam", "choice", "fused", "AdamW 实现", affects="train", choices=("fused", "foreach", "forloop")),
    Param("lr", "float", 1e-4, "学习率", affects="train", min=1e-8, max=1.0),
    # ---------- 数据 ----------
    Param("data", "choice", "gpu_random", "数据来源", affects="data", choices=("gpu_random", "cpu_pipeline"),
          choice_notes={"gpu_random": "直接在 GPU 上生成随机 token（数据成本为 0）",
                        "cpu_pipeline": "每条样本在 CPU 上做纯 Python 规范化、贪心分词和在线 BPE 合并，再走 DataLoader"}),
    Param("num_workers", "int", 0, "DataLoader worker 数（cpu_pipeline）", affects="data", min=0, max=64),
    Param("pin_memory", "bool", False, "锁页内存（cpu_pipeline）", affects="data"),
    Param("prefetch_factor", "int", 2, "每个 worker 预取几批", affects="data", min=1, max=64),
    Param("prep_passes", "int", 400, "每条样本在线 BPE 合并几轮（控制 CPU 预处理的工作量）", affects="data", min=0, max=100000),
    # ---------- 测量 ----------
    Param("steps", "int", 30, "总步数（前 5 步热身、接着 15 步测吞吐、其余逐阶段计时；至少 24）", affects="train", min=24, max=100000),
    Param("profile_steps", "int", 3, "最后用 torch.profiler 采几步", affects="eval", min=0, max=100),
    Param("bench_hw", "bool", True, "测本机 GEMM 算力、all-reduce 带宽和 HBM 拷贝带宽", affects="eval"),
    Param("peak_tflops", "float", 148.0, "bf16 稠密算力标称值（H20）", affects="eval", min=0.1, max=100000.0),
    Param("seed", "int", 0, "种子", affects="train", min=0, max=1_000_000),
    Param("device", "choice", "auto", "设备", affects="runtime", choices=("auto", "cuda", "cpu")),
    Param("max_minutes", "int", 30, "超时（分钟）", affects="runtime", min=1, max=1440),
])
