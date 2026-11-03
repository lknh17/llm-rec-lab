"""广告元素抽取项目的全部可调参数。"""

from labkit.params import Param, ParamSpace

PARAMS = ParamSpace([
    # ---------- 数据 ----------
    Param("train_size", "int", 400, "教学合成训练样本数", affects="data", min=8, max=20000,
          cost_note="训练 token 数和训练时间约与样本数成正比"),
    Param("hard_ratio", "float", 0.15, "训练集中困难样本（多优惠、远距离条件、无品类、否定句）的比例",
          affects="data", min=0.0, max=1.0, cost_note="不改样本数时训练 token 基本不变；困难样本略长"),
    Param("exclusion_ratio", "float", 0.0, "把多大比例的训练样本替换成排除型条件样本（'活动款除外''联名款不参与'）",
          affects="data", min=0.0, max=0.5,
          cost_note="替换而非追加：样本数不变，其余样本逐条相同；训练 token 基本不变"),
    Param("data_seed", "int", 0, "合成训练数据的随机种子", affects="data"),
    Param("prompt_mode", "choice", "zero_shot", "提示词形式", affects="data", choices=("zero_shot", "few_shot"),
          choice_notes={"few_shot": "系统提示后附 2 个示例，每条样本多约 180 个 prompt token"}),
    # ---------- 监督与模板 ----------
    Param("supervision", "choice", "assistant_only", "哪些 token 参与 loss", affects="data",
          choices=("assistant_only", "all_tokens"),
          choice_notes={"assistant_only": "只监督 assistant 回复（JSON + <|im_end|>）",
                        "all_tokens": "整段对话（含系统提示和广告原文）都算 loss"}),
    Param("max_len", "int", 512, "单条样本最大 token 数，超过则截断", affects="data", min=32, max=4096),
    Param("truncation", "choice", "drop_long", "样本超过 max_len 时怎么处理", affects="data",
          choices=("drop_long", "right"),
          choice_notes={"drop_long": "整条丢弃并计数，保证留下的样本都有完整回答",
                        "right": "直接截掉尾部。回答在尾部，最先被截掉的就是监督目标"}),
    Param("infer_template", "choice", "chat", "推理时的输入格式", affects="infer", choices=("chat", "plain"),
          choice_notes={"chat": "与训练一致：Qwen 聊天模板 + 系统提示",
                        "plain": "纯文本拼接（'广告：...\\n输出：'），与训练格式不一致"}),
    # ---------- 模型与训练 ----------
    Param("model", "choice", "Qwen/Qwen2.5-0.5B-Instruct", "基础模型", affects="model",
          choices=("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
          choice_notes={"Qwen/Qwen2.5-1.5B-Instruct": "建议在 GPU 环境运行；Mac 上训练约慢 3 倍",
                        "Qwen/Qwen2.5-7B-Instruct": "只在 GPU 上跑：bf16 权重约 15 GB，单卡 H20（96 GB）LoRA 绰绰有余；"
                                                    "和 0.5B 共用同一套 tokenizer，prompt 长度完全相同"},
          cost_note="训练 FLOPs ≈ 6 × 参数量 × token 数：7B 是 0.5B 的约 15 倍"),
    Param("train_mode", "choice", "lora", "训练方式", affects="model", choices=("none", "lora"),
          choice_notes={"none": "不训练，直接评估基础模型（基线）", "lora": "LoRA 微调，基础权重冻结"}),
    Param("lora_r", "int", 16, "LoRA 秩", affects="model", min=1, max=256,
          cost_note="可训练参数约与 r 成正比"),
    Param("lora_alpha", "int", 32, "LoRA 缩放 alpha（实际缩放 = alpha / r）", affects="model", min=1),
    Param("lora_targets", "choice", "attn_mlp", "LoRA 挂在哪些线性层", affects="model", choices=("attn", "attn_mlp")),
    Param("lr", "float", 2e-4, "学习率（峰值）", affects="train", min=1e-7, max=1e-1),
    Param("epochs", "int", 2, "训练轮数", affects="train", min=1, max=50),
    Param("batch_size", "int", 8, "每步样本数", affects="train", min=1, max=256),
    Param("max_steps", "int", 0, "最多训练步数，0 表示不限", affects="train", min=0),
    Param("seed", "int", 0, "训练随机种子（LoRA 初始化、打乱顺序）", affects="train"),
    Param("eval_each_epoch", "bool", False, "每轮结束都在 dev 上做一次生成评估（观察 loss 与任务指标的关系）",
          affects="eval", cost_note="每次 dev 生成评估在 Mac 上约 30-60 秒"),
    # ---------- 评估 ----------
    Param("max_new_tokens", "int", 160, "生成的最大新 token 数", affects="infer", min=16, max=1024),
    Param("final_test", "bool", False, "是否在保留测试集上评估（阶段性最终检验时才打开）", affects="eval"),
    # ---------- 运行 ----------
    Param("device", "choice", "auto", "运行设备", affects="runtime", choices=("auto", "cpu", "mps", "cuda")),
    Param("max_minutes", "float", 30.0, "时间预算（分钟）", affects="runtime", min=1.0),
    Param("grad_checkpointing", "bool", False, "梯度检查点：不存中间激活，反向时重算", affects="runtime",
          advanced=True, cost_note="激活显存降到约 1/层数，训练时间多约 30%（多一次前向）；数值结果不变"),
])
