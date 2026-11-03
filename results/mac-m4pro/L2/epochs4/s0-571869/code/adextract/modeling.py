"""模型与 tokenizer 的加载、LoRA 挂载、设备与精度选择。"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from labkit.assets import resolve_model_path as _resolve


def resolve_model_path(name: str, must_exist: bool = True) -> str | None:
    """优先 LAB_ASSETS_DIR/models/<name>，其次 HuggingFace 本地缓存。不联网下载——先运行 python -m labkit assets。"""
    return _resolve(name, must_exist)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # CUDA 上用 bf16；MPS 上 bf16 训练有算子覆盖和数值问题，用 fp32（0.5B 模型约 2 GB，放得下）
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(resolve_model_path(name))
    return tok


def load_model(name: str, device: torch.device):
    from transformers import AutoModelForCausalLM

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    path = resolve_model_path(name)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=pick_dtype(device))
    return model.to(device), path


LORA_TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


# --8<-- [start:lora]
def attach_lora(model, cfg: dict):
    """冻结全部基础权重，只在指定线性层旁边挂低秩矩阵 A[r, in]、B[out, r]。

    前向：y = W x + (alpha / r) · B A x。B 初始化为 0，所以训练开始时模型输出与基础模型完全一致。
    可训练参数 = Σ r·(in + out)，0.5B 模型 r=16 挂全部 7 类线性层约 880 万，占 1.8%。
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
                        target_modules=LORA_TARGETS[cfg["lora_targets"]], task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    if cfg.get("grad_checkpointing"):
        # 冻结的基础模型 + 检查点：输入 embedding 必须 requires_grad，否则重算段里没有梯度可传
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model
# --8<-- [end:lora]


def param_report(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    return {"trainable": trainable, "total": total, "trainable_pct": round(100 * trainable / total, 3),
            "trainable_tensors": len(names), "examples": names[:6]}


def adapter_path(run_dir: Path) -> Path:
    return run_dir / "artifacts" / "adapter"
