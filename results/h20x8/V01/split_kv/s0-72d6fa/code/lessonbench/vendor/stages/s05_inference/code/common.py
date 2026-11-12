"""S5 共用：加载 S1 训练好的字节级代码模型、构造真实的代码 prompt。"""

import random
import sysconfig
from pathlib import Path

import torch

from stages.s01_transformer.code.model import GPT
from tools.device import get_device

ROOT = Path(__file__).resolve().parents[3]
CKPT = ROOT / "data" / "checkpoints" / "code_gpt.pt"


def load_code_gpt(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> GPT:
    if not CKPT.exists():
        raise FileNotFoundError(f"{CKPT} 不存在，先运行：uv run python -m stages.s01_transformer.code.train")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = GPT(ckpt["cfg"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device or get_device(), dtype).eval()


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


def code_snippets(n: int, min_len: int, max_len: int, seed: int = 0) -> list[list[int]]:
    """从标准库源码里随机截取 n 段（字节 token），长度在 [min_len, max_len]。"""
    rng = random.Random(seed)
    files = sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py"))
    out = []
    while len(out) < n:
        data = files[rng.randrange(len(files))].read_bytes()
        length = rng.randint(min_len, max_len)
        if len(data) <= length:
            continue
        start = rng.randrange(len(data) - length)
        out.append(list(data[start:start + length]))
    return out
