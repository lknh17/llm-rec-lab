"""分词器：字节级 BPE（本语料上训练）/ 纯 UTF-8 字节 / Qwen2.5 现成分词器，统一成同一个小接口。

    tok.encode(text) -> list[int]        tok.vocab_size        tok.eos_id        tok.save(dir) / load(dir)

一个汉字在 UTF-8 里是 3 个字节。字节级 BPE 的初始词表就是 256 个字节，合并规则把高频字节串逐步合成更长的 token：
常用汉字很快合成 1 个 token，常用词再合成 1 个 token；低频字合不上，就"回退"成 2-3 个字节 token。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

EOS = "<|endoftext|>"
# Qwen2 的预切分正则：字母串、单个数字、标点串、空白分开切；汉字属于 \p{L}，连续汉字是一个预切分片段
QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
                r"\s+(?!\S)|\s+")


class BytesTokenizer:
    """不训练的基线：token = UTF-8 字节（0-255），256 = 文档分隔符。"""

    kind = "bytes"
    vocab_size = 257
    eos_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def piece(self, i: int) -> str:
        return f"<0x{i:02X}>" if i < 256 else EOS

    def save(self, d: Path) -> None:
        (Path(d) / "tokenizer_kind.json").write_text(json.dumps({"kind": "bytes"}))


class HFTokenizer:
    """包装 tokenizers.Tokenizer（BPE）或 transformers 的 Qwen 分词器。"""

    def __init__(self, tk, kind: str):
        self.tk, self.kind = tk, kind
        if kind == "qwen":
            self.vocab_size = len(tk)
            self.eos_id = tk.convert_tokens_to_ids(EOS)
        else:
            self.vocab_size = tk.get_vocab_size()
            self.eos_id = tk.token_to_id(EOS)

    def encode(self, text: str) -> list[int]:
        if self.kind == "qwen":
            return self.tk.encode(text, add_special_tokens=False)
        return self.tk.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if not texts:  # transformers 的分词器不接受空列表
            return []
        if self.kind == "qwen":
            return self.tk(texts, add_special_tokens=False)["input_ids"]
        return [e.ids for e in self.tk.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids)

    def piece(self, i: int) -> str:
        """token 对应的原文片段（字节级 token 解码不出完整字时显示字节）。"""
        s = self.decode([i])
        return s if "\ufffd" not in s else (self.tk.convert_ids_to_tokens(i) if self.kind == "qwen" else self.tk.id_to_token(i))

    def save(self, d: Path) -> None:
        d = Path(d)
        if self.kind == "qwen":
            (d / "tokenizer_kind.json").write_text(json.dumps({"kind": "qwen"}))
        else:
            self.tk.save(str(d / "tokenizer.json"))
            (d / "tokenizer_kind.json").write_text(json.dumps({"kind": "bpe"}))


# --8<-- [start:train_bpe]
def train_bpe(texts: list[str], vocab_size: int) -> HFTokenizer:
    """字节级 BPE：先按 Qwen2 的正则预切分，再把每个片段转成字节，从 256 个字节出发反复合并最高频的相邻对，
    直到词表达到 vocab_size。"""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOS], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(texts, trainer=trainer)
    return HFTokenizer(tk, "bpe")
# --8<-- [end:train_bpe]


def load_qwen() -> HFTokenizer:
    from transformers import AutoTokenizer

    from labkit.assets import resolve_model_path

    return HFTokenizer(AutoTokenizer.from_pretrained(resolve_model_path("Qwen/Qwen2.5-0.5B-Instruct")), "qwen")


def load(d: Path):
    kind = json.loads((Path(d) / "tokenizer_kind.json").read_text())["kind"]
    if kind == "bytes":
        return BytesTokenizer()
    if kind == "qwen":
        return load_qwen()
    from tokenizers import Tokenizer

    return HFTokenizer(Tokenizer.from_file(str(Path(d) / "tokenizer.json")), "bpe")


def build(cfg: dict, tok_texts: list[str], log=print) -> tuple[object, dict]:
    t0 = time.time()
    if cfg["tokenizer"] == "bytes":
        tok = BytesTokenizer()
    elif cfg["tokenizer"] == "qwen":
        tok = load_qwen()
    else:
        tok = train_bpe(tok_texts, cfg["vocab_size"])
    info = {"kind": cfg["tokenizer"], "vocab_size": tok.vocab_size, "train_seconds": round(time.time() - t0, 1),
            "train_bytes": sum(len(t.encode()) for t in tok_texts) if cfg["tokenizer"] == "bpe" else 0}
    log(f"分词器 {cfg['tokenizer']}：词表 {tok.vocab_size}，用时 {info['train_seconds']} 秒")
    return tok, info


# --8<-- [start:compression]
def compression(tok, texts: list[str]) -> dict:
    """压缩率：同一批文本切成多少 token。bytes_per_token 越大，序列越短。"""
    ids = tok.encode_batch(texts)
    n_tok = sum(len(x) for x in ids)
    n_bytes = sum(len(t.encode()) for t in texts)
    n_chars = sum(len(t) for t in texts)
    return {"docs": len(texts), "tokens": n_tok, "bytes": n_bytes, "chars": n_chars,
            "bytes_per_token": round(n_bytes / max(n_tok, 1), 4), "chars_per_token": round(n_chars / max(n_tok, 1), 4),
            "tokens_per_doc": round(n_tok / max(len(texts), 1), 2)}


def char_split(tok, chars: list[tuple[str, int]]) -> dict:
    """常用汉字（按出现次数加权）里，有多少被拆成 2 个以上 token（BPE 没合上，回退到字节）。"""
    total = sum(c for _, c in chars)
    multi = [(ch, c) for ch, c in chars if len(tok.encode(ch)) > 1]
    return {"chars_checked": len(chars), "multi_token_chars": len(multi),
            "multi_token_weighted": round(sum(c for _, c in multi) / max(total, 1), 4),
            "examples": [ch for ch, _ in multi[:12]]}
# --8<-- [end:compression]


def show(tok, text: str, limit: int = 40) -> list[str]:
    """一段文本切成的 token（原文片段），给追踪页看。"""
    return [tok.piece(i) for i in tok.encode(text)[:limit]]
