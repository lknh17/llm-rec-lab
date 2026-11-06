"""模型输入：messages -> 聊天模板文本 -> token -> labels（监督位置）-> batch 张量。

最关键的一行是 labels 的构造：值为 -100 的位置不参与 loss。
assistant_only 时，系统提示、广告原文、"<|im_start|>assistant\\n" 这段模板头都是 -100，
只有回答 JSON 和结尾的 "<|im_end|>\\n" 有真实标签——模型只学"怎么回答"，不学"怎么复述题目"。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .prompts import build_messages, plain_prompt
from .schema import to_json

IGNORE = -100
EVALSET_DIR = Path(__file__).parent / "evalsets"


def load_evalset(name: str) -> list[dict]:
    rows = [json.loads(line) for line in (EVALSET_DIR / f"{name}.jsonl").read_text().splitlines() if line.strip()]
    return rows


def evalset_hash(name: str) -> str:
    return hashlib.sha256((EVALSET_DIR / f"{name}.jsonl").read_bytes()).hexdigest()[:12]


def sample_trace_payload(ex: dict, prompt_mode: str, note: str) -> dict:
    """样本追踪 ①②：原始字段、messages、聊天模板文本。ex 需带 raw（encode 后由 experiment 附上）。"""
    src = ex["raw"]
    msgs = build_messages(src["text"], prompt_mode) + [{"role": "assistant", "content": to_json(src["gold"])}]
    return {"note": note, "raw": src, "messages": msgs, "template_text": ex["full_text"],
            "prompt_text_len_chars": len(ex["prompt_text"]), "n_prompt": ex["n_prompt"],
            "n_total": len(ex["input_ids"]), "n_supervised": ex["n_supervised"], "truncated": ex["truncated"]}


def rows_hash(rows: list[dict]) -> str:
    """训练集指纹（文案 + 标注）：两次运行的训练数据是否逐条相同，看这个值。"""
    h = hashlib.sha256()
    for r in rows:
        h.update(r["text"].encode())
        h.update(to_json(r["gold"]).encode())
    return h.hexdigest()[:12]


def prompt_text(tok, text: str, cfg: dict, template: str = "chat") -> str:
    if template == "plain":
        return plain_prompt(text)
    return tok.apply_chat_template(build_messages(text, cfg["prompt_mode"]), add_generation_prompt=True,
                                   tokenize=False)


# --8<-- [start:encode]
def encode_example(tok, text: str, target: dict, cfg: dict) -> dict | None:
    """一条训练样本 -> input_ids 和 labels。超长且 truncation=drop_long 时返回 None（整条丢弃）。"""
    messages = build_messages(text, cfg["prompt_mode"])
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(messages + [{"role": "assistant", "content": to_json(target)}], tokenize=False)
    # 模板一致性：完整对话必须以 prompt 开头，否则"前 n_prompt 个 token 是题目"这个假设就不成立
    assert full.startswith(prompt), "聊天模板的 prompt 不是完整对话的前缀，监督位置会错位"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert input_ids[: len(prompt_ids)] == prompt_ids, "分词边界不一致"
    n_prompt = len(prompt_ids)

    if cfg["supervision"] == "assistant_only":
        labels = [IGNORE] * n_prompt + input_ids[n_prompt:]   # 只监督回答部分
    else:
        labels = list(input_ids)                              # 全部 token 都监督

    truncated = False
    if len(input_ids) > cfg["max_len"]:
        if cfg["truncation"] == "drop_long":
            return None
        # right：直接砍掉尾部。回答在尾部，最先被砍掉的就是监督目标
        input_ids, labels = input_ids[: cfg["max_len"]], labels[: cfg["max_len"]]
        truncated = True
    n_sup = sum(1 for x in labels[1:] if x != IGNORE)  # 第 0 位没有"前一个位置"去预测它，不计入
    return {"input_ids": input_ids, "labels": labels, "n_prompt": n_prompt, "n_supervised": n_sup,
            "truncated": truncated, "prompt_text": prompt, "full_text": full}
# --8<-- [end:encode]


# --8<-- [start:collate]
def collate(examples: list[dict], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    """右侧补齐成矩形 batch。pad 位置 attention_mask=0、label=-100：既不被看到，也不计 loss。"""
    T = max(len(e["input_ids"]) for e in examples)
    B = len(examples)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)        # [B, T]
    labels = torch.full((B, T), IGNORE, dtype=torch.long)           # [B, T]
    attention_mask = torch.zeros((B, T), dtype=torch.long)          # [B, T]
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids.to(device), "labels": labels.to(device), "attention_mask": attention_mask.to(device)}
# --8<-- [end:collate]


def token_table(tok, example: dict, per_token_loss: list[float] | None = None, limit: int = 400) -> list[dict]:
    """把一条样本逐 token 展开：文本、id、是否被监督。网页用它给监督位置上色。"""
    rows = []
    for i, (tid, lab) in enumerate(zip(example["input_ids"], example["labels"])):
        if i >= limit:
            break
        row = {"i": i, "id": tid, "text": tok.decode([tid]), "supervised": lab != IGNORE and i > 0,
               "region": "prompt" if i < example["n_prompt"] else "answer"}
        if per_token_loss is not None and i > 0 and i - 1 < len(per_token_loss) and row["supervised"]:
            row["loss"] = round(per_token_loss[i - 1], 3)
        rows.append(row)
    return rows
