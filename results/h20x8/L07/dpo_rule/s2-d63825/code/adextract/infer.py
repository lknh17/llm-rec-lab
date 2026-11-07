"""评估与使用：批量贪心生成 -> 解析 -> 打分 -> 错例。"""

from __future__ import annotations

import time

import torch

from .data import prompt_text
from .schema import aggregate, parse_output, score_one


# --8<-- [start:generate]
@torch.no_grad()
def generate(model, tok, prompts: list[str], device, max_new_tokens: int, batch_size: int = 16) -> list[dict]:
    """左侧补齐后批量生成。生成时新 token 接在序列末尾，所以 pad 必须在左边，不能像训练那样补在右边。"""
    model.eval()
    tok.padding_side = "left"
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    out = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s: s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)  # [b, T_prompt]
        t0 = time.perf_counter()
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)  # [b, T_prompt + T_new]
        ms = (time.perf_counter() - t0) * 1e3
        new = gen[:, enc["input_ids"].shape[1]:]                                                  # [b, T_new]
        for i in range(len(chunk)):
            ids = new[i].tolist()
            n_new = next((k + 1 for k, t in enumerate(ids) if t in stop_ids), len(ids))
            out.append({"output": tok.decode(ids[:n_new], skip_special_tokens=True), "new_tokens": n_new,
                        "hit_max_new_tokens": n_new >= max_new_tokens, "batch_ms": round(ms, 1)})
    tok.padding_side = "right"
    model.train()
    return out
# --8<-- [end:generate]


def evaluate_set(model, tok, rows: list[dict], cfg: dict, device, set_name: str) -> tuple[dict, list[dict]]:
    prompts = [prompt_text(tok, r["text"], cfg, cfg["infer_template"]) for r in rows]
    t0 = time.perf_counter()
    gens = generate(model, tok, prompts, device, cfg["max_new_tokens"])
    secs = time.perf_counter() - t0
    preds, scores = [], []
    for r, g in zip(rows, gens):
        sc = score_one(r["text"], g["output"], r["gold"])
        scores.append(sc)
        parsed, _ = parse_output(g["output"])
        preds.append({"set": set_name, "id": r["id"], "text": r["text"], "gold": r["gold"], "output": g["output"],
                      "parsed": parsed, "errors": sc["errors"], "format_error": sc["format_error"],
                      "ungrounded_values": sc["ungrounded_values"], "tags": r.get("tags") or [r.get("kind")],
                      "new_tokens": g["new_tokens"], "hit_max_new_tokens": g["hit_max_new_tokens"]})
    agg = aggregate(scores)
    agg["gen_seconds"] = round(secs, 1)
    agg["gen_tokens"] = sum(g["new_tokens"] for g in gens)
    # 按标签分组：整体变好可能只来自某一类样本
    by_tag: dict[str, list] = {}
    for p, sc in zip(preds, scores):
        for t in p["tags"]:
            by_tag.setdefault(t, []).append(sc)
    agg["by_tag"] = {t: {"n": len(v), "all_ok": round(sum(1 for s in v if s["format_ok"] and not s["errors"]) / len(v), 3),
                         "condition_keep": aggregate(v)["condition_keep"]} for t, v in sorted(by_tag.items())}
    return agg, preds
