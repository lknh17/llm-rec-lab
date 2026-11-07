"""偏好数据：在线采样、偏好对构造、回答编码。

偏好对 = (prompt, chosen, rejected)。两种来源：
  - rule（规则构造）：chosen = 标准答案；rejected = 标准答案上做一处"典型错误"——有条件就删掉条件（漏条件），
    没条件就塞一个原文没有的卖点（无依据）。负例覆盖哪些错误，完全由规则决定。
  - onpolicy（在线采样）：SFT 模型对每个 prompt 采样 G 条；chosen = 标准答案，rejected = G 条里奖励最低、
    且不全对的一条。负例是"模型真的会犯的错"；但模型在这类 prompt 上 G 条全对时，就不产生偏好对。
"""

from __future__ import annotations

import copy
import random
import time

import torch

from adextract.data import IGNORE, prompt_text
from adextract.prompts import build_messages
from adextract.schema import norm, to_json
from adextract.synth import CATALOG

from .reward import all_ok, reward


# --8<-- [start:sample]
@torch.no_grad()
def sample(model, tok, prompts: list[str], device, n: int, temperature: float, max_new_tokens: int,
           batch_prompts: int = 8) -> tuple[list[list[dict]], float]:
    """每个 prompt 采样 n 条（num_return_sequences=n）。返回 [prompt][n] 的 {output, ids, new_tokens}，以及生成秒数。

    ids 是生成出来的 token（截到第一个停止符，含停止符）——GRPO 直接对这些 token 算对数概率，
    不重新分词，保证"被奖励的"和"被求梯度的"是同一串 token。
    """
    model.eval()
    tok.padding_side = "left"
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    out, secs = [], 0.0
    for s in range(0, len(prompts), batch_prompts):
        chunk = prompts[s: s + batch_prompts]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)   # [b, T_p]
        t0 = time.perf_counter()
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                             top_p=1.0, top_k=0, repetition_penalty=1.0,  # 覆盖 Qwen 默认的 top_k/top_p/重复惩罚：
                             num_return_sequences=n, pad_token_id=tok.pad_token_id,  # 采样分布 = 模型分布 / 温度
                             eos_token_id=stop_ids)                                                    # [b·n, T_p + T_new]
        if device.type == "cuda":
            torch.cuda.synchronize()
        secs += time.perf_counter() - t0
        new = gen[:, enc["input_ids"].shape[1]:].tolist()                                              # [b·n, T_new]
        for i in range(len(chunk)):
            group = []
            for j in range(n):
                ids = new[i * n + j]
                k = next((t + 1 for t, x in enumerate(ids) if x in stop_ids), len(ids))
                group.append({"output": tok.decode(ids[:k], skip_special_tokens=True), "ids": ids[:k], "new_tokens": k})
            out.append(group)
    tok.padding_side = "right"
    model.train()
    return out, secs
# --8<-- [end:sample]


def _ungrounded_point(text: str, rng: random.Random) -> str | None:
    src = norm(text)
    pool = [p for pts in CATALOG.values() for p in pts if norm(p) not in src]
    return rng.choice(pool) if pool else None


# --8<-- [start:rule_pairs]
def rule_rejected(row: dict, rng: random.Random) -> tuple[dict, str]:
    """在标准答案上做一处典型错误。返回 (rejected, 错误类型)。"""
    bad = copy.deepcopy(row["gold"])
    conds = [p for p in bad["promotions"] if p.get("condition")]
    if conds:
        rng.choice(conds)["condition"] = None             # 漏条件："指定款第二件半价"写成"第二件半价"
        return bad, "drop_condition"
    extra = _ungrounded_point(row["text"], rng)
    if extra:
        bad["selling_points"] = bad["selling_points"] + [extra]   # 无依据：原文没有的卖点
        return bad, "add_ungrounded"
    bad["category"] = None if bad["category"] else "跑鞋"
    return bad, "wrong_category"
# --8<-- [end:rule_pairs]


def build_pairs(rows: list[dict], cfg: dict, model=None, tok=None, device=None, log=print) -> tuple[list[dict], dict]:
    """返回偏好对列表和统计。每个偏好对：{id, text, chosen(str), rejected(str), kind}。"""
    rng = random.Random(f"pairs-{cfg['data_seed']}-{cfg['seed']}")
    pairs, stats = [], {"source": cfg["pref_pairs"], "prompts": len(rows), "kinds": {}, "gen_seconds": 0.0}
    if cfg["pref_pairs"] == "rule":
        for r in rows:
            bad, kind = rule_rejected(r, rng)
            pairs.append({"id": r["id"], "text": r["text"], "chosen": to_json(r["gold"]), "rejected": to_json(bad),
                          "kind": kind, "row_kind": r["kind"]})
    else:
        prompts = [prompt_text(tok, r["text"], cfg) for r in rows]
        groups, secs = sample(model, tok, prompts, device, cfg["pref_group"], cfg["pref_temperature"], cfg["max_new_tokens"])
        stats["gen_seconds"] = round(secs, 1)
        all_right = 0
        rewards = []
        for r, g in zip(rows, groups):
            scored = []
            for x in g:
                rw, sc = reward(r["text"], x["output"], r["gold"])
                scored.append((rw, all_ok(sc), x))
            rewards += [s for s, _, _ in scored]
            wrong = [t for t in scored if not t[1]]
            if not wrong:
                all_right += 1
                continue
            worst = min(wrong, key=lambda t: t[0])
            pairs.append({"id": r["id"], "text": r["text"], "chosen": to_json(r["gold"]), "rejected": worst[2]["output"],
                          "kind": "onpolicy", "row_kind": r["kind"], "rejected_reward": worst[0]})
        stats.update(group_all_correct=all_right, group_all_correct_frac=round(all_right / max(len(rows), 1), 4),
                     sample_reward_mean=round(sum(rewards) / max(len(rewards), 1), 4))
        log(f"在线采样 {len(rows)} 个 prompt × {cfg['pref_group']} 条，用时 {secs:.0f}s；"
            f"{all_right} 个 prompt 全对（不产生偏好对），得到 {len(pairs)} 个偏好对")
    if cfg["pref_flip"]:
        for p in pairs:
            p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
    for p in pairs:
        stats["kinds"][p["kind"]] = stats["kinds"].get(p["kind"], 0) + 1
    stats["pairs"] = len(pairs)
    stats["pair_yield"] = round(len(pairs) / max(len(rows), 1), 4)
    stats["flipped"] = bool(cfg["pref_flip"])
    return pairs, stats


# --8<-- [start:encode_response]
def encode_response(tok, text: str, response: str, cfg: dict) -> dict:
    """prompt + 一条回答 -> input_ids、labels（只监督回答，和 SFT 的 assistant-only 一致）。"""
    messages = build_messages(text, cfg["prompt_mode"])
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(messages + [{"role": "assistant", "content": response}], tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    assert ids[: len(p_ids)] == p_ids, "分词边界不一致"
    return {"input_ids": ids, "labels": [IGNORE] * len(p_ids) + ids[len(p_ids):], "n_prompt": len(p_ids)}
# --8<-- [end:encode_response]


def encode_sampled(prompt_ids: list[int], resp_ids: list[int]) -> dict:
    """GRPO：prompt token + 采样出的 token（原样，不重新分词）。"""
    return {"input_ids": prompt_ids + resp_ids, "labels": [IGNORE] * len(prompt_ids) + resp_ids,
            "n_prompt": len(prompt_ids)}
