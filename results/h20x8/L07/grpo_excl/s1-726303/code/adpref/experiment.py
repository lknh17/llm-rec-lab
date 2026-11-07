"""偏好优化的执行入口：SFT（与 L05 基线同配置）→ 合并 SFT 适配器 → 偏好阶段（DPO / GRPO，新挂一个 LoRA）
→ 评估（dev、合成验证、排除型条件合成集）→ 保存两个适配器 → 重新加载核对。"""

from __future__ import annotations

import gc
import random
from collections import Counter

import torch

from adextract import data as D
from adextract import synth
from adextract.infer import evaluate_set, generate
from adextract.modeling import attach_lora, load_model, load_tokenizer, param_report, pick_device
from adextract.prompts import PROMPT_VERSION
from adextract.train import train

from .optim import train_dpo, train_grpo
from .pairs import build_pairs

EXCL_EVAL_SEED = 20_000
PREF_SEED_OFFSET = 7_000


def synth_exclusion(n: int = 60) -> list[dict]:
    """排除型条件合成评估集：SFT 训练集里没有这种句式（L06 的错例类别），用来看偏好阶段能不能补上。"""
    rng = random.Random(EXCL_EVAL_SEED)
    rows = []
    for i in range(n):
        ad = synth.make_ad(rng, "exclusion")
        ad["id"], ad["tags"] = f"excl-{i}", ["排除型条件"]
        rows.append(ad)
    return rows


def pref_rows(cfg: dict, exclude: set[str]) -> list[dict]:
    """偏好阶段的 prompt：同一个生成器、另一个种子，和 SFT 训练集、评估集都不重叠。"""
    return synth.generate(cfg["pref_size"], cfg["hard_ratio"], PREF_SEED_OFFSET + cfg["data_seed"], exclude=exclude,
                          exclusion_ratio=cfg["pref_exclusion_ratio"])


def _free(device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def run(ctx, cfg: dict) -> None:
    # ---------------- 1. 数据 ----------------
    ctx.stage("prepare")
    torch.manual_seed(cfg["seed"])
    dev, synval, excl = D.load_evalset("dev"), synth.synth_valid(), synth_exclusion()
    for r in synval:
        r["tags"] = [r["kind"]]
    eval_texts = {r["text"] for r in dev + D.load_evalset("test") + synval + excl}
    train_rows = synth.generate(cfg["train_size"], cfg["hard_ratio"], cfg["data_seed"], exclude=eval_texts,
                                exclusion_ratio=cfg["exclusion_ratio"])
    prefs = pref_rows(cfg, eval_texts | {r["text"] for r in train_rows}) if cfg["pref_method"] != "none" else []
    ctx.info("data", {
        "kind": "教学合成训练数据 + 人工标注评估集",
        "train": {"source": f"模板生成器 v{synth.GENERATOR_VERSION}", "size": len(train_rows), "seed": cfg["data_seed"],
                  "kinds": dict(Counter(r["kind"] for r in train_rows)), "hash": D.rows_hash(train_rows)},
        "pref": {"size": len(prefs), "seed": PREF_SEED_OFFSET + cfg["data_seed"],
                 "exclusion_ratio": cfg["pref_exclusion_ratio"], "kinds": dict(Counter(r["kind"] for r in prefs))},
        "eval": {"dev": {"n": len(dev), "hash": D.evalset_hash("dev")}, "synth_valid": {"n": len(synval)},
                 "synth_excl": {"n": len(excl), "seed": EXCL_EVAL_SEED, "source": "排除型条件合成样本（SFT 训练集里没有）"}},
        "prompt_version": PROMPT_VERSION,
    })
    device = pick_device(cfg["device"])
    tok = load_tokenizer(cfg["model"])
    tok.padding_side = "right"
    encoded = []
    for r in train_rows:
        e = D.encode_example(tok, r["text"], r["gold"], cfg)
        if e is not None:
            e["id"], e["kind"], e["raw"] = r["id"], r["kind"], r
            encoded.append(e)
    dev_encoded = [D.encode_example(tok, r["text"], r["gold"], dict(cfg, max_len=4096)) for r in dev]
    ctx.info("data", {"tokenized": {"examples": len(encoded), "tokens_total": sum(len(e["input_ids"]) for e in encoded),
                                    "supervised_tokens_total": sum(e["n_supervised"] for e in encoded),
                                    "zero_supervision_examples": sum(1 for e in encoded if e["n_supervised"] == 0)}})

    model, model_path = load_model(cfg["model"], device)
    ctx.info("model", {"base": cfg["model"], "path": model_path, "dtype": str(next(model.parameters()).dtype),
                       "device": str(device), "train_mode": "lora"})

    # ---------------- 2. SFT（与 L05 基线同一套代码和参数） ----------------
    ctx.stage("train")
    model = attach_lora(model, cfg)
    ctx.info("model", {"lora": {"r": cfg["lora_r"], "alpha": cfg["lora_alpha"], "targets": cfg["lora_targets"]}})
    sft = train(ctx, cfg, model, tok, encoded, dev_encoded, device)
    ctx.info("train", {"sft": sft})
    model.save_pretrained(ctx.artifact("sft_adapter"))
    ctx.register_artifact("sft_adapter", "lora_adapter", f"SFT 的 LoRA 适配器，基础模型 {cfg['model']}")

    # ---------------- 3. 偏好阶段 ----------------
    pref_result, pair_stats = None, None
    if cfg["pref_method"] != "none":
        # 合并 SFT 适配器，再挂一个新 LoRA：关掉新 LoRA 就是参考模型 π_ref（= SFT 模型）
        model = model.merge_and_unload()
        _free(device)
        model = attach_lora(model, cfg)
        rep = param_report(model)
        ctx.info("model", {"params": rep, "pref_lora": {"r": cfg["lora_r"], "targets": cfg["lora_targets"]}})
        if cfg["pref_method"] == "dpo":
            pairs, pair_stats = build_pairs(prefs, cfg, model, tok, device, ctx.log)
            ctx.info("pref", {"pairs": pair_stats})
            ctx.trace("pairs", {"note": "前 6 个偏好对（chosen 在上，rejected 在下）", "pairs": pairs[:6]})
            pref_result = train_dpo(ctx, cfg, model, tok, pairs, device)
        else:
            pref_result = train_grpo(ctx, cfg, model, tok, prefs, device)
        ctx.info("train", {"pref": pref_result})
    else:
        ctx.info("model", {"params": param_report(model)})

    # ---------------- 4. 评估 ----------------
    ctx.stage("evaluate")
    summary, all_preds = {}, []
    for name, rows in (("dev", dev), ("synth_valid", synval), ("synth_excl", excl)):
        agg, preds = evaluate_set(model, tok, rows, cfg, device, name)
        summary[name] = agg
        all_preds += preds
        ctx.log(f"[{name}] 格式 {agg['format_ok']} 卖点F1 {agg['point_f1']} 条件保留 {agg['condition_keep']} "
                f"无依据 {agg['ungrounded_rate']} 全对 {agg['exact_all_ok']}")
        ctx.check()
    ctx.predictions(all_preds)
    ctx.summary(summary)
    ctx.summary({"train": sft, "pref": {"result": pref_result, "pairs": pair_stats}, "headline": {
        "dev_exact_all_ok": summary["dev"]["exact_all_ok"], "dev_condition_keep": summary["dev"]["condition_keep"],
        "dev_ungrounded_rate": summary["dev"]["ungrounded_rate"],
        "excl_exact_all_ok": summary["synth_excl"]["exact_all_ok"],
        "excl_condition_keep": summary["synth_excl"]["condition_keep"],
        "synth_exact_all_ok": summary["synth_valid"]["exact_all_ok"]}})

    # ---------------- 5. 保存、重新加载、推理 ----------------
    probe = dev[:4]
    prompts = [D.prompt_text(tok, r["text"], cfg) for r in probe]
    before = [g["output"] for g in generate(model, tok, prompts, device, cfg["max_new_tokens"])]
    if cfg["pref_method"] != "none":
        model.save_pretrained(ctx.artifact("adapter"))
        ctx.register_artifact("adapter", "lora_adapter", "偏好阶段的 LoRA，加载前要先合并 sft_adapter")
    del model
    _free(device)
    from peft import PeftModel

    base, _ = load_model(cfg["model"], device)
    reloaded = PeftModel.from_pretrained(base, ctx.artifact("sft_adapter"))
    if cfg["pref_method"] != "none":
        reloaded = PeftModel.from_pretrained(reloaded.merge_and_unload(), ctx.artifact("adapter"))
    after = [g["output"] for g in generate(reloaded, tok, prompts, device, cfg["max_new_tokens"])]
    same = [a == b for a, b in zip(before, after)]
    ctx.trace("reload", {"checked": len(same), "identical": sum(same), "ok": all(same),
                         "chain": "基础模型 + sft_adapter（合并）" + (" + adapter" if cfg["pref_method"] != "none" else "")})
    ctx.info("checks", {"reload_identical": f"{sum(same)}/{len(same)}"})
    ctx.log(f"重新加载核对：{sum(same)}/{len(same)} 条输出一致")
