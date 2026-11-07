"""广告元素抽取的执行入口：数据 -> （LoRA 微调）-> 固定评估集评估 -> 保存 -> 重新加载并推理。"""

from __future__ import annotations

import gc
from collections import Counter

import torch

from . import data as D
from . import synth
from .infer import evaluate_set, generate
from .modeling import adapter_path, attach_lora, load_model, load_tokenizer, param_report, pick_device
from .prompts import PROMPT_VERSION
from .train import train


def run(ctx, cfg: dict) -> None:
    # ---------------- 1. 数据 ----------------
    ctx.stage("prepare")
    torch.manual_seed(cfg["seed"])
    dev, test, synval = D.load_evalset("dev"), D.load_evalset("test"), synth.synth_valid()
    for r in synval:
        r["tags"] = [r["kind"]]
    eval_texts = {r["text"] for r in dev + test + synval}
    train_rows = synth.generate(cfg["train_size"], cfg["hard_ratio"], cfg["data_seed"], exclude=eval_texts,
                                exclusion_ratio=cfg["exclusion_ratio"])
    ctx.info("data", {
        "kind": "教学合成训练数据 + 人工标注评估集",
        "train": {"source": f"模板生成器 v{synth.GENERATOR_VERSION}", "size": len(train_rows), "seed": cfg["data_seed"],
                  "hard_ratio": cfg["hard_ratio"], "exclusion_ratio": cfg["exclusion_ratio"],
                  "kinds": dict(Counter(r["kind"] for r in train_rows)),
                  "hash": D.rows_hash(train_rows)},
        "eval": {"dev": {"n": len(dev), "hash": D.evalset_hash("dev"), "source": "人工撰写并核对，迭代用"},
                 "synth_valid": {"n": len(synval), "source": "同一生成器、不同种子，分布内参照"},
                 "test": {"n": len(test), "hash": D.evalset_hash("test"), "source": "人工撰写并核对，保留测试",
                          "used": cfg["final_test"]}},
        "prompt_version": PROMPT_VERSION, "prompt_mode": cfg["prompt_mode"], "infer_template": cfg["infer_template"],
        "leak_check": {"train_texts_in_eval": sum(1 for r in train_rows if r["text"] in eval_texts)},
    })
    device = pick_device(cfg["device"])
    tok = load_tokenizer(cfg["model"])
    tok.padding_side = "right"

    encoded, dropped, truncated = [], 0, 0
    for r in train_rows:
        e = D.encode_example(tok, r["text"], r["gold"], cfg)
        if e is None:
            dropped += 1
            continue
        truncated += e["truncated"]
        e["id"], e["kind"], e["raw"] = r["id"], r["kind"], r
        encoded.append(e)
    dev_eval_cfg = dict(cfg, supervision="assistant_only", truncation="drop_long", max_len=4096)
    dev_encoded = [D.encode_example(tok, r["text"], r["gold"], dev_eval_cfg) for r in dev]
    lens = [len(e["input_ids"]) for e in encoded]
    sup = [e["n_supervised"] for e in encoded]
    stats = {"examples": len(encoded), "dropped_too_long": dropped, "truncated": truncated,
             "zero_supervision_examples": sum(1 for s in sup if s == 0),
             "tokens_total": sum(lens), "supervised_tokens_total": sum(sup),
             "len_min_mean_max": [min(lens, default=0), round(sum(lens) / max(len(lens), 1), 1), max(lens, default=0)],
             "prompt_tokens": encoded[0]["n_prompt"] if encoded else None}
    ctx.info("data", {"tokenized": stats})
    ctx.log(f"训练样本 {len(encoded)} 条（丢弃 {dropped}，截断 {truncated}），总 token {stats['tokens_total']}，"
            f"监督 token {stats['supervised_tokens_total']}，prompt 长度 {stats['prompt_tokens']}")
    if encoded:
        # 训练开始后会被"第 1 个 batch 的第 1 条"覆盖，保证 ①-⑤ 追踪的是同一条样本；不训练时保留这一条
        ctx.trace("sample", D.sample_trace_payload(encoded[0], cfg["prompt_mode"],
                                                   "训练集第 1 条样本（教学合成；本次未训练，只展示模板与监督位置）"))

    model, model_path = load_model(cfg["model"], device)
    ctx.info("model", {"base": cfg["model"], "path": model_path, "dtype": str(next(model.parameters()).dtype),
                       "device": str(device), "train_mode": cfg["train_mode"]})

    # ---------------- 2. 训练 ----------------
    train_result = None
    if cfg["train_mode"] == "lora":
        ctx.stage("train")
        model = attach_lora(model, cfg)
        rep = param_report(model)
        ctx.info("model", {"lora": {"r": cfg["lora_r"], "alpha": cfg["lora_alpha"], "targets": cfg["lora_targets"]},
                           "params": rep})
        ctx.log(f"可训练参数 {rep['trainable']:,} / {rep['total']:,}（{rep['trainable_pct']}%）")

        def on_epoch_end(epoch: int, step: int) -> None:
            if cfg["eval_each_epoch"]:
                agg, _ = evaluate_set(model, tok, dev, cfg, device, "dev")
                for k in ("format_ok", "category_acc", "point_f1", "condition_keep", "ungrounded_rate", "exact_all_ok"):
                    if agg.get(k) is not None:
                        ctx.metric(f"dev_{k}", agg[k], step=step, split="dev", epoch=epoch)
                ctx.log(f"epoch {epoch} dev：全对率 {agg['exact_all_ok']} 条件保留 {agg['condition_keep']}")

        train_result = train(ctx, cfg, model, tok, encoded, dev_encoded, device, on_epoch_end)
        ctx.info("train", train_result)
        ctx.trace("update_check", {
            "lora_B_norm_before": train_result["lora_B_norm_before"], "lora_B_norm_after": train_result["lora_B_norm_after"],
            "base_weight_probe_before": train_result["base_weight_probe_before"],
            "base_weight_probe_after": train_result["base_weight_probe_after"],
            "note": "LoRA 的 B 初始化为 0：训练后范数 > 0 说明参数确实被更新；基础权重指纹必须不变",
        })
        if stats["supervised_tokens_total"] == 0:
            ctx.log("注意：本次训练没有任何被监督的 token")

    # ---------------- 3. 评估 ----------------
    ctx.stage("evaluate")
    sets = [("dev", dev), ("synth_valid", synval)] + ([("test", test)] if cfg["final_test"] else [])
    summary, all_preds = {}, []
    for name, rows in sets:
        agg, preds = evaluate_set(model, tok, rows, cfg, device, name)
        summary[name] = agg
        all_preds += preds
        ctx.log(f"[{name}] 格式合法 {agg['format_ok']} 品类 {agg['category_acc']} 卖点F1 {agg['point_f1']} "
                f"条件保留 {agg['condition_keep']} 无依据 {agg['ungrounded_rate']} 全对 {agg['exact_all_ok']} "
                f"（{agg['gen_seconds']}s）")
        ctx.check()
    ctx.predictions(all_preds)
    ctx.summary(summary)
    ctx.summary({"train": train_result, "headline": {
        "dev_exact_all_ok": summary["dev"]["exact_all_ok"], "dev_format_ok": summary["dev"]["format_ok"],
        "dev_condition_keep": summary["dev"]["condition_keep"], "dev_ungrounded_rate": summary["dev"]["ungrounded_rate"],
        "synth_exact_all_ok": summary["synth_valid"]["exact_all_ok"]}})

    # ---------------- 4. 保存、重新加载、推理 ----------------
    if cfg["train_mode"] == "lora":
        out = adapter_path(ctx.run_dir)
        model.save_pretrained(out)
        ctx.register_artifact("adapter", "lora_adapter", f"LoRA 适配器，基础模型 {cfg['model']}")
        probe_rows = dev[:4]
        from .data import prompt_text

        prompts = [prompt_text(tok, r["text"], cfg, cfg["infer_template"]) for r in probe_rows]
        before = [g["output"] for g in generate(model, tok, prompts, device, cfg["max_new_tokens"])]
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        from peft import PeftModel

        base, _ = load_model(cfg["model"], device)
        reloaded = PeftModel.from_pretrained(base, out)
        after = [g["output"] for g in generate(reloaded, tok, prompts, device, cfg["max_new_tokens"])]
        same = [a == b for a, b in zip(before, after)]
        ctx.trace("reload", {"artifact": "artifacts/adapter", "checked": len(same), "identical": sum(same),
                             "ok": all(same), "samples": [{"id": r["id"], "in_memory": a, "reloaded": b}
                                                          for r, a, b in zip(probe_rows, before, after)]})
        ctx.info("checks", {"reload_identical": f"{sum(same)}/{len(same)}"})
        ctx.log(f"重新加载核对：{sum(same)}/{len(same)} 条输出与内存中的模型完全一致")
