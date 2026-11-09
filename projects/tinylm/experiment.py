"""tinylm 执行入口：语料（清洗 / 注入副本 / 去重 / 切分）→ 分词器 → token 流 → 小 GPT 从零训练一遍 → 评估。

三个准备阶段都有缓存（labkit.cache）：同一份语料和分词器只建一次，3 个训练种子、换模型参数的场景都复用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from labkit.cache import stage_cache

from . import corpus as Cp
from . import tokenizer as Tk
from .model import GPT, flops_per_token, kv_cache_stats, param_counts
from .params import PARAMS
from .train import evaluate, evaluate_docs, to_stream, train

TOK_VERSION = 1
STREAM_VERSION = 2
CORPUS_KEYS = ("corpus", "min_chars", "source_dedup", "dup_ratio", "dedup", "train_mb", "valid_mb", "test_mb", "data_seed")
SPLITS = ("train", "valid", "test", "valid_leaked", "valid_leak_few", "valid_leak_many", "valid_dup", "valid_uniq", "ads")
EVAL_SPLITS = ("valid", "test", "ads", "valid_leaked", "valid_leak_few", "valid_leak_many", "valid_dup", "valid_uniq")
LEAK_FEW, LEAK_MANY = 2, 8
TOK_KEYS = ("tokenizer", "vocab_size", "tok_train_mb", "tok_ads_ratio")
MODEL_KEYS = tuple(p.name for p in PARAMS.params if p.affects == "model")
# bf16 稠密矩阵乘的标称峰值（TFLOPS），用来算 MFU；查不到时 MFU 记 None
PEAK_TFLOPS = {"H20": 148.0, "H100": 989.0, "H800": 989.0, "A100": 312.0, "A800": 312.0, "L40S": 362.0, "4090": 165.0}
VALID_AT = {"f0125": 0.125, "f025": 0.25, "f05": 0.5, "f1": 1.0}


def peak_tflops(device) -> float | None:
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    return next((v for k, v in PEAK_TFLOPS.items() if k in name), None)


# --8<-- [start:flops_budget]
def budget_steps(cfg: dict, vocab_size: int, n_tokens: int, fpt: float) -> int | None:
    """等算力对照：预算 = flops_budget_frac × "默认结构（d=384、6 层）把整个训练流过一遍"的训练 FLOPs；
    步数 = 预算 ÷ (本模型每 token FLOPs × 每步 token)。模型越小，同样算力能看越多 token。"""
    if not cfg.get("flops_budget_frac"):
        return None
    ref_cfg = {**cfg, **{k: PARAMS.by_name[k].default for k in MODEL_KEYS}}
    with torch.device("meta"):
        ref = GPT(vocab_size, ref_cfg)
    budget = cfg["flops_budget_frac"] * flops_per_token(ref, ref_cfg)["train_flops_per_token"] * n_tokens
    return max(1, round(budget / (fpt * cfg["batch_size"] * cfg["seq_len"])))
# --8<-- [end:flops_budget]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def corpus_key(cfg: dict) -> dict:
    return {"v": Cp.CORPUS_VERSION, **{k: cfg[k] for k in CORPUS_KEYS}}


def tok_key(cfg: dict) -> dict:
    k = {"v": TOK_VERSION, "corpus": corpus_key(cfg), "tokenizer": cfg["tokenizer"]}
    if cfg["tokenizer"] == "bpe":  # 字节级和 Qwen 不训练，词表参数不影响结果
        k.update({x: cfg[x] for x in TOK_KEYS[1:]})
    return k


def _write_lines(path: Path, docs: list[str]) -> None:
    path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs))


def _read_lines(path: Path) -> list[str]:
    # 不能用 splitlines()：它还会在 \u2028、\x85 等字符处断行，而 ensure_ascii=False 的 JSON 里这些字符是原样保留的
    return [json.loads(x) for x in path.read_text().split("\n") if x]


def tokenizer_texts(cfg: dict, train_docs: list[str]) -> tuple[list[str], dict]:
    """BPE 训练文本：训练集开头 tok_train_mb；tok_ads_ratio > 0 时其中这部分字节换成模板广告文案。"""
    wiki = Cp.take_bytes(train_docs, cfg["tok_train_mb"] * (1 - cfg["tok_ads_ratio"]))
    ads: list[str] = []
    if cfg["tok_ads_ratio"] > 0:
        want = cfg["tok_train_mb"] * cfg["tok_ads_ratio"] * 1e6
        n = max(1, int(want / 85))  # 模板广告平均约 85 字节
        ads = Cp.ad_texts_synth(n)
    return wiki + ads, {"wiki_bytes": Cp.nbytes(wiki), "ads_docs": len(ads), "ads_bytes": Cp.nbytes(ads)}


def run(ctx, cfg: dict) -> None:
    # ---------------- 1. 语料 ----------------
    ctx.stage("prepare")
    torch.manual_seed(cfg["seed"])

    def build_corpus(d: Path) -> None:
        r = Cp.build(cfg, ctx.log)
        for name in ("train", "valid", "test"):
            _write_lines(d / f"{name}.jsonl", r[name])
        (d / "stats.json").write_text(json.dumps(r["stats"], ensure_ascii=False))
        (d / "valid_flags.json").write_text(json.dumps(r["valid_flags"]))

    cdir = stage_cache(ctx, "tinylm_corpus", corpus_key(cfg), build_corpus)
    docs = {n: _read_lines(cdir / f"{n}.jsonl") for n in ("train", "valid", "test")}
    cstats = json.loads((cdir / "stats.json").read_text())
    flags = json.loads((cdir / "valid_flags.json").read_text())
    # 验证集拆组：
    #   leaked = 同源文档进了训练集（这个场景下"背过"的验证文档）；
    #   dup  = 来源被转载过的**原文**；uniq = 来源从没被转载过的原文。这两组在各场景里是同一批文档
    #   （去重时原文优先保留、切分按来源哈希），所以 bpb(dup) − bpb(uniq) 可以跨场景配对比较：
    #   去重干净时它只反映两组文档本身的难易；去重没做干净时，dup 组被背过，差值会往负方向走。
    #   leak_few / leak_many = 训练集里有 1-2 篇 / 8 篇以上同源文档：记忆随重复次数增长
    fl = [dict(zip(flags, v)) for v in zip(*flags.values())]
    docs["valid_leaked"] = [d for d, f in zip(docs["valid"], fl) if f["leaked"]]
    docs["valid_leak_few"] = [d for d, f in zip(docs["valid"], fl) if 1 <= f["train_n"] <= LEAK_FEW]
    docs["valid_leak_many"] = [d for d, f in zip(docs["valid"], fl) if f["train_n"] >= LEAK_MANY]
    docs["valid_dup"] = [d for d, f in zip(docs["valid"], fl) if f["orig"] and f["dup_src"]]
    docs["valid_uniq"] = [d for d, f in zip(docs["valid"], fl) if f["orig"] and not f["dup_src"]]
    ads = Cp.ad_texts_human()
    ctx.info("data", {"kind": "公开语料 + 人工标注广告（只评估）", "corpus": {
        "source": "wikimedia/wikipedia 20231101.zh 第 3 分片", "license": "CC-BY-SA-4.0", **cstats},
        "ads_eval": {"source": "adextract dev + test 人工广告", "docs": len(ads)}})

    # ---------------- 2. 分词器 ----------------
    def build_tok(d: Path) -> None:
        texts, mix = tokenizer_texts(cfg, docs["train"])
        tok, info = Tk.build(cfg, texts, ctx.log)
        tok.save(d)
        (d / "info.json").write_text(json.dumps({**info, "mix": mix}, ensure_ascii=False))

    tdir = stage_cache(ctx, "tinylm_tokenizer", tok_key(cfg), build_tok)
    tok = Tk.load(tdir)
    tinfo = json.loads((tdir / "info.json").read_text())

    # ---------------- 3. token 流与压缩率 ----------------
    def build_streams(d: Path) -> None:
        stats = {}
        for name, ds in [(n, docs[n]) for n in SPLITS[:-1]] + [("ads", ads)]:
            np.save(d / f"{name}.npy", to_stream(tok, ds))
            stats[name] = Tk.compression(tok, ds)
        stats["char_split"] = Tk.char_split(tok, Cp.char_profile(docs["valid"]))
        (d / "stats.json").write_text(json.dumps(stats, ensure_ascii=False))

    sdir = stage_cache(ctx, "tinylm_streams", {"v": STREAM_VERSION, "tok": tok_key(cfg)}, build_streams)
    streams = {n: np.load(sdir / f"{n}.npy") for n in SPLITS}
    comp = json.loads((sdir / "stats.json").read_text())
    ctx.info("data", {"tokenizer": tinfo, "tokenized": {
        "tokens_total": int(len(streams["train"])), "bytes_total": comp["train"]["bytes"],
        "bytes_per_token": comp["train"]["bytes_per_token"], "chars_per_token": comp["train"]["chars_per_token"]}})
    sample_wiki = docs["valid"][0][:80]
    ctx.trace("tokenize", {
        "note": "同一段文本被切成的 token（原文片段；字节级 token 解码不出完整字时显示字节）",
        "ad": {"text": ads[0], "tokens": Tk.show(tok, ads[0]), "n": len(tok.encode(ads[0]))},
        "wiki": {"text": sample_wiki, "tokens": Tk.show(tok, sample_wiki), "n": len(tok.encode(sample_wiki))},
        "char_split_examples": comp["char_split"]["examples"],
        "stream_head": [int(i) for i in streams["train"][:24]], "eos_id": tok.eos_id})
    ctx.log(f"压缩率：验证集每 token {comp['valid']['bytes_per_token']} 字节 / {comp['valid']['chars_per_token']} 字；"
            f"广告每条 {comp['ads']['tokens_per_doc']} 个 token")

    # ---------------- 4. 模型 ----------------
    device = pick_device(cfg["device"])
    model = GPT(tok.vocab_size, {**cfg, "eos_id": tok.eos_id}).to(device)
    pc, fl, kv = param_counts(model), flops_per_token(model, cfg), kv_cache_stats(model)
    ffn_name = f"MoE {cfg['moe_experts']} 专家 top-{cfg['moe_top_k']}" if cfg["ffn"] == "moe" else "SwiGLU"
    arch = (f"GPT（{'Post' if cfg['norm_pos'] == 'post' else 'Pre'}-RMSNorm、位置 {cfg['pos_emb']}、注意力 {kv['kind']}、"
            f"{ffn_name}、tied embedding）")
    ctx.info("model", {"arch": arch, "params": pc, "flops": fl, "kv_cache": kv,
                       "device": str(device), "dtype": "float32", "train_mode": "full",
                       "precision": "bf16 autocast" if (cfg["precision"] == "bf16" or (cfg["precision"] == "auto"
                                                                                      and device.type == "cuda")) else "fp32"})
    ctx.log(f"模型：总参数 {pc['total']:,}，其中 embedding {pc['embedding']:,}（{pc['embedding_share']:.0%}）；"
            f"每 token 训练 {fl['train_flops_per_token'] / 1e6:.0f} MFLOPs，输出层占 {fl['lm_head_share']:.0%}；"
            f"KV cache 每 token {kv['bytes_per_token']:,} 字节（bf16）")
    b0 = model.blocks[0]
    dqk = b0.D + (b0.Dr if b0.R else 0)
    hk = b0.H if b0.R else b0.Hkv
    ctx.trace("shapes", {"note": "一层注意力里各张量的形状（B=batch，T=序列）", "attention": kv["kind"],
                         "q": f"[B, {b0.H}, T, {dqk}]", "k": f"[B, {hk}, T, {dqk}]", "v": f"[B, {hk}, T, {b0.D}]",
                         "cached_per_token_layer": kv["elems_per_token_layer"],
                         "cached": "潜向量 c [R] + 共享 RoPE key [Dr]" if b0.R else "k、v [Hkv, D] 各一份",
                         "ffn": ffn_name, "doc_mask": cfg["doc_mask"]})
    steps_budget = budget_steps(cfg, tok.vocab_size, len(streams["train"]), fl["train_flops_per_token"])

    # ---------------- 5. 训练 ----------------
    ctx.stage("train")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    valid = {"stream": streams["valid"], "bytes": comp["valid"]["bytes"]}
    tr = train(ctx, cfg, model, streams["train"], comp["train"]["bytes"], valid, tok.eos_id, device, steps_budget)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None
    curve = tr.pop("valid_curve")
    tr.update(tr.pop("stats"))
    precision, moe = tr.pop("precision"), tr.pop("moe")
    ctx.info("train", {k: tr[k] for k in ("steps", "seconds", "train_tokens", "tokens_per_s", "bytes_per_s")})
    bc = tr["batch_check"]
    ctx.trace("batch", {"note": "第一个 batch 第 1 行的前 8 个输入和目标（next-token：目标是输入右移一位）",
                        "x": bc["x_head"], "y": bc["y_head"], "x_tokens": Tk.show(tok, tok.decode(bc["x_head"])),
                        "y_eq_x_frac": bc["y_eq_x_frac"], "target_offset": cfg["target_offset"],
                        "steps": tr["steps"], "epochs": tr["epochs"], "budget_steps": steps_budget})
    if cfg["ffn"] == "moe":
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            x16 = torch.from_numpy(streams["valid"][:16].astype(np.int64)).to(device)[None]
            h = model.wte(x16)
            r = model.blocks[0].moe.router(model.blocks[0].ln2(h)).softmax(-1)[0].topk(cfg["moe_top_k"], -1)
        ctx.trace("routing", {"note": "验证集前 16 个 token 在第 1 层选中的专家（近似：用 embedding 直接路由）",
                              "tokens": Tk.show(tok, tok.decode(x16[0].tolist())), "experts": r.indices.tolist(),
                              "weights": [[round(v, 3) for v in row] for row in r.values.tolist()]})

    # ---------------- 6. 评估 ----------------
    ctx.stage("evaluate")
    ev = {}
    for name in EVAL_SPLITS:
        ev[name] = evaluate(model, streams[name], comp[name]["bytes"], tok.eos_id, cfg, device)
        ctx.log(f"[{name}] {comp[name]['bytes'] / 1e6:.2f} MB  每 token loss {ev[name]['loss']}  bpb {ev[name]['bpb']}")
        ctx.check()

    # L02-L04 的补充评估
    T_full = len(streams["train"])
    seen_win = max(1, min(tr["unique_tokens"], int(10e6 / max(comp["train"]["bytes_per_token"], 1e-9))) // cfg["seq_len"])
    ev["seen"] = evaluate(model, streams["train"], comp["train"]["bytes"], tok.eos_id, cfg, device, max_windows=seen_win)
    ev["valid_long"] = (evaluate(model, streams["valid"], comp["valid"]["bytes"], tok.eos_id, cfg, device, T=2 * cfg["seq_len"])
                        if cfg["pos_emb"] != "learned" else {"loss": None, "bpb": None})
    ev["valid_t512"] = (ev["valid"] if cfg["seq_len"] == 512 else
                        evaluate(model, streams["valid"], comp["valid"]["bytes"], tok.eos_id, cfg, device, T=512))
    ev["ads_isolated"] = evaluate_docs(model, [tok.encode(a) for a in ads], comp["ads"]["bytes"], tok.eos_id, cfg, device)
    valid_at = {}
    for key, frac in VALID_AT.items():
        hit = [c for c in curve if abs(c["train_tokens"] / max(T_full, 1) - frac) < 0.02]
        valid_at[key] = hit[-1]["bpb"] if hit else None
    ptf = peak_tflops(device)
    achieved = fl["train_flops_per_token"] * tr["tokens_per_s"] / 1e12

    def diff(a: str, b: str) -> float | None:
        return round(ev[a]["bpb"] - ev[b]["bpb"], 5) if ev[a]["bpb"] is not None and ev[b]["bpb"] is not None else None

    gap = {"test_minus_valid_bpb": diff("test", "valid"),
           "dup_minus_uniq_bpb": diff("valid_dup", "valid_uniq"),      # 跨场景配对比较用（同一批文档）
           "leaked_minus_uniq_bpb": diff("valid_leaked", "valid_uniq"),
           "leak_few_minus_uniq_bpb": diff("valid_leak_few", "valid_uniq"),
           "leak_many_minus_uniq_bpb": diff("valid_leak_many", "valid_uniq"),
           "valid_bytes_frac": {k: round(comp[f"valid_{k}"]["bytes"] / max(comp["valid"]["bytes"], 1), 4)
                                for k in ("leaked", "leak_few", "leak_many", "dup", "uniq")}}
    gap["valid_minus_seen_bpb"] = diff("valid", "seen")
    pflops = tr["train_tokens"] * fl["train_flops_per_token"]
    tr["tokens_per_param"] = round(tr["train_tokens"] / pc["total"], 3)
    tr["unique_mb"] = round(tr["unique_tokens"] * comp["train"]["bytes_per_token"] / 1e6, 2)
    six_nd = fl["six_n_per_token"] * tr["train_tokens"]
    flops = {**fl, "train_total": pflops, "train_total_pf": round(pflops / 1e15, 4), "six_nd": six_nd,
             "ratio_vs_6nd": round(pflops / six_nd, 4), "achieved_tflops": round(achieved, 2), "peak_tflops": ptf,
             "mfu": round(achieved / ptf, 4) if ptf else None}
    extra_display = {}
    if cfg["ffn"] == "moe" and moe:
        extra_display["专家负载 最大/平均"] = moe["load_max_over_mean"]
    if kv["kind"] != "mha":
        extra_display["KV cache 每 token 字节"] = kv["bytes_per_token"]
    if cfg["epochs"] > 1 or cfg["data_frac"] < 1:
        extra_display["验证 − 已见数据 bpb"] = gap["valid_minus_seen_bpb"]
    if flops["mfu"] is not None:
        extra_display["MFU"] = flops["mfu"]
    if peak_gb is not None:
        extra_display["训练峰值显存 GB"] = peak_gb
    if precision["dlogits_zero_frac"]:
        extra_display["logits 梯度为 0 的比例"] = precision["dlogits_zero_frac"]
    ctx.summary({
        **ev, "gap": gap, "compression": {k: comp[k] for k in ("train", "valid", "test", "ads")},
        "char_split": comp["char_split"], "params": pc, "flops": flops, "train": tr, "valid_at": valid_at,
        "kv_cache": kv, "precision": precision, "moe": moe,
        "memory": {"peak_train_gb": peak_gb, "static_est_gb": round(16 * pc["total"] / 1e9, 3)},
        "attn": {"doc_mask": cfg["doc_mask"], "impl": "sdpa + 显式块对角 mask" if cfg["doc_mask"] else "sdpa is_causal"},
        "valid_curve": curve, "corpus": {k: cstats[k] for k in ("train_docs", "train_bytes", "train_copy_frac",
                                                                 "train_copy_bytes_frac", "valid_docs", "valid_copy_docs",
                                                                 "valid_leak_frac", "valid_leaked_train_copies_mean",
                                                                 "valid_self_dup_frac", "valid_boilerplate_frac", "test_docs")}
        | {"source_dedup": cstats.get("source_dedup"), "dedup_removed": cstats["dedup"]["removed"],
           "injected_copies": cstats["injected"]["copies"]},
        "headline": {"valid_bpb": ev["valid"]["bpb"], "test_bpb": ev["test"]["bpb"], "ads_bpb": ev["ads"]["bpb"],
                     "valid_loss": ev["valid"]["loss"], "ads_tokens_per_doc": comp["ads"]["tokens_per_doc"],
                     "embedding_share": pc["embedding_share"]},
        "display": {"验证 bpb": ev["valid"]["bpb"], "留出 bpb": ev["test"]["bpb"], "广告 bpb": ev["ads"]["bpb"],
                    "验证 每 token loss": ev["valid"]["loss"], "每 token 字节数": comp["valid"]["bytes_per_token"],
                    "广告 每条 token 数": comp["ads"]["tokens_per_doc"], "embedding 占比": pc["embedding_share"],
                    "被转载原文 − 未转载原文 bpb": gap["dup_minus_uniq_bpb"],
                    "训练 token（百万）": round(tr["train_tokens"] / 1e6, 2), "字节/秒（百万）": round(tr["bytes_per_s"] / 1e6, 3),
                    **extra_display},
        "display_pct": ["embedding 占比", "MFU", "logits 梯度为 0 的比例"],
    })

    # ---------------- 7. 保存与重新加载 ----------------
    path = ctx.artifact("model.pt")
    torch.save({"state_dict": model.state_dict(), "vocab_size": tok.vocab_size,
                "cfg": {k: cfg[k] for k in MODEL_KEYS}}, path)
    ctx.register_artifact("model.pt", "checkpoint", "小 GPT 权重（tied embedding）")
    ck = torch.load(path, map_location=device)
    reloaded = GPT(ck["vocab_size"], {**cfg, **ck["cfg"], "eos_id": tok.eos_id}).to(device)
    reloaded.load_state_dict(ck["state_dict"])
    again = evaluate(reloaded, streams["ads"], comp["ads"]["bytes"], tok.eos_id, cfg, device)
    same = abs(again["bpb"] - ev["ads"]["bpb"]) < 1e-4
    ctx.trace("reload", {"ads_bpb_before": ev["ads"]["bpb"], "ads_bpb_after": again["bpb"], "ok": same})
    ctx.info("checks", {"reload_identical": same})
    ctx.log(f"重新加载核对：广告 bpb {ev['ads']['bpb']} → {again['bpb']}（{'一致' if same else '不一致'}）")
