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
from .model import GPT, flops_per_token, param_counts
from .train import evaluate, to_stream, train

TOK_VERSION = 1
STREAM_VERSION = 2
CORPUS_KEYS = ("corpus", "min_chars", "source_dedup", "dup_ratio", "dedup", "train_mb", "valid_mb", "test_mb", "data_seed")
SPLITS = ("train", "valid", "test", "valid_leaked", "valid_leak_few", "valid_leak_many", "valid_dup", "valid_uniq", "ads")
EVAL_SPLITS = ("valid", "test", "ads", "valid_leaked", "valid_leak_few", "valid_leak_many", "valid_dup", "valid_uniq")
LEAK_FEW, LEAK_MANY = 2, 8
TOK_KEYS = ("tokenizer", "vocab_size", "tok_train_mb", "tok_ads_ratio")


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
    model = GPT(tok.vocab_size, cfg).to(device)
    pc, fl = param_counts(model), flops_per_token(model, cfg)
    ctx.info("model", {"arch": "GPT（Pre-RMSNorm、RoPE、SwiGLU、tied embedding）", "params": pc, "flops": fl,
                       "device": str(device), "dtype": "float32", "train_mode": "full",
                       "precision": "bf16 autocast" if (cfg["precision"] == "bf16" or (cfg["precision"] == "auto"
                                                                                      and device.type == "cuda")) else "fp32"})
    ctx.log(f"模型：总参数 {pc['total']:,}，其中 embedding {pc['embedding']:,}（{pc['embedding_share']:.0%}）；"
            f"每 token 训练 {fl['train_flops_per_token'] / 1e6:.0f} MFLOPs，输出层占 {fl['lm_head_share']:.0%}")

    # ---------------- 5. 训练 ----------------
    ctx.stage("train")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    valid = {"stream": streams["valid"], "bytes": comp["valid"]["bytes"]}
    tr = train(ctx, cfg, model, streams["train"], comp["train"]["bytes"], valid, tok.eos_id, device)
    curve = tr.pop("valid_curve")
    ctx.info("train", {k: tr[k] for k in ("steps", "seconds", "train_tokens", "tokens_per_s", "bytes_per_s")})

    # ---------------- 6. 评估 ----------------
    ctx.stage("evaluate")
    ev = {}
    for name in EVAL_SPLITS:
        ev[name] = evaluate(model, streams[name], comp[name]["bytes"], tok.eos_id, cfg, device)
        ctx.log(f"[{name}] {comp[name]['bytes'] / 1e6:.2f} MB  每 token loss {ev[name]['loss']}  bpb {ev[name]['bpb']}")
        ctx.check()

    def diff(a: str, b: str) -> float | None:
        return round(ev[a]["bpb"] - ev[b]["bpb"], 5) if ev[a]["bpb"] is not None and ev[b]["bpb"] is not None else None

    gap = {"test_minus_valid_bpb": diff("test", "valid"),
           "dup_minus_uniq_bpb": diff("valid_dup", "valid_uniq"),      # 跨场景配对比较用（同一批文档）
           "leaked_minus_uniq_bpb": diff("valid_leaked", "valid_uniq"),
           "leak_few_minus_uniq_bpb": diff("valid_leak_few", "valid_uniq"),
           "leak_many_minus_uniq_bpb": diff("valid_leak_many", "valid_uniq"),
           "valid_bytes_frac": {k: round(comp[f"valid_{k}"]["bytes"] / max(comp["valid"]["bytes"], 1), 4)
                                for k in ("leaked", "leak_few", "leak_many", "dup", "uniq")}}
    pflops = tr["train_tokens"] * fl["train_flops_per_token"]
    ctx.summary({
        **ev, "gap": gap, "compression": {k: comp[k] for k in ("train", "valid", "test", "ads")},
        "char_split": comp["char_split"], "params": pc, "flops": {**fl, "train_total": pflops}, "train": tr,
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
                    "训练 token（百万）": round(tr["train_tokens"] / 1e6, 2), "字节/秒（百万）": round(tr["bytes_per_s"] / 1e6, 3)},
        "display_pct": ["embedding 占比"],
    })

    # ---------------- 7. 保存与重新加载 ----------------
    path = ctx.artifact("model.pt")
    torch.save({"state_dict": model.state_dict(), "vocab_size": tok.vocab_size,
                "cfg": {k: cfg[k] for k in ("d_model", "n_layer", "n_head", "d_ff", "seq_len")}}, path)
    ctx.register_artifact("model.pt", "checkpoint", "小 GPT 权重（tied embedding）")
    ck = torch.load(path, map_location=device)
    reloaded = GPT(ck["vocab_size"], {**cfg, **ck["cfg"]}).to(device)
    reloaded.load_state_dict(ck["state_dict"])
    again = evaluate(reloaded, streams["ads"], comp["ads"]["bytes"], tok.eos_id, cfg, device)
    same = abs(again["bpb"] - ev["ads"]["bpb"]) < 1e-4
    ctx.trace("reload", {"ads_bpb_before": ev["ads"]["bpb"], "ads_bpb_after": again["bpb"], "ok": same})
    ctx.info("checks", {"reload_identical": same})
    ctx.log(f"重新加载核对：广告 bpb {ev['ads']['bpb']} → {again['bpb']}（{'一致' if same else '不一致'}）")
