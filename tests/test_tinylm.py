"""tinylm（L01 起的小 GPT 项目）：清洗、副本注入与去重、分词器压缩率、bits-per-byte 的口径、模型形状与参数量。
不需要下载数据集：语料用合成文档替换 load_raw。"""

import math
import random

import numpy as np
import pytest
import torch

from tinylm import corpus as Cp
from tinylm import tokenizer as Tk
from tinylm.model import GPT, flops_per_token, padded_vocab, param_counts
from tinylm.params import PARAMS
from tinylm.train import evaluate, lr_at, to_stream

SMALL = {"d_model": 32, "n_layer": 2, "n_head": 2, "d_ff": 64, "seq_len": 16, "batch_size": 4, "precision": "fp32"}


def _docs(n: int = 400, seed: int = 0) -> list[str]:
    """互不相同的"文章"：随机汉字组成的句子，长度 60-200 字。"""
    rng = random.Random(seed)
    chars = [chr(c) for c in range(0x4E00, 0x4E00 + 800)]
    out = []
    for i in range(n):
        sents = ["".join(rng.choice(chars) for _ in range(rng.randint(10, 30))) + "。" for _ in range(rng.randint(4, 8))]
        out.append(f"第{i}篇" + "".join(sents))
    return out


def test_clean_removes_empty_brackets_and_blank_lines():
    assert Cp.clean("劉馥（）字奇甫\n\n\n\n生平（，）") == "劉馥字奇甫\n\n生平"


def test_minhash_merges_near_copies_but_not_distinct_docs():
    docs = _docs(200)
    rng = random.Random(1)
    near = [Cp.near_copy(d, rng) for d in docs[:50]]
    rep, _ = Cp.minhash_clusters(docs + near)

    def jaccard(a, b):
        sa, sb = set(Cp.shingle_hashes(a).tolist()), set(Cp.shingle_hashes(b).tolist())
        return len(sa & sb) / len(sa | sb)

    js = [jaccard(docs[i], near[i]) for i in range(50)]
    hi = [i for i in range(50) if js[i] >= 0.85]
    lo = [i for i in range(50) if js[i] < 0.5]
    assert hi and lo  # 近似副本的相似度有高有低（删掉的句子占比不同）
    assert sum(rep[200 + i] != 200 + i for i in hi) >= 0.9 * len(hi)   # 很像的副本几乎都被找到
    assert sum(rep[200 + i] != 200 + i for i in lo) <= 0.2 * len(lo)   # 相似度低于阈值的大多漏掉：这是阈值的代价
    assert sum(1 for i in range(200) if rep[i] != i) == 0             # 互不相同的文章不会被合并


def test_dedup_modes_and_leak_bookkeeping(monkeypatch):
    docs = _docs(600)
    monkeypatch.setattr(Cp, "load_raw", lambda corpus: docs)
    base = PARAMS.resolve({"train_mb": 0.2, "valid_mb": 0.02, "test_mb": 0.02, "dup_ratio": 0.4, "min_chars": 10})
    res = {how: Cp.build(dict(base, dedup=how), log=lambda *_: None) for how in ("none", "exact", "minhash")}
    removed = {how: r["stats"]["dedup"]["removed"] for how, r in res.items()}
    assert removed["none"] == 0 < removed["exact"] < removed["minhash"]
    copies = {how: r["stats"]["train_copy_frac"] for how, r in res.items()}
    assert copies["minhash"] < copies["exact"] < copies["none"]
    # 干净留出集在注入副本之前拿走：三种去重方式下完全相同
    assert res["none"]["test"] == res["exact"]["test"] == res["minhash"]["test"]
    leak = {how: r["stats"]["valid_leak_frac"] for how, r in res.items()}
    assert leak["minhash"] < leak["exact"] < leak["none"]
    origs = {}
    for how, r in res.items():
        f = r["valid_flags"]
        assert len(f["leaked"]) == len(f["orig"]) == len(f["dup_src"]) == len(f["boiler"]) == len(r["valid"])
        assert r["stats"]["valid_leak_frac"] == pytest.approx(sum(f["leaked"]) / max(len(r["valid"]), 1), abs=1e-4)
        origs[how] = [d for d, o in zip(r["valid"], f["orig"]) if o]
    # 去重时原文优先保留、切分按来源哈希：验证集里的原文三种方式下是同一批（才能跨场景配对比较）
    assert origs["none"] == origs["exact"] == origs["minhash"] and origs["none"]
    assert len(res["none"]["valid"]) > len(res["minhash"]["valid"])   # 不去重时验证集里还多了副本


def test_n_copies_is_long_tailed_and_capped():
    rng = random.Random(0)
    ks = [Cp.n_copies(rng) for _ in range(20000)]
    assert min(ks) >= 1 and max(ks) <= Cp.MAX_COPIES
    assert sum(k == 1 for k in ks) / len(ks) > 0.4          # 多数只转载 1 次
    assert sum(k >= 10 for k in ks) / len(ks) > 0.05         # 少数被转载 10 次以上


def test_bytes_tokenizer_is_utf8():
    tok = Tk.BytesTokenizer()
    assert tok.encode("跑鞋") == list("跑鞋".encode())
    assert len(tok.encode("跑鞋")) == 6 and tok.vocab_size == 257
    assert Tk.compression(tok, ["跑鞋轻量"])["bytes_per_token"] == 1.0


def test_bpe_compresses_better_with_larger_vocab():
    texts = _docs(300)
    small, big = Tk.train_bpe(texts, 400), Tk.train_bpe(texts, 2000)
    cs, cb = Tk.compression(small, texts), Tk.compression(big, texts)
    assert 1.0 < cs["bytes_per_token"] < cb["bytes_per_token"]
    assert small.decode(small.encode(texts[0])) == texts[0]        # 字节级 BPE 无损
    assert small.eos_id is not None and small.eos_id < small.vocab_size


def test_to_stream_puts_eos_between_docs():
    tok = Tk.BytesTokenizer()
    s = to_stream(tok, ["ab", "c"])
    assert s.tolist() == [256, 97, 98, 256, 99, 256]


def test_model_shapes_tied_embedding_and_param_counts():
    m = GPT(257, SMALL)
    x = torch.randint(0, 257, (3, 16))
    assert m(x).shape == (3, 16, padded_vocab(257))
    pc = param_counts(m)
    assert pc["embedding"] == padded_vocab(257) * 32
    assert pc["total"] == pc["embedding"] + pc["body"]
    body = 2 * (3 * 32 * 32 + 32 * 32 + 2 * 32 * 64 + 64 * 32 + 2 * 32) + 32     # 每层 qkv/proj/gate_up/down/2 个 norm + ln_f
    assert pc["body"] == body
    assert flops_per_token(m, SMALL)["train_flops_per_token"] > 6 * pc["body"]


def test_bpb_of_uniform_model_is_log2_vocab_over_bytes_per_token():
    """输出全 0 的模型 = 均匀分布：每 token NLL = ln V，bpb = log2(V) × token 数 / 字节数。"""
    tok = Tk.BytesTokenizer()
    m = GPT(257, SMALL)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    text = ["跑鞋轻量透气" * 5]
    stream = to_stream(tok, text)
    ev = evaluate(m, stream, len(text[0].encode()), tok.eos_id, SMALL, torch.device("cpu"))
    V = padded_vocab(257)
    assert ev["loss"] == pytest.approx(math.log(V), rel=1e-4)
    assert ev["bpb"] == pytest.approx(math.log2(V) * ev["tokens"] / ev["bytes"], rel=1e-3)
    empty = evaluate(m, np.array([tok.eos_id], dtype=np.int32), 0, tok.eos_id, SMALL, torch.device("cpu"))
    assert empty["bpb"] is None


def test_lr_schedule_warmup_then_cosine_to_10pct():
    cfg = {"lr": 1.0, "warmup_frac": 0.1}
    assert lr_at(0, 100, cfg) == pytest.approx(0.1)
    assert lr_at(9, 100, cfg) == pytest.approx(1.0)
    assert lr_at(99, 100, cfg) == pytest.approx(0.1, abs=1e-3)
