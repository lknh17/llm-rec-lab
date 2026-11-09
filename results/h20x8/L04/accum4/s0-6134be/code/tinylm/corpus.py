"""语料：读取 → 清洗 → 底稿去重 → 留出干净测试集 → 模拟转载（注入副本）→ 去重 → 切分训练 / 验证。

顺序是这一课的关键：
- 底稿去重先去掉维基自带的模板化近重复条目，保证之后 test 和 valid 同分布、各场景的 test 完全相同；
- 干净留出集（test）在注入副本**之前**拿走，和训练集没有任何副本关系，是"真实泛化"的标尺；
- 验证集（valid）在去重**之后**按（来源, 副本号）哈希切分（相当于按 URL 哈希）。不去重时，同一篇文章的副本
  会分别落进训练集和验证集，验证集 loss 被"背过的原文"拉低——这就是 fault_nodedup 要复现的泄漏；
- 去重时原文优先保留，所以各场景验证集里的原文是同一批文档："被转载过的原文"和"没被转载过的原文"
  两组在每个模型上的 bpb 差，可以跨场景配对比较（差中差），文档内容本身的难易被抵消。
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from collections import Counter

import numpy as np

from labkit import assets

CORPUS_VERSION = 4
ASSET = {"zhwiki": "zhwiki-2311-s2"}
BOILERPLATE = ["本文转载自网络，如有侵权请联系删除。", "（来源：网络整理）", "更多精彩内容，请关注我们。",
               "声明：本文仅代表作者观点。", "原标题：", "编辑：小编"]
_EMPTY_BRACKETS = re.compile(r"（[，、；：\s]*）|\([,;:\s]*\)|「\s*」")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_SENT = re.compile(r"(?<=[。！？!?\n])")


# ------------------------------------------------------------------ 读取与清洗
def load_raw(corpus: str) -> list[str]:
    import pyarrow.parquet as pq

    files = assets.resolve_dataset_files(ASSET[corpus])
    texts: list[str] = []
    for f in files:
        texts += pq.read_table(f, columns=["text"]).column("text").to_pylist()
    return texts


# --8<-- [start:clean]
def clean(text: str) -> str:
    """维基转纯文本后留下的痕迹：模板被删掉后的空括号"（）"、连续空行。繁简不转换（原文就是混排的）。"""
    text = _EMPTY_BRACKETS.sub("", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
# --8<-- [end:clean]


def norm_key(text: str) -> str:
    """精确去重用的规范化：去掉所有空白后取哈希。"""
    return hashlib.sha1(re.sub(r"\s+", "", text).encode()).hexdigest()


# ------------------------------------------------------------------ 模拟转载
def near_copy(text: str, rng: random.Random) -> str:
    """近似副本：随机删 1-2 句，前后加转载声明。和原文的 5 字 shingle Jaccard 通常在 0.7-0.95。"""
    sents = [s for s in _SENT.split(text) if s]
    if len(sents) > 3:
        for _ in range(rng.randint(1, 2)):
            sents.pop(rng.randrange(1, len(sents)))
    out = "".join(sents)
    if rng.random() < 0.5:
        out = rng.choice(BOILERPLATE) + out
    if rng.random() < 0.5:
        out = out + "\n" + rng.choice(BOILERPLATE)
    return out


MAX_COPIES = 40


def n_copies(rng: random.Random) -> int:
    """副本数是长尾分布（截断 Pareto，α = 1.1）：大多数被转载 1-2 次，少数模板页、热门文章被转载几十次。
    真实网页语料就是这样：Lee et al. 2021 在 C4 里发现个别段落重复了上万次。"""
    return min(MAX_COPIES, int(rng.paretovariate(1.1)))


def inject_duplicates(docs: list[str], ratio: float, rng: random.Random) -> tuple[list[tuple[int, int, str]], set, dict]:
    """从 docs 里抽 ratio 比例的文档，各复制 n_copies() 份（一半逐字相同，一半近似），打乱后返回
    [(来源编号, 副本号, 文本)]（副本号 0 = 原文）和被转载过的来源集合。
    来源编号、副本号只用来切分和事后统计，去重和训练都看不到文本以外的东西。"""
    n_src = int(len(docs) * ratio)
    src = rng.sample(range(len(docs)), n_src) if n_src else []
    copies, exact, near = [], 0, 0
    for i in src:
        for k in range(1, n_copies(rng) + 1):
            if rng.random() < 0.5:
                copies.append((i, k, docs[i]))
                exact += 1
            else:
                copies.append((i, k, near_copy(docs[i], rng)))
                near += 1
    out = [(i, 0, d) for i, d in enumerate(docs)] + copies
    rng.shuffle(out)
    return out, set(src), {"source_docs": n_src, "copies": len(copies), "exact_copies": exact, "near_copies": near,
                           "copy_bytes": sum(len(c.encode()) for _, _, c in copies)}


def in_valid(data_seed: int, src: int, k: int, frac: float) -> bool:
    """按（来源, 副本号）的哈希决定进不进验证集：相当于按 URL 哈希切分，同一篇的不同转载 URL 不同。
    和去重方式无关，所以各场景验证集里的原文是同一批文档，可以跨模型配对比较。"""
    h = hashlib.sha1(f"{data_seed}:{src}:{k}".encode()).digest()
    return int.from_bytes(h[:8], "little") / 2**64 < frac


# ------------------------------------------------------------------ MinHash 去重
# --8<-- [start:minhash]
N_PERM, BANDS, ROWS, SHINGLE, JACCARD = 64, 16, 4, 5, 0.7
_MASK61 = np.uint64((1 << 61) - 1)


def shingle_hashes(text: str, k: int = SHINGLE, max_chars: int = 4000) -> np.ndarray:
    """文档 → 所有 k 字片段（shingle）的 64 位哈希，去重后返回。用 numpy 一次算完，不逐个片段调 Python。"""
    a = np.frombuffer(text[:max_chars].encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)  # [n_chars]
    if len(a) < k:
        a = np.concatenate([a, np.zeros(k - len(a), dtype=np.uint64)])
    n = len(a) - k + 1
    h = np.zeros(n, dtype=np.uint64)
    for j in range(k):  # 多项式滚动哈希，uint64 溢出即取模 2^64
        h = h * np.uint64(1_000_003) + a[j:j + n]
    return np.unique(h)  # [n_shingles]


def signature(h: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """MinHash 签名：64 个随机线性哈希 (a·x + b) mod p 下的最小值。两篇文档签名相同位置的比例 ≈ Jaccard。"""
    x = h & _MASK61                                   # [n]
    v = (x[:, None] * A[None, :] + B[None, :]) & _MASK61  # [n, 64]
    return v.min(axis=0)                               # [64]


def minhash_clusters(docs: list[str], seed: int = 0) -> tuple[list[int], dict]:
    """LSH：签名切成 16 段 × 4 行，任一段完全相同的文档对成为候选，再用签名估计的 Jaccard ≥ 0.7 确认。
    返回每篇文档所属簇的代表（第一次出现的文档下标）。"""
    rs = np.random.default_rng(seed)
    A = rs.integers(1, (1 << 61) - 1, N_PERM, dtype=np.uint64)
    B = rs.integers(0, (1 << 61) - 1, N_PERM, dtype=np.uint64)
    t0 = time.time()
    sigs = np.stack([signature(shingle_hashes(d), A, B) for d in docs])  # [n_docs, 64]
    parent = list(range(len(docs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    candidates = confirmed = 0
    for b in range(BANDS):
        buckets: dict[bytes, int] = {}
        band = sigs[:, b * ROWS:(b + 1) * ROWS]
        for i in range(len(docs)):
            key = band[i].tobytes()
            j = buckets.setdefault(key, i)
            if j != i:
                candidates += 1
                ri, rj = find(i), find(j)
                if ri != rj and float((sigs[i] == sigs[j]).mean()) >= JACCARD:
                    parent[max(ri, rj)] = min(ri, rj)
                    confirmed += 1
    rep = [find(i) for i in range(len(docs))]
    return rep, {"candidate_pairs": candidates, "merged_pairs": confirmed, "seconds": round(time.time() - t0, 1)}
# --8<-- [end:minhash]


def dedup(docs: list[str], how: str, seed: int) -> tuple[list[int], dict]:
    """返回保留的文档下标（保持原顺序）。"""
    if how == "none":
        return list(range(len(docs))), {"method": "none", "removed": 0}
    keep, keys = [], set()
    for i, d in enumerate(docs):
        k = norm_key(d)
        if k not in keys:
            keys.add(k)
            keep.append(i)
    stats = {"method": how, "exact_removed": len(docs) - len(keep)}
    if how == "exact":
        stats["removed"] = stats["exact_removed"]
        return keep, stats
    rep, mh = minhash_clusters([docs[i] for i in keep], seed)
    out = [i for j, i in enumerate(keep) if rep[j] == j]
    stats.update({"near_removed": len(keep) - len(out), "removed": len(docs) - len(out), "minhash": mh})
    return out, stats


# ------------------------------------------------------------------ 广告文案（评估分词压缩率、可选混入 BPE 训练）
def ad_texts_human() -> list[str]:
    """adextract 的人工标注广告（dev + test，60 条），只用来评估，不进任何训练。"""
    from adextract import data as D

    return [r["text"] for r in D.load_evalset("dev") + D.load_evalset("test")]


def ad_texts_synth(n: int, seed: int = 31_000) -> list[str]:
    """模板生成的广告文案（和人工评估集不重叠），只用于 BPE 训练混入。"""
    from adextract import synth

    human = set(ad_texts_human())
    return [r["text"] for r in synth.generate(n, 0.15, seed, exclude=human)]


def take_bytes(docs: list[str], budget_mb: float) -> list[str]:
    out, total, limit = [], 0, int(budget_mb * 1e6)
    for d in docs:
        if total >= limit:
            break
        out.append(d)
        total += len(d.encode())
    return out


def nbytes(docs: list[str]) -> int:
    return sum(len(d.encode()) for d in docs)


# --8<-- [start:build]
def build(cfg: dict, log=print) -> dict:
    """返回 {train, valid, test, stats}。所有随机性来自 data_seed，和训练种子无关。"""
    rng = random.Random(10_007 + cfg["data_seed"])
    raw = load_raw(cfg["corpus"])
    cleaned = [clean(t) for t in raw]
    docs = [t for t in cleaned if len(t) >= cfg["min_chars"]]
    stats = {"raw_docs": len(raw), "raw_bytes": nbytes(raw), "short_dropped": len(cleaned) - len(docs),
             "brackets_removed_bytes": nbytes(raw) - nbytes(cleaned)}
    if cfg["source_dedup"] != "none":                          # ⓪ 底稿去重：维基自带的模板化近重复条目
        before = nbytes(docs)
        keep, sd = dedup(docs, cfg["source_dedup"], cfg["data_seed"])
        docs = [docs[i] for i in keep]
        stats["source_dedup"] = {"removed_docs": sd["removed"], "removed_bytes_frac": round(1 - nbytes(docs) / before, 4)}
    rng.shuffle(docs)
    test = take_bytes(docs, cfg["test_mb"])                    # ① 干净留出：注入副本之前拿走
    rest = docs[len(test):]
    tagged, dup_src, inj = inject_duplicates(rest, cfg["dup_ratio"], rng)  # ② 模拟转载：[(来源, 副本号, 文本)]
    stats["injected"] = inj
    # ③ 去重：只看文本。同一簇保留最先出现的一篇，这里让原文排在副本前面（相当于保留最早发布的版本），
    #    这样各种去重方式下，验证集里的原文是同一批
    order = sorted(range(len(tagged)), key=lambda i: tagged[i][1] > 0)
    kept, dd = dedup([tagged[i][2] for i in order], cfg["dedup"], cfg["data_seed"])
    pool = [tagged[i] for i in sorted(order[j] for j in kept)]  # 恢复打乱后的顺序
    stats["dedup"] = dd
    frac = min(1.0, cfg["valid_mb"] * 1e6 / max(nbytes(rest), 1))  # ④ 按 URL 哈希切分：原文约 valid_mb
    is_valid = [in_valid(cfg["data_seed"], s, k, frac) for s, k, _ in pool]
    valid_t = [p for p, v in zip(pool, is_valid) if v]
    train_all = [p for p, v in zip(pool, is_valid) if not v]
    train_t = train_all[:len(take_bytes([t for _, _, t in train_all], cfg["train_mb"]))]
    train, valid = [t for _, _, t in train_t], [t for _, _, t in valid_t]
    # 验证集里有多少文档的同源副本进了训练集：泄漏的直接证据（只用于统计）
    train_src = Counter(s for s, _, _ in train_t)
    leaked = sum(1 for s, _, _ in valid_t if s in train_src)
    train_copies = len(train_t) - len(train_src)
    valid_self_dup = len(valid_t) - len({s for s, _, _ in valid_t})          # 验证集内部互为副本的文档
    valid_boiler = sum(1 for t in valid if any(b in t for b in BOILERPLATE))  # 带转载声明的文档
    stats.update({"valid_self_dup_frac": round(valid_self_dup / max(len(valid_t), 1), 4),
                  "valid_boilerplate_frac": round(valid_boiler / max(len(valid), 1), 4),
                  "after_dedup_docs": len(pool), "train_docs": len(train), "train_bytes": nbytes(train),
                  "train_copy_docs": train_copies, "train_copy_frac": round(train_copies / max(len(train_t), 1), 4),
                  "train_copy_bytes_frac": round(nbytes([t for _, k, t in train_t if k > 0]) / max(nbytes(train), 1), 4),
                  "train_available_bytes": nbytes([t for _, _, t in train_all]),
                  "valid_docs": len(valid), "valid_bytes": nbytes(valid),
                  "valid_copy_docs": sum(1 for _, k, _ in valid_t if k > 0),
                  "test_docs": len(test), "test_bytes": nbytes(test), "valid_with_train_copy": leaked,
                  "valid_leak_frac": round(leaked / max(len(valid), 1), 4),
                  "valid_leaked_train_copies_mean": round(sum(train_src[s] for s, _, _ in valid_t if s in train_src)
                                                          / max(leaked, 1), 2),
                  "train_chars": sum(len(d) for d in train)})
    sd_removed = stats.get("source_dedup", {}).get("removed_docs", 0)
    log(f"语料：原始 {len(raw)} 篇 → 清洗后 {len(docs) + sd_removed} 篇 → 底稿去重后 {len(docs)} 篇；注入副本 {inj['copies']} 份；"
        f"去重（{cfg['dedup']}）去掉 {dd['removed']} 篇；训练 {len(train)} 篇 {stats['train_bytes'] / 1e6:.1f} MB，"
        f"验证 {len(valid)} 篇（其中 {leaked} 篇在训练集有副本），留出 {len(test)} 篇")
    # 每篇验证文档的标记（只用于分组统计，训练看不到）：
    #   leaked = 同源文档进了训练集；train_n = 训练集里同源文档几篇；orig = 原文（不是副本）；
    #   dup_src = 这篇的来源被转载过；boiler = 带转载声明
    return {"train": train, "valid": valid, "test": test, "stats": stats,
            "valid_flags": {"leaked": [s in train_src for s, _, _ in valid_t],
                            "train_n": [train_src.get(s, 0) for s, _, _ in valid_t],
                            "orig": [k == 0 for _, k, _ in valid_t],
                            "dup_src": [s in dup_src for s, _, _ in valid_t],
                            "boiler": [any(b in t for b in BOILERPLATE) for t in valid]}}
# --8<-- [end:build]


def char_profile(docs: list[str], top: int = 3000) -> list[tuple[str, int]]:
    """出现最多的汉字及次数（用来统计"一个汉字被拆成几个 token"）。"""
    c: Counter = Counter()
    for d in docs:
        c.update(ch for ch in d if "\u4e00" <= ch <= "\u9fff")
    return c.most_common(top)
