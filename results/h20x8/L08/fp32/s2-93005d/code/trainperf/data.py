"""数据：GPU 上直接生成随机 token（成本为 0）；或者每条样本在 CPU 上做纯 Python 规范化 + 贪心最长匹配分词（L08 的故障）。"""

from __future__ import annotations

import random
import unicodedata

import torch

WORDS = ["recommend", "content", "understanding", "advertiser", "keyword", "creative", "video", "audience", "conversion",
         "landing", "budget", "category", "Ünïcödé", "Ｆｕｌｌｗｉｄｔｈ", "naïve", "café", "多语言", "标签", "商品", "直播"]


# --8<-- [start:gpu_random]
def gpu_random(mbs: int, T: int, vocab: int, device, gen: torch.Generator) -> torch.Tensor:
    """[mbs, T + 1] 随机 token，直接在设备上生成：输入是前 T 个，目标是后 T 个。"""
    return torch.randint(0, vocab, (mbs, T + 1), device=device, generator=gen)
# --8<-- [end:gpu_random]


# --8<-- [start:cpu_pipeline]
def normalize(text: str) -> str:
    """逐字符：NFKC、去重音、小写、把非字母数字换成空格。"""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        base = unicodedata.normalize("NFKD", ch)[0]
        out.append(base.lower() if base.isalnum() else " ")
    return "".join(out)


def bpe_merges(ids: list[int], rounds: int, vocab: int) -> list[int]:
    """在线 BPE：每一轮数一遍相邻 token 对、把最常见的那一对合并成一个新 token。纯 Python，每轮 O(长度)。
    rounds 控制 CPU 工作量（慢分词器、逐样本的在线处理就是这个量级）。"""
    for _ in range(rounds):
        counts: dict[tuple[int, int], int] = {}
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break
        pair = max(counts, key=counts.get)
        new = (pair[0] * 1_000_003 + pair[1]) % (vocab - 1) + 1
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


class GreedyTokenizer:
    """词表 = 字符 + 常见子串（最长 6 个字符）；从左到右每次取最长的匹配。纯 Python，O(字符数 × 6) 次字典查找。"""

    def __init__(self, vocab: int, seed: int = 0):
        rng = random.Random(seed)
        pieces = set()
        for w in WORDS:
            n = normalize(w)
            for i in range(len(n)):
                for j in range(i + 1, min(len(n), i + 6) + 1):
                    pieces.add(n[i:j])
        pieces |= {chr(c) for c in range(32, 127)}
        self.ids = {p: (rng.randrange(1, vocab)) for p in sorted(pieces)}

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for L in range(6, 0, -1):
                tok = self.ids.get(text[i:i + L])
                if tok is not None:
                    out.append(tok)
                    i += L
                    break
            else:
                out.append(0)
                i += 1
        return out


class TextDataset(torch.utils.data.Dataset):
    """每条样本：随机拼一段"原始文本" → 规范化 → 贪心分词 → prep_passes 轮在线 BPE 合并 → 截断 / 补齐到 T + 1。
    所有工作都在 __getitem__ 里按样本做；文本按 3T 个词准备，合并之后仍够 T + 1 个 token。"""

    def __init__(self, T: int, vocab: int, passes: int, n: int, seed: int):
        self.T, self.vocab, self.passes, self.n, self.seed = T, vocab, passes, n, seed
        self.tok = GreedyTokenizer(vocab, seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        text = " ".join(rng.choice(WORDS) for _ in range(3 * self.T + 8))
        ids = bpe_merges(self.tok.encode(normalize(text)), self.passes, self.vocab)[: self.T + 1]
        ids += [0] * (self.T + 1 - len(ids))
        return torch.tensor(ids, dtype=torch.long)


def cpu_loader(cfg: dict, vocab: int, rank: int, n_batches: int) -> torch.utils.data.DataLoader:
    ds = TextDataset(cfg["seq_len"], vocab, cfg["prep_passes"], n_batches * cfg["mbs"], cfg["seed"] * 100 + rank)
    kw = {"batch_size": cfg["mbs"], "num_workers": cfg["num_workers"], "pin_memory": cfg["pin_memory"], "shuffle": False}
    if cfg["num_workers"] > 0:
        kw["prefetch_factor"] = cfg["prefetch_factor"]
    return torch.utils.data.DataLoader(ds, **kw)
# --8<-- [end:cpu_pipeline]
