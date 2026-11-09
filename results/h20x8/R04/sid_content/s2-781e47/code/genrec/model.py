"""语义 ID 的 token 化、GPU 前缀树（trie）、encoder-decoder（TIGER）和 beam search。

token：0 = padding，1 = BOS；之后依次是第 1..L 层的码（每层 K 个）和去重位。各层 token 区间不重叠。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

PAD, BOS = 0, 1


class Tokenizer:
    """item（1..N）↔ token 元组。codes [N, D]：前 levels 位是语义码（每位 K 个取值），之后（如果有）是去重位。"""

    def __init__(self, codes: torch.Tensor, k: int, levels: int):
        self.depth = codes.shape[1]
        self.sizes = [k] * levels + ([int(codes[:, levels:].max()) + 1] if self.depth > levels else [])
        self.offsets, o = [], 2
        for s in self.sizes:
            self.offsets.append(o); o += s
        self.vocab = o
        tok = codes.cpu() + torch.tensor(self.offsets)
        self.item_tokens = torch.cat([torch.zeros(1, self.depth, dtype=torch.long), tok])   # [N+1, D]，第 0 行是 padding
        self.level_of = torch.full((self.vocab,), -1, dtype=torch.long)
        for j, (o, s) in enumerate(zip(self.offsets, self.sizes)):
            self.level_of[o:o + s] = j


# --8<-- [start:trie]
class TensorTrie:
    """前缀树展开成稠密转移表：next_node[node, token] = 子节点（−1 = 这个前缀下不存在）；leaf_bucket[node] = 叶子对应的桶号。
    约束解码每一步只要一次 gather，没有 Python 循环。"""

    def __init__(self, item_tokens: torch.Tensor, vocab: int, device):
        seqs = item_tokens[1:].tolist()
        ids, edges, leaf = {(): 0}, [], {}
        for item, seq in enumerate(seqs, start=1):
            parent = 0
            for d in range(len(seq)):
                key = tuple(seq[:d + 1])
                if key not in ids:
                    ids[key] = len(ids)
                    edges.append((parent, seq[d], ids[key]))
                parent = ids[key]
            leaf.setdefault(parent, []).append(item)
        n = len(ids)
        nxt = torch.full((n, vocab), -1, dtype=torch.int32)
        for p, t, c in edges:
            nxt[p, t] = c
        self.next_node = nxt.to(device)
        self.leaf_items = leaf                                              # 叶子 → 物品列表（没有去重位时一个桶里可能有多个）
        self.nodes = n
        self.depth = len(seqs[0])
        self.mb = round(nxt.numel() * 4 / 1e6, 3)
        lv = torch.zeros(n, dtype=torch.long)
        for key, i in ids.items():
            lv[i] = len(key)
        self.branching = [round(float(((nxt[lv == d] >= 0).sum(1)).float().mean()), 2) for d in range(self.depth)]

    def allowed(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.next_node[nodes] >= 0                                  # [..., V]

    def step(self, nodes: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.next_node[nodes, tokens].long()
# --8<-- [end:trie]


# --8<-- [start:encdec]
class TigerEncDec(nn.Module):
    """encoder：每个历史物品一个位置，输入 = 它各位 token 的 embedding 之和 + 位置编码，[B, T, d]。
    decoder：从 BOS 开始自回归生成目标物品的 D 个 token，cross-attend encoder 输出。输出层是全词表 V。
    选 encoder-decoder：历史只编码一次，W 条 beam 共享同一份 memory，不用把历史复制 W 份。"""

    def __init__(self, vocab: int, depth: int, cfg: dict):
        super().__init__()
        d, h = cfg["d_model"], cfg["n_head"]
        self.vocab, self.depth = vocab, depth
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(cfg["max_hist"], d)
        self.dpos = nn.Embedding(depth + 1, d)
        mk = lambda: dict(d_model=d, nhead=h, dim_feedforward=4 * d, dropout=cfg["dropout"], batch_first=True, norm_first=True)  # noqa: E731
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(**mk()), cfg["enc_layers"], enable_nested_tensor=False)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(**mk()), cfg["dec_layers"])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)
        nn.init.normal_(self.tok.weight, std=0.02)

    def encode(self, hist_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hist_tokens [B, T, D]（左补齐，padding 物品全为 0）→ memory [B, T, d]、padding mask [B, T]。"""
        B, T, _ = hist_tokens.shape
        pad = hist_tokens[..., 0] == PAD
        x = self.tok(hist_tokens).sum(2) + self.pos(torch.arange(T, device=hist_tokens.device))[None]
        return self.enc(x, src_key_padding_mask=pad), pad

    def decode(self, prefix: torch.Tensor, mem: torch.Tensor, mem_pad: torch.Tensor) -> torch.Tensor:
        """prefix [B', t]（以 BOS 开头）→ 每个位置下一个 token 的 logits [B', t, V]。"""
        t = prefix.shape[1]
        y = self.tok(prefix) + self.dpos(torch.arange(t, device=prefix.device))[None]
        causal = torch.triu(torch.full((t, t), float("-inf"), device=prefix.device), 1)
        return self.out(self.norm(self.dec(y, mem, tgt_mask=causal, memory_key_padding_mask=mem_pad)))

    def forward(self, hist_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        mem, pad = self.encode(hist_tokens)
        prefix = torch.cat([torch.full_like(target_tokens[:, :1], BOS), target_tokens[:, :-1]], 1)
        return self.decode(prefix, mem, pad)                                # [B, D, V]
# --8<-- [end:encdec]


# --8<-- [start:beam]
@torch.no_grad()
def beam_search(model: TigerEncDec, hist_tokens: torch.Tensor, trie: TensorTrie | None, W: int, init: str = "one_root"):
    """按 [B, W] 批量的 beam search：每步在 W·V 个候选里取全局 top-W。返回 token 序列 [B, W, D] 和分数 [B, W]。
    init="one_root"：只有第 0 条 beam 从 0 分开始，其余 −∞，第一步才会展开成 W 条不同的前缀。
    init="all_zero"（故障）：W 条 beam 一模一样，全局 top-W 选到的是同一个 token 的 W 份拷贝。"""
    B = hist_tokens.shape[0]
    mem, pad = model.encode(hist_tokens)
    mem = mem.repeat_interleave(W, 0)                                       # [B·W, T, d]：共享同一份历史编码
    pad = pad.repeat_interleave(W, 0)
    dev = hist_tokens.device
    score = torch.zeros(B, W, device=dev)
    if init == "one_root":
        score[:, 1:] = float("-inf")
    seq = torch.full((B, W, 1), BOS, dtype=torch.long, device=dev)
    nodes = torch.zeros(B, W, dtype=torch.long, device=dev)
    V = model.vocab
    for step in range(model.depth):
        logits = model.decode(seq.view(B * W, -1), mem, pad)[:, -1].float()   # [B·W, V]
        logp = F.log_softmax(logits, -1).view(B, W, V)
        if trie is not None:
            logp = logp.masked_fill(~trie.allowed(nodes), float("-inf"))
        cand = (score[..., None] + logp).view(B, W * V)                     # [B, W·V]
        top, idx = cand.topk(W, dim=1)
        beam, tok = idx // V, idx % V
        seq = torch.cat([seq.gather(1, beam[..., None].expand(-1, -1, seq.shape[2])), tok[..., None]], 2)
        if trie is not None:
            nodes = trie.step(nodes.gather(1, beam).clamp(min=0), tok)
        score = top
    return seq[..., 1:], score
# --8<-- [end:beam]
