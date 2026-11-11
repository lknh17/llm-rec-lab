"""语义 ID 前缀树（trie）：约束 beam search 只生成真实存在的物品。

不加约束时，模型可能生成 (12, 7, 33) 这种不对应任何物品的组合（"幻觉物品"）。
每一步只允许"当前前缀在 trie 里的合法后继 token"，其余 token 的 logit 置为 -inf。

这里是教学版：用 Python 字典存"前缀 -> 合法后继的 bool 向量 [V]"。S7 会把它改成 GPU 上的稠密查表。
"""

import torch


# --8<-- [start:trie]
# --8<-- [start:trie_build]
class Trie:
    def __init__(self, item_token_seqs: list[list[int]], vocab_size: int, device: torch.device | str = "cpu"):
        self.vocab_size = vocab_size
        # 每个物品几个 token（3 层码 + 去重位 = 4）
        self.depth = len(item_token_seqs[0])
        # 前缀（已生成的 token 元组）-> 合法的下一个 token 集合
        children: dict[tuple, set[int]] = {}
        for seq in item_token_seqs:
            # 一个物品贡献 depth 条边：() -> c1，(c1,) -> c2，(c1, c2) -> c3 ...
            for d in range(self.depth):
                # setdefault：键不存在就先放一个空集合，再往里加
                children.setdefault(tuple(seq[:d]), set()).add(seq[d])
        # 预先把每个前缀的合法集合转成 [V] 的 bool 向量，解码时直接拿来用
        # 节点数 ≈ 1 + K + 已用 (c1,c2) 组合数 + 已用 (c1,c2,c3) 组合数，ML-1M 约几千个
        # 每个 V=203 字节，总共 1 MB 量级；亿级物品时要改用 CSR 存稀疏后继
        self.masks: dict[tuple, torch.Tensor] = {}
        for prefix, nxt in children.items():
            m = torch.zeros(vocab_size, dtype=torch.bool)  # [V]
            # 用 token 列表做下标，一次把这些位置置 True
            m[list(nxt)] = True
            self.masks[prefix] = m.to(device)
        # 不在 trie 里的前缀：全 False，这条 beam 下一步没有任何合法 token
        self._empty = torch.zeros(vocab_size, dtype=torch.bool, device=device)  # [V]
# --8<-- [end:trie_build]

# --8<-- [start:trie_allowed]
    def allowed(self, prefixes: list[list[int]]) -> torch.Tensor:
        """prefixes: W 个 beam 已生成的 token -> bool [W, V]"""
        # 每个 beam 查一次字典，再把 W 个 [V] 叠成 [W, V]
        # prefixes 是 Python 列表，调用方每步都要先把 GPU 上选出的 token .tolist() 拷回 CPU
        # 这就是一次 CPU-GPU 同步；S7 改成 GPU 上的稠密查表来消掉它
        return torch.stack([self.masks.get(tuple(p), self._empty) for p in prefixes])  # [W, V]
# --8<-- [end:trie_allowed]
# --8<-- [end:trie]

    def num_nodes(self) -> int:
        return len(self.masks)
