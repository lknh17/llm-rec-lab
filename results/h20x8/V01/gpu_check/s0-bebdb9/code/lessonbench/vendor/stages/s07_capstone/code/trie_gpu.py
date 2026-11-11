"""GPU 友好的语义 ID 前缀树：把 trie 展开成一张稠密转移表，约束解码的每一步只需要一次张量索引。

S6 的 `Trie` 是 Python 字典：每个 beam 每步要用 tuple(prefix) 查一次表、再 stack 成 [W, V]。
B 个用户 × W 个 beam 时就是 B·W 次 Python 调用，并且掩码要从 CPU 拷到 GPU —— 每步都有 CPU-GPU 同步。

这里给每个前缀（trie 节点）编号，建一张表：
    next_node[node, token] = 子节点编号（-1 表示 token 在该前缀下不合法）     [n_nodes, V] int32
于是：
    合法掩码  = next_node[nodes] >= 0          nodes [W] -> [W, V]
    状态转移  = next_node[nodes, tokens]       [W] -> [W]
    叶子节点  -> 物品 id：leaf_item[node]
全部是 GPU 上的 gather，没有 Python 循环，可以放进 CUDA Graph。

显存：n_nodes × V × 4 字节。ML-1M（3706 物品，约 6.5K 节点，V=203）约 5 MB。
千万级物品、V 上千时稠密表放不下（几十 GB），工业做法：
- 按层存：第 l 层的节点只需要第 l 层的 token 范围（K 列），表变成 Σ n_nodes_l × K；
- CSR：每个节点只存它的子节点列表（总长 = 节点数），掩码用 scatter 生成；
- 前两层稠密（节点少），后面的层 CSR。
"""

import torch


class TensorTrie:
    # --8<-- [start:build]
    def __init__(self, item_token_seqs: list[list[int]], vocab_size: int, device: torch.device | str = "cpu"):
        self.depth = len(item_token_seqs[0])
        self.vocab_size = vocab_size
        ids: dict[tuple, int] = {(): 0}
        edges: list[tuple[int, int, int]] = []
        leaf: dict[int, int] = {}
        for item, seq in enumerate(item_token_seqs, start=1):
            parent = 0
            for d in range(self.depth):
                key = tuple(seq[:d + 1])
                if key not in ids:
                    ids[key] = len(ids)
                    edges.append((parent, seq[d], ids[key]))
                parent = ids[key]
            leaf[parent] = item
        n = len(ids)
        self.next_node = torch.full((n, vocab_size), -1, dtype=torch.int32)
        for parent, tok, child in edges:
            self.next_node[parent, tok] = child
        self.leaf_item = torch.full((n,), -1, dtype=torch.long)
        for node, item in leaf.items():
            self.leaf_item[node] = item
        self.next_node = self.next_node.to(device)
        self.leaf_item = self.leaf_item.to(device)
        self.num_nodes = n
    # --8<-- [end:build]

    # --8<-- [start:ops]
    def root(self, n: int) -> torch.Tensor:
        # n 个 beam 都从根节点出发：编号 0 对应空前缀 ()
        # 和转移表放在同一个设备上，后面的索引才不会触发跨设备拷贝
        return torch.zeros(n, dtype=torch.long, device=self.next_node.device)  # [n]

    def allowed(self, nodes: torch.Tensor) -> torch.Tensor:
        """nodes [W] -> bool [W, V]"""
        # next_node[nodes]：用 [W] 个节点号做行索引，一次 gather 取出 W 行，得到 [W, V] 的 int32
        # >= 0 逐元素比较变成 bool：True 表示这个 token 在当前前缀下合法（有子节点）
        # 640 个 beam、V=203 时只读 640 × 203 × 4 B ≈ 0.5 MB，一个 kernel，结果留在 GPU 上，不用同步回 CPU
        return self.next_node[nodes] >= 0  # [W, V]

    def step(self, nodes: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """nodes [W]、tokens [W] -> 子节点 [W]"""
        # 两个 [W] 下标成对索引：第 i 个结果是 next_node[nodes[i], tokens[i]]，得到 [W] 而不是 [W, W]
        # 表用 int32 存（省一半显存），.long() 转回 int64，和 root() 的类型一致，下一步还要当下标用
        # 选中的 token 一定合法，所以结果不会是 -1
        return self.next_node[nodes, tokens].long()  # [W]
    # --8<-- [end:ops]

    def nbytes(self) -> int:
        return self.next_node.numel() * 4 + self.leaf_item.numel() * 8
