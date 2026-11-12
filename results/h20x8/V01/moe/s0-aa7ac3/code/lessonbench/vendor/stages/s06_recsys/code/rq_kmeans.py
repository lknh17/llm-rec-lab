"""残差量化 k-means（RQ-KMeans）：把物品向量编码成几个离散码组成的"语义 ID"。

    第 1 层：对物品向量做 k-means，得到码 c1，残差 r1 = x - C1[c1]
    第 2 层：对 r1 做 k-means，得到 c2，残差 r2 = r1 - C2[c2]
    ...
    语义 ID = (c1, c2, c3)，相似的物品共享前缀（c1 相同 = 大类相同）。

L 层、每层 K 个码，能表示 K^L 个物品，词表却只有 L·K。3 层 × 256 码 = 1677 万个组合，只要 768 个 token。
碰撞（两个物品的码完全相同）时，追加第 L+1 位去重码（TIGER 的做法）。

和 RQ-VAE（TIGER 原文）的区别：RQ-VAE 端到端训练编码器和码本，RQ-KMeans 直接在固定向量上逐层聚类，
更简单、更稳定（不会码本坍塌），OneRec 等工业系统用的就是 RQ-KMeans（加上均衡约束）。
"""

from dataclasses import dataclass

import torch


# --8<-- [start:kmeans]
# --8<-- [start:kmeans_init]
def kmeans(x: torch.Tensor, k: int, iters: int = 30, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [N, D] -> (centroids [K, D], assign [N])。k-means++ 初始化 + Lloyd 迭代，空簇用最远点重新初始化。"""
    # 独立的随机数生成器：结果只由 seed 决定，不受外部 torch.manual_seed 影响
    g = torch.Generator().manual_seed(seed)
    n = x.size(0)
    # 第一个中心：随机挑 1 个物品。下标是长度 1 的张量，所以结果保持二维
    centroids = x[torch.randint(n, (1,), generator=g)]  # [1, D]
    for _ in range(1, k):  # k-means++：按到最近中心距离的平方成比例采样下一个中心
        # cdist：两组向量两两之间的欧氏距离，[N, D] 和 [k', D] -> [N, k']
        # min(dim=1) 沿中心维取最小：每个点到"最近的已选中心"的距离
        # .values 取最小值本身（.indices 是最近中心的编号）
        d2 = torch.cdist(x, centroids).min(dim=1).values.pow(2)  # [N]
        # multinomial 按权重 d2 抽 1 个下标：离已有中心越远的点越可能被选
        # 这样新中心倾向落在还没被覆盖的区域；+1e-12 防止全 0 权重报错
        # 整个初始化共 K 次 cdist，代价 O(N·K²·D)；N=3706、K=256 时不到 1 秒
        centroids = torch.cat([centroids, x[torch.multinomial(d2 + 1e-12, 1, generator=g)]])  # [k'+1, D]
# --8<-- [end:kmeans_init]
# --8<-- [start:kmeans_lloyd]
    # Lloyd 迭代：分配 -> 更新中心，固定跑 iters 轮（不判收敛，结果可复现）
    for _ in range(iters):
        # 距离矩阵 [N, K]：3706 × 256 × 4 B ≈ 3.8 MB；亿级物品时放不下
        dist = torch.cdist(x, centroids)  # [N, K]
        # argmin(dim=1)：每个点最近中心的编号
        assign = dist.argmin(dim=1)  # [N]
        # bincount：统计每个编号出现几次；minlength=k 保证没被用到的簇也有一个 0
        counts = torch.bincount(assign, minlength=k)  # [K]
        # index_add_(0, assign, x)：把 x 的第 i 行加到结果的第 assign[i] 行（scatter 累加）
        # 一次 kernel 算出每个簇的向量和，再除以簇大小得到均值，没有 Python 循环
        # clamp(min=1) 防止空簇除以 0；[:, None] 把 [K] 变成 [K, 1] 以便广播到 [K, D]
        new = torch.zeros_like(centroids).index_add_(0, assign, x) / counts.clamp(min=1)[:, None]  # [K, D]
        # 空簇：bool 掩码 [K]
        empty = counts == 0
        if empty.any():
            # 每个点到最近中心的距离，取最大的几个：当前"最没被照顾到"的点
            # 用它们重新初始化空簇。这一步保证每层码利用率 100%，不会码本坍塌
            far = dist.min(dim=1).values.topk(int(empty.sum())).indices  # [空簇数]
            # bool 掩码做下标赋值：只覆盖空簇那几行
            new[empty] = x[far]
        centroids = new
    # 用最终的中心再分配一次，保证 assign 和返回的 centroids 一致
    assign = torch.cdist(x, centroids).argmin(dim=1)  # [N]
    return centroids, assign
# --8<-- [end:kmeans_lloyd]
# --8<-- [end:kmeans]


@dataclass
class SemanticIDs:
    codes: torch.Tensor  # [N, L+1]：L 层残差码 + 1 位去重码
    codebooks: list[torch.Tensor]  # 每层 [K, D]
    k: int
    recon_error: list[float]  # 每层之后的平均相对重建误差 ||x - x̂|| / ||x||
    collision_rate: float  # 前 L 位相同、需要去重码的物品比例
    utilization: list[float]  # 每层实际用到的码占 K 的比例

    @property
    def levels(self) -> int:
        return self.codes.size(1)


# --8<-- [start:rq_kmeans]
# --8<-- [start:rq_levels]
def rq_kmeans(x: torch.Tensor, levels: int = 3, k: int = 64, iters: int = 30, seed: int = 0) -> SemanticIDs:
    """x: [N, D] 物品向量（不含 padding 行）。"""
    # clone 复制一份：后面要反复减残差，不能改调用方传进来的 x
    residual = x.clone()  # [N, D]
    # 每个物品原始向量的 L2 范数，用来算相对误差；clamp 防止除以 0
    norm = x.norm(dim=1).clamp(min=1e-8)  # [N]
    codes, books, errs, util = [], [], [], []
    for level in range(levels):
        # 对"上一层没解释掉的部分"做 k-means；每层换一个种子
        c, a = kmeans(residual, k, iters, seed + level)  # c [K, D]，a [N]
        # c[a]：用码 a 查码本，得到每个物品这一层的重建向量 [N, D]
        # 减掉它，剩下的残差交给下一层。第 1 层抓大类，后面每层在大类内细分
        residual = residual - c[a]  # [N, D]
        codes.append(a)
        books.append(c)
        # 这一层之后的相对重建误差 ||x - x̂|| / ||x||，对所有物品取平均
        errs.append(float((residual.norm(dim=1) / norm).mean()))
        # 码利用率：这一层实际出现过几种码 / K
        util.append(float(a.unique().numel() / k))
    # stack(dim=1)：L 个 [N] 叠成 [N, L]，每行是一个物品的 (c1, c2, c3)
    codes = torch.stack(codes, dim=1)  # [N, L]
# --8<-- [end:rq_levels]
# --8<-- [start:rq_dedup]
    # 去重码：前 L 位相同的物品依次编号 0, 1, 2...
    dedup = torch.zeros(len(codes), dtype=torch.long)  # [N]
    # 字典：码组合 -> 已经出现过几次。tolist 转成 Python 列表，tuple 才能当字典键
    seen: dict[tuple, int] = {}
    # 纯 Python 循环，O(N)；3706 个物品毫秒级，亿级物品要换成排序 + 分组计数
    for i, row in enumerate(map(tuple, codes.tolist())):
        # 第一次出现的组合拿 0，第二次拿 1……
        dedup[i] = seen.get(row, 0)
        seen[row] = dedup[i].item() + 1
    # 碰撞物品数：出现次数大于 1 的组合，里面的所有物品（含拿到 0 的那个）都算
    collided = sum(v for v in seen.values() if v > 1)
    # dedup[:, None]：[N] -> [N, 1]，拼到最后一列，得到 [N, L+1]
    return SemanticIDs(torch.cat([codes, dedup[:, None]], dim=1), books, k, errs, collided / len(codes), util)
# --8<-- [end:rq_dedup]
# --8<-- [end:rq_kmeans]


# --8<-- [start:tokenizer]
# --8<-- [start:tokenizer_vocab]
class SemanticTokenizer:
    """语义 ID <-> token。第 l 层的码 c 映射到 token offset_l + c，各层的 token 不重叠。

    token 0 是 padding；之后依次是第 1..L 层的码，最后是去重码。
    """

    def __init__(self, sid: SemanticIDs):
        self.sid = sid
        # levels 是 L+1（含去重位），3 层 RQ 时是 4
        self.levels = sid.levels
        # 去重位只需要"最大去重码 + 1"个 token，不用给满 K 个
        max_dedup = int(sid.codes[:, -1].max()) + 1
        # 每一位各占多少个 token：K=64 时 [64, 64, 64, 10]
        self.sizes = [sid.k] * (self.levels - 1) + [max_dedup]
        # 每一位的起始 token；从 1 开始，因为 0 是 padding
        self.offsets = [1]
        for s in self.sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
        # K=64 时 offsets = [1, 65, 129, 193]，词表 193 + 10 = 203
        self.vocab_size = self.offsets[-1] + self.sizes[-1]
    # --8<-- [end:tokenizer_vocab]
    # --8<-- [start:tokenizer_encode]
        # item（1..N）-> token 序列；第 0 行留给 padding
        # [N, L+1] + [L+1] 广播：每一列加上自己那一位的 offset，各位的 token 区间不重叠
        tokens = sid.codes + torch.tensor(self.offsets)  # [N, L+1]
        # 在最上面拼一行全 0，使"物品 id i 对应第 i 行"，查表时不用减 1
        self.item_tokens = torch.cat([torch.zeros(1, self.levels, dtype=torch.long), tokens])  # [N+1, L+1]
        # 反查表：token 元组 -> 物品 id。加了去重位后每个元组唯一，所以字典不会冲突
        self.token_to_item = {tuple(row): i + 1 for i, row in enumerate(tokens.tolist())}

    def encode(self, items: list[int]) -> list[int]:
        # 整数下标 gather 出 [n, L+1]，flatten 拉平成 n·(L+1) 个 token
        # 50 个物品 -> 200 个 token，这就是 TIGER 输入序列长 4 倍的来源
        return self.item_tokens[items].flatten().tolist()

    def decode(self, tokens: list[int]) -> int | None:
        # 不存在的组合返回 None（"幻觉物品"），trie 约束解码就是为了不让它发生
        return self.token_to_item.get(tuple(tokens))
    # --8<-- [end:tokenizer_encode]
# --8<-- [end:tokenizer]
