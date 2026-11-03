"""模型与目标：离散特征 -> embedding -> 拼接 -> MLP -> 一个 logit。

和工业 CTR 精排模型的骨架一致（Embedding + MLP，即 DNN / Wide&Deep 的 deep 部分），
只是字段少、规模小。特征组由配置决定，不改代码就能做特征消融。
"""

from __future__ import annotations

import torch
from torch import nn

# 每个特征组包含哪些字段。sparse 走 nn.Embedding，dense 直接作为数值拼进 MLP 输入
GROUP_FIELDS = {
    "ids": {"sparse": ["user_id", "movie_id"], "dense": [], "bag": False},
    "user_profile": {"sparse": ["gender", "age", "occupation", "zip1"], "dense": [], "bag": False},
    "item_attr": {"sparse": ["year_bucket"], "dense": [], "bag": True},  # bag = 题材多值 pooling
    "user_stats": {"sparse": ["user_cnt_bucket"], "dense": ["user_like_rate"], "bag": False},
    "item_stats": {"sparse": ["item_cnt_bucket"], "dense": ["item_like_rate"], "bag": False},
}


def fields_for(groups: list[str]) -> tuple[list[str], list[str], bool]:
    sparse, dense, bag = [], [], False
    for g in groups:
        sparse += GROUP_FIELDS[g]["sparse"]
        dense += GROUP_FIELDS[g]["dense"]
        bag = bag or GROUP_FIELDS[g]["bag"]
    return sparse, dense, bag


# --8<-- [start:model]
class RankingModel(nn.Module):
    def __init__(self, vocab_sizes: dict, groups: list[str], emb_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        self.sparse, self.dense, self.use_genres = fields_for(groups)
        # 每个离散字段一张表：[vocab, d]。索引 0 是 OOV/冷启动，也有自己的向量
        self.emb = nn.ModuleDict({f: nn.Embedding(vocab_sizes[f], emb_dim) for f in self.sparse})
        if self.use_genres:
            # 一部电影最多 6 个题材：查 6 个向量后求平均（padding_idx=0 不参与平均）
            self.genre_bag = nn.EmbeddingBag(vocab_sizes["genre"], emb_dim, mode="mean", padding_idx=0)
        in_dim = emb_dim * (len(self.sparse) + int(self.use_genres)) + len(self.dense)
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        self.in_dim = in_dim
        for e in self.emb.values():
            nn.init.normal_(e.weight, std=0.05)

    def features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.emb[f](batch[f]) for f in self.sparse]           # 每个 [B, d]
        if self.use_genres:
            parts.append(self.genre_bag(batch["genres"]))              # [B, 6] -> [B, d]
        if self.dense:
            parts.append(torch.stack([batch[f] for f in self.dense], dim=-1))  # [B, n_dense]
        return torch.cat(parts, dim=-1)                                # [B, in_dim]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.mlp(self.features(batch)).squeeze(-1)              # [B] logit，未过 sigmoid
# --8<-- [end:model]


# --8<-- [start:loss]
def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """二分类交叉熵，batch 内取平均。

    用 with_logits 版本：内部用 log-sum-exp 计算，logit 很大时不会出现 log(0)。
    每个样本的权重相同，所以 batch 的 loss = 这 B 条样本各自 loss 的平均；
    要做样本加权（例如按曝光位置去偏）就改这里。
    """
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
# --8<-- [end:loss]


def build_model(meta: dict, cfg: dict) -> RankingModel:
    hidden = [int(h) for h in str(cfg["hidden"]).split(",") if h.strip()]
    return RankingModel(meta["vocab_sizes"], cfg["feature_groups"], cfg["embedding_dim"], hidden, cfg["dropout"])
