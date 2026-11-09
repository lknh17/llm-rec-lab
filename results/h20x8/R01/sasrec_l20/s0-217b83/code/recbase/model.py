"""排序模型：离散特征 → embedding → 拼接 → MLP → 一个 logit；可选 FM 二阶交叉（DeepFM）或 DCN-v2 cross 并联。

和工业 CTR 精排的骨架一致（Embedding + MLP），只是字段少、规模小。特征组由配置决定，不改代码就能做特征消融。
"""

from __future__ import annotations

import torch
from torch import nn

# 每个特征组包含哪些字段。sparse 查 embedding 表；dense 是数值，直接拼进 MLP 输入；genres 是多值字段（求平均）
GROUP_FIELDS = {
    "ids": {"sparse": ["user_id", "movie_id"], "dense": [], "genres": False},
    "user_profile": {"sparse": ["gender", "age", "occupation", "zip1"], "dense": [], "genres": False},
    "item_attr": {"sparse": ["year_bucket"], "dense": [], "genres": True},
    "user_stats": {"sparse": ["user_cnt_bucket"], "dense": ["user_rate", "user_lcnt"], "genres": False},
    "item_stats": {"sparse": ["item_cnt_bucket"], "dense": ["item_rate", "item_lcnt"], "genres": False},
}
CROSS_LAYERS = 3


def fields_for(groups: list[str]) -> tuple[list[str], list[str], bool]:
    sparse, dense, genres = [], [], False
    for g in groups:
        sparse += GROUP_FIELDS[g]["sparse"]
        dense += GROUP_FIELDS[g]["dense"]
        genres = genres or GROUP_FIELDS[g]["genres"]
    return sparse, dense, genres


def parse_hidden(s) -> list[int]:
    return [int(h) for h in str(s).split(",") if h.strip()]


# --8<-- [start:model]
class RecModel(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], cfg: dict):
        super().__init__()
        d, self.arch = cfg["embedding_dim"], cfg["arch"]
        self.sparse, self.dense, self.use_genres = fields_for(cfg["feature_groups"])
        # 每个离散字段一张表 [V, d]。下标 0 = 训练集没见过（OOV / 冷启动），也有自己的向量
        self.emb = nn.ModuleDict({f: nn.Embedding(vocab_sizes[f], d) for f in self.sparse})
        if self.use_genres:  # 一部电影最多 6 个题材：查 6 个向量，padding（0）不参与平均
            self.genre_emb = nn.Embedding(vocab_sizes["genre"], d, padding_idx=0)
        self.n_fields = len(self.sparse) + int(self.use_genres)
        self.in_dim = d * self.n_fields + len(self.dense)
        layers, h = [], self.in_dim
        for w in parse_hidden(cfg["hidden"]):
            layers += [nn.Linear(h, w), nn.ReLU(), nn.Dropout(cfg["dropout"])]
            h = w
        self.mlp = nn.Sequential(*layers)
        if self.arch == "dcn":  # DCN-v2：x_{l+1} = x0 ⊙ (W x_l + b) + x_l，和 MLP 并联，最后拼起来过输出层
            self.cross = nn.ModuleList(nn.Linear(self.in_dim, self.in_dim) for _ in range(CROSS_LAYERS))
            h += self.in_dim
        if self.arch == "deepfm":  # FM 一阶项：每个字段每个取值一个标量
            self.first = nn.ModuleDict({f: nn.Embedding(vocab_sizes[f], 1) for f in self.sparse})
            for e in self.first.values():
                nn.init.zeros_(e.weight)
        self.head = nn.Linear(h, 1)
        for e in self.emb.values():
            nn.init.normal_(e.weight, std=0.05)
        if self.use_genres:
            nn.init.normal_(self.genre_emb.weight, std=0.05)
            with torch.no_grad():
                self.genre_emb.weight[0].zero_()

    def field_vectors(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.emb[f](batch[f]) for f in self.sparse]                # 每个 [B, d]
        if self.use_genres:
            g = batch["genres"]                                             # [B, 6]，0 = padding
            v = self.genre_emb(g)                                           # [B, 6, d]
            k = (g > 0).sum(-1, keepdim=True).clamp(min=1)                  # [B, 1] 实际题材数
            parts.append(v.sum(1) / k)                                      # [B, d] 平均
        return torch.stack(parts, 1)                                        # [B, F, d]

    def concat(self, v: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = v.flatten(1)                                                    # [B, F·d]
        if self.dense:
            x = torch.cat([x, torch.stack([batch[f] for f in self.dense], -1)], -1)  # [B, in_dim]
        return x

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        v = self.field_vectors(batch)                                       # [B, F, d]
        x = self.concat(v, batch)                                           # [B, in_dim]
        h = self.mlp(x)                                                     # [B, h_last]
        if self.arch == "dcn":
            c = x
            for lin in self.cross:
                c = x * lin(c) + c                                          # [B, in_dim]
            h = torch.cat([h, c], -1)
        logit = self.head(h).squeeze(-1)                                    # [B]
        if self.arch == "deepfm":
            s = v.sum(1)                                                    # [B, d]
            logit = logit + 0.5 * (s * s - (v * v).sum(1)).sum(-1)          # FM 二阶：两两内积之和
            logit = logit + sum(self.first[f](batch[f]).squeeze(-1) for f in self.sparse)
        return logit                                                        # [B]，未过 sigmoid
# --8<-- [end:model]


# --8<-- [start:loss]
def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """二分类交叉熵，batch 内平均。with_logits 版本内部用 log-sum-exp，logit 很大时也不会 log(0)。"""
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
# --8<-- [end:loss]


def param_counts(model: RecModel) -> dict:
    emb = sum(p.numel() for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "embedding": emb, "mlp": total - emb, "embedding_share": round(emb / max(total, 1), 4),
            "mlp_input_dim": model.in_dim, "fields": model.n_fields, "dense": len(model.dense)}
