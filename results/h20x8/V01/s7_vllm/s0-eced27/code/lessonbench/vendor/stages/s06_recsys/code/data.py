"""MovieLens-1M：6040 用户、3706 部电影、100 万条评分（全部当作隐式反馈）。

- 每个用户的行为按时间排序，leave-one-out 切分：最后 1 个是测试目标，倒数第 2 个是验证目标，其余用于训练。
- 物品 id 重新编号为 1..N（0 留给 padding）。
- 物品嵌入：用训练序列的共现矩阵做 PPMI + SVD（协同过滤嵌入），拼上类型（genre）的 multi-hot。
  线上的语义 ID 通常用内容理解模型（你们团队的 LLM）产出的物品向量；这里用协同 + 类型作为替代。
"""

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "ml-1m"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass
class SeqData:
    n_items: int
    train: list[list[int]]  # 每个用户的训练序列（去掉最后 2 个）
    valid_target: list[int]
    test_target: list[int]
    full: list[list[int]]  # 完整序列
    genres: np.ndarray  # [n_items + 1, n_genres] multi-hot
    titles: list[str]

    def history(self, user: int, split: str) -> list[int]:
        """split='valid' 返回去掉最后 2 个的序列；'test' 返回去掉最后 1 个的序列。"""
        return self.full[user][:-2] if split == "valid" else self.full[user][:-1]

    def target(self, user: int, split: str) -> int:
        return self.valid_target[user] if split == "valid" else self.test_target[user]


def download() -> None:
    if (DATA / "ratings.dat").exists():
        return
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA.parent)


def load() -> SeqData:
    download()
    rows = [ln.split("::") for ln in (DATA / "ratings.dat").read_text(encoding="latin-1").splitlines()]
    movies = [ln.split("::") for ln in (DATA / "movies.dat").read_text(encoding="latin-1").splitlines()]
    used = sorted({int(r[1]) for r in rows})
    remap = {m: i + 1 for i, m in enumerate(used)}
    by_user: dict[int, list[tuple[int, int, int]]] = {}
    for idx, (u, m, _, t) in enumerate(rows):
        by_user.setdefault(int(u), []).append((int(t), idx, remap[int(m)]))
    full = [[m for _, _, m in sorted(v)] for _, v in sorted(by_user.items())]

    genre_names = sorted({g for mv in movies for g in mv[2].split("|")})
    genres = np.zeros((len(used) + 1, len(genre_names)), dtype=np.float32)
    titles = [""] * (len(used) + 1)
    for mid, title, gs in movies:
        if int(mid) in remap:
            i = remap[int(mid)]
            titles[i] = title
            for g in gs.split("|"):
                genres[i, genre_names.index(g)] = 1
    return SeqData(len(used), [s[:-2] for s in full], [s[-2] for s in full], [s[-1] for s in full], full, genres, titles)


# --8<-- [start:item_emb]
# --8<-- [start:item_emb_cooc]
def item_embeddings(data: SeqData, dim: int = 64, window: int = 5, genre_weight: float = 0.5) -> torch.Tensor:
    """PPMI-SVD 协同嵌入 + 类型 multi-hot。返回 [n_items + 1, dim + n_genres]（第 0 行是 padding）。

    PPMI(i, j) = max(0, log(P(i,j) / (P(i)P(j))))：两个物品在同一用户的相邻 window 内共现得多，说明相似。
    SVD 取前 dim 个奇异向量，是 word2vec（SGNS）的闭式近似（Levy & Goldberg 2014）。
    """
    # n = 3706 个物品 + 1 行 padding = 3707
    n = data.n_items + 1
    # 稠密共现矩阵 [n, n]：3707² × 4 B ≈ 55 MB；百万物品时是 4 TB，必须换稀疏矩阵
    co = np.zeros((n, n), dtype=np.float32)
    for seq in data.train:  # 只用训练部分，避免验证/测试目标泄漏
        s = np.array(seq)
        # 距离为 w 的物品对：s[:-w] 是前一个，s[w:] 是后一个，两者逐位配对
        # 例：s=[a,b,c,d]、w=1 时配出 (a,b) (b,c) (c,d)
        for w in range(1, window + 1):
            # np.add.at 是"带重复下标的累加"：同一对出现两次就加两次
            # 普通写法 co[i, j] += 1 遇到重复下标只会加一次
            np.add.at(co, (s[:-w], s[w:]), 1.0)
    # 只统计了"前 -> 后"，加上转置变成无向共现，矩阵对称
    co = co + co.T
    # --8<-- [end:item_emb_cooc]
    # --8<-- [start:item_emb_ppmi]
    # 所有共现次数之和，用来把计数变成概率
    total = co.sum()
    # 边缘概率 P(i)：按行求和再归一化。keepdims=True 保留被求和的那一维
    # 结果是 [n, 1] 而不是 [n]，后面才能和 [1, n] 广播成 [n, n]
    p_i = co.sum(1, keepdims=True) / total  # [n, 1]
    # 没共现过的对 log(0) = -inf、padding 行 0/0 = nan，先关掉警告，下一行统一处理
    with np.errstate(divide="ignore", invalid="ignore"):
        # PMI = log(P(i,j) / (P(i)·P(j)))；p_i * p_i.T 是 [n,1]×[1,n] 广播出的 [n, n]
        # 大于 0 表示"比随机共现得多"，即协同意义上的相似
        pmi = np.log((co / total) / (p_i * p_i.T))  # [n, n]
    # PPMI：负值截成 0（负相关的估计噪声很大），nan / inf 也置 0
    ppmi = np.nan_to_num(np.maximum(pmi, 0), posinf=0)  # [n, n]
    # --8<-- [end:item_emb_ppmi]
    # --8<-- [start:item_emb_svd]
    # 稠密 SVD 是 O(n³)：n=3707 时约 5e10 次运算，几秒；百万物品要换随机 SVD
    # full_matrices=False 只返回前 n 个奇异向量（经济型分解）
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)  # u [n, n]，s [n]
    # 取前 dim 个方向，乘 sqrt(奇异值)：PPMI ≈ (U√S)(√S Vᵀ)，每行就是物品向量
    cf = u[:, :dim] * np.sqrt(s[:dim])  # [n, dim]
    # 按行 L2 归一化：只保留方向。热门物品的范数不再压过冷门物品
    cf /= np.linalg.norm(cf, axis=1, keepdims=True) + 1e-8  # [n, dim]
    # 类型 multi-hot 也按行归一化：多类型电影每个类型分到的权重更小
    g = data.genres / (np.linalg.norm(data.genres, axis=1, keepdims=True) + 1e-8)  # [n, 18]
    # 拼接：协同 64 维 + 类型 18 维。genre_weight 决定类型在 k-means 距离里占多大比重
    emb = np.concatenate([cf, genre_weight * g], axis=1)  # [n, 82]
    # padding 行置 0（它不是真实物品，调用方会用 [1:] 去掉它）
    emb[0] = 0
    return torch.from_numpy(emb.astype(np.float32))  # [n, 82]
    # --8<-- [end:item_emb_svd]
# --8<-- [end:item_emb]


# --8<-- [start:make_batch]
# --8<-- [start:make_batch_window]
def make_batch(seqs: list[list[int]], batch: int, max_len: int, rng: np.random.Generator,
               min_len: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """随机取 batch 个用户，各截一个随机终点的窗口（长度 ≤ max_len+1），左侧补 0。
    返回 x [B, L]、y [B, L]（padding 位置是 -100，交叉熵忽略）、doc_ids [B, L]（padding 是 -1，真实 token 是 0）。"""
    xs, ys = [], []
    # 有放回地随机抽 batch 个用户；rng.integers(lo, hi) 的 hi 不含
    users = rng.integers(0, len(seqs), size=batch)
    for u in users:
        s = seqs[u]
        # 随机终点：end 在 [min_len, len(s)] 里均匀取，同一个用户每次抽到的窗口不同
        # 相当于做了"时间截断"的数据增强，一个长序列能提供很多样本
        end = int(rng.integers(min_len, len(s) + 1))
        # 取终点前最近的 max_len+1 个物品；多取 1 个是因为 x、y 要错开一位
        w = s[max(0, end - max_len - 1):end]
        # 不够长就在左边补；pad 加 len(w)-1 正好等于 max_len
        pad = max_len + 1 - len(w)
        # x：去掉最后一个物品；y：去掉第一个。第 i 位的 y 就是第 i 位 x 的下一个物品
        # 和 LLM 预训练的 next-token 预测完全一样，每个位置都有监督信号
        xs.append([0] * pad + w[:-1])
        # -100 是 F.cross_entropy 默认的 ignore_index：padding 位置不算 loss
        ys.append([-100] * pad + w[1:])
    # --8<-- [end:make_batch_window]
    # --8<-- [start:make_batch_tensor]
    x, y = torch.tensor(xs), torch.tensor(ys)  # [B, L]，L = max_len
    # torch.where(条件, a, b)：逐元素选，条件为真取 a，否则取 b
    # padding 位置 doc_id = -1，真实位置 = 0：复用 S4 packing 的"同 doc 才能互相看"mask
    doc = torch.where(x == 0, -1, 0)  # [B, L]
    return x, y, doc
    # --8<-- [end:make_batch_tensor]
# --8<-- [end:make_batch]
