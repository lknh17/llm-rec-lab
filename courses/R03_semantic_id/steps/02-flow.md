<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一部电影 →](03-trace.md)

# 第 2 步 / 共 15 步：流程结构

> 阶段 1 / 7：**理解任务**

从物品向量到语义 ID 和统计。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 物品向量 | [N, D]：协同 64 维 / 内容 896 维 / 合成 128 维 | 协同向量来自和 R01 同一套代码训的 SASRec，只用 seq[:-2] 训练。 | [`genrec/items.py#L65-L87`](../../../projects/genrec/items.py#L65-L87) |
| 2. 归一化 | [N, D] → 每行除以范数 | k-means 用欧氏距离，归一化后等价于按余弦聚类。 |  |
| 3. k-means | [N, D] → 中心 [K, D]、分配 [N] | k-means++ 初始化 + Lloyd 迭代；距离分块算；空簇用最远点重置。 | [`genrec/sid.py#L26-L48`](../../../projects/genrec/sid.py#L26-L48) |
| 4. 逐层残差 | r_l = r_{l−1} − C_l[c_l]，L 层 → codes [N, L] | 每层对上一层没解释掉的部分聚类。 | [`genrec/sid.py#L53-L74`](../../../projects/genrec/sid.py#L53-L74) |
| 5. 去重位 | [N, L] → [N, L+1] | 码完全相同的物品依次编号；排序 + 分组，向量化。 | [`genrec/sid.py#L79-L97`](../../../projects/genrec/sid.py#L79-L97) |
| 6. 统计 | 碰撞率 / 均匀基准 / 利用率 / 困惑度 / 重建 / NMI / 跨视角 | 4 层运行的前 d 层就是 d 层的结果（逐层贪心）。 | [`genrec/sid.py#L102-L156`](../../../projects/genrec/sid.py#L102-L156) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一部电影 →](03-trace.md)
