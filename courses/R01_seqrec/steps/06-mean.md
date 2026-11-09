<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [DIN →](07-din.md)

# 第 6 步 / 共 14 步：mean pooling

> 阶段 3 / 6：**三种模型** · 场景：`mean_l50`（对照 `baseline`）

把最近 50 个行为的 embedding 求平均，过一个线性层，作为用户表示。不看顺序，也不看候选是什么。训练、评估和 SASRec 完全相同。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：SASRec，保留最近 50 个行为 | — | 3 |
| `mean_l50` | 本步 | 历史 embedding 求平均（mean pooling），L = 50 | `model=mean` | 3 |

## 先预测

**mean pooling 和 SASRec 比，测试 Recall@10 怎样？**

- **A.** 低 0.03 以上
- **B.** 差不多（±0.03）
- **C.** 高 0.03 以上

> 先选定一个（记下来），再展开下面的实测。Recall@10 在 6040 个用户上的抽样标准误约 0.005；均值差小于 0.01（模型对比小于 0.03）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M · leave-one-out · 对全部 3706 部电影排序 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

两个模型的全量 Recall 差几倍？采样 Recall 差几倍？

---

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [DIN →](07-din.md)
