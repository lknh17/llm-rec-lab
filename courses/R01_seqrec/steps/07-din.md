<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← mean pooling](06-mean.md) · [课程首页](../README.md) · [只看最近 20 个 →](08-len20.md)

# 第 7 步 / 共 14 步：DIN

> 阶段 3 / 6：**三种模型** · 场景：`din_l50`（对照 `baseline`）

DIN 对每个候选物品重新给历史加权：和候选相似的历史物品权重大。和 mean pooling 一样不看顺序，但"看候选"。每个位置都监督（和 SASRec 一样多的训练信号）。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：SASRec，保留最近 50 个行为 | — | 3 |
| `mean_l50` | 参照 | 历史 embedding 求平均（mean pooling），L = 50 | `model=mean` | 3 |
| `din_l50` | 本步 | DIN（按候选 target attention），L = 50 | `model=din` | 3 |

## 先预测

**DIN 和 SASRec 比，测试 Recall@10 怎样？**

- **A.** 低 0.03 以上
- **B.** 差不多（±0.03）
- **C.** 高 0.03 以上

> 先选定一个（记下来），再展开下面的实测。Recall@10 在 6040 个用户上的抽样标准误约 0.005；均值差小于 0.01（模型对比小于 0.03）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M · leave-one-out · 对全部 3706 部电影排序 · 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

三个模型的训练曲线（valid_recall10）谁先到顶？

---

[← mean pooling](06-mean.md) · [课程首页](../README.md) · [只看最近 20 个 →](08-len20.md)
