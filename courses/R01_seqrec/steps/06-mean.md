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

<details><summary>展开实测：H20 · MovieLens-1M · leave-one-out · 对全部 3706 部电影排序 · 3 种子（已有 6 / 6 个运行）</summary>

| 场景 | 类型 | Recall@10 | NDCG@10 | 采样 100 Recall@10 | 训练末 loss | Recall@10：与 `baseline` 差 | Recall@10：各种子 | 断言 |
| - | - | - | - | - | - | - | - | - |
| `baseline` | 基线 | 0.233 ± 0.004 | 0.129 ± 0.001 | 0.800 ± 0.002 | 5.366 ± 0.011 | — | s0 0.233 / s1 0.229 / s2 0.238 | 通过 |
| `mean_l50` | 候选改动 | 0.083 ± 0.000 | 0.042 ± 0.000 | 0.691 ± 0.005 | 5.783 ± 0.016 | -0.150 | s0 0.083 / s1 0.083 / s2 0.083 |  |

![Recall@10（均值，误差线为种子间最小到最大）](figures/h20x8-p_mean-metrics.svg)

![valid_recall10（均值，阴影为种子间范围）](figures/h20x8-p_mean-loss.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `baseline` [s0](../../../results/h20x8/R01/baseline/s0-585a43/) · [s1](../../../results/h20x8/R01/baseline/s1-87e137/) · [s2](../../../results/h20x8/R01/baseline/s2-471d19/)
- `mean_l50` [s0](../../../results/h20x8/R01/mean_l50/s0-648a16/) · [s1](../../../results/h20x8/R01/mean_l50/s1-aef33a/) · [s2](../../../results/h20x8/R01/mean_l50/s2-9c27fe/)

</details>

## 看完实测再想一步

两个模型的全量 Recall 差几倍？采样 Recall 差几倍？

---

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [DIN →](07-din.md)
