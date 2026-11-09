<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 内容向量做的 ID](10-content.md) · [课程首页](../README.md) · [不加去重位 →](12-nodedup.md)

# 第 11 步 / 共 16 步：随机向量做的 ID

> 阶段 5 / 8：**ID 质量** · 场景：`sid_random`（对照 `baseline`）

语义 ID 换成用随机向量量化：码就是一个随机编号，前缀没有任何含义。模型、训练完全相同。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：3 × 64 + 去重位，trie 约束，beam 10 | — | 3 |
| `sid_random` | 本步 | 用随机向量做语义 ID（码没有语义） | `item_repr=random` | 3 |

## 先预测

**Recall@10 和协同 ID 比怎样？**

- **A.** 低 0.01 以上
- **B.** 差不多（±0.01）
- **C.** 高 0.01 以上

> 先选定一个（记下来），再展开下面的实测。Recall@10 在 6040 个测试用户上的抽样标准误约 0.005；均值差小于 0.01 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · ML-1M · 语义 ID 3 × 64 + 去重位 · encoder-decoder（4 + 2 层）· 3 种子（已有 6 / 6 个运行）</summary>

| 场景 | 类型 | Recall@10 | 碰撞率 | Recall@10：与 `baseline` 差 | Recall@10：各种子 | 断言 |
| - | - | - | - | - | - | - |
| `baseline` | 基线 | 0.220 ± 0.002 | 21.2% ± 0.0% | — | s0 0.222 / s1 0.220 / s2 0.219 | 通过 |
| `sid_random` | 候选改动 | 0.121 ± 0.002 | 1.0% ± 0.1% | -0.099 | s0 0.123 / s1 0.121 / s2 0.118 |  |

![Recall@10（均值，误差线为种子间最小到最大）](figures/h20x8-p_random-metrics.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `baseline` [s0](../../../results/h20x8/R04/baseline/s0-1eaff1/) · [s1](../../../results/h20x8/R04/baseline/s1-e8343c/) · [s2](../../../results/h20x8/R04/baseline/s2-24d603/)
- `sid_random` [s0](../../../results/h20x8/R04/sid_random/s0-2425cf/) · [s1](../../../results/h20x8/R04/sid_random/s1-840a80/) · [s2](../../../results/h20x8/R04/sid_random/s2-789b1d/)

</details>

---

[← 内容向量做的 ID](10-content.md) · [课程首页](../README.md) · [不加去重位 →](12-nodedup.md)
