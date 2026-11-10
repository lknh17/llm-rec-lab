<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 表规模与精度](08-grid.md) · [课程首页](../README.md) · [端到端换稠密 Adam →](10-e2e-dense.md)

# 第 9 步 / 共 17 步：SparseAdam 对稠密 Adam

> 阶段 4 / 7：**优化器** · 场景：`baseline`

1000 万 × 64 fp32 的表（2.56 GB），一步 = 查表 → 求和 → 反向 → 优化器。一批 16384 × 10 个 id，碰到的行不到表的 2%。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 本步 | 基线：查表微基准（1000 万 × 64 fp32，bag 1 / 10 / 50，均匀与 Zipf）、SparseAdam 与 Adam、数据管线、端到端 DLRM | — | 3 |

## 先预测

**稠密 Adam 每步耗时是 SparseAdam 的几倍？**

- **A.** 不到 2 倍
- **B.** 2 - 10 倍
- **C.** 10 倍以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 微基准 + 28 张表的 DLRM · 3 种子（已有 3 / 3 个运行）</summary>

| 场景 | 类型 | SparseAdam ms | 稠密 Adam ms | 稠密 增量峰值 GB | SparseAdam ms：各种子 | 断言 |
| - | - | - | - | - | - | - |
| `baseline` | 基线 | 1.166 ± 0.002 | 15.245 ± 0.008 | 2.56 ± 0.00 | s0 1.169 / s1 1.165 / s2 1.165 | 通过 |

![SparseAdam ms（均值，误差线为种子间最小到最大）](figures/h20x8-p_optim-metrics.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `baseline` [s0](../../../results/h20x8/R05/baseline/s0-6c6d33/) · [s1](../../../results/h20x8/R05/baseline/s1-3c85f1/) · [s2](../../../results/h20x8/R05/baseline/s2-de361b/)

</details>

---

[← 表规模与精度](08-grid.md) · [课程首页](../README.md) · [端到端换稠密 Adam →](10-e2e-dense.md)
