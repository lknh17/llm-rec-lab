<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 端到端换稠密 Adam](10-e2e-dense.md) · [课程首页](../README.md) · [不开 worker →](12-workers0.md)

# 第 11 步 / 共 17 步：一步的时间花在哪

> 阶段 5 / 7：**数据管线** · 场景：`baseline`

基线端到端（向量化特征处理、2 个 worker、锁页内存、SparseAdam），按阶段计时：等数据、拷到 GPU、前向 + 反向、优化器。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 本步 | 基线：查表微基准（1000 万 × 64 fp32，bag 1 / 10 / 50，均匀与 Zipf）、SparseAdam 与 Adam、数据管线、端到端 DLRM | — | 3 |

## 先预测

**哪个阶段占比最大？**

- **A.** 等数据
- **B.** 拷到 GPU
- **C.** 前向 + 反向
- **D.** 优化器

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 微基准 + 28 张表的 DLRM · 3 种子（已有 3 / 3 个运行）</summary>

| 场景 | 类型 | 等数据 ms | 拷贝 ms | 前向反向 ms | 优化器 ms | 等数据 ms：各种子 | 断言 |
| - | - | - | - | - | - | - | - |
| `baseline` | 基线 | 0.12 ± 0.01 | 0.12 ± 0.00 | 3.44 ± 0.07 | 12.02 ± 0.15 | s0 0.12 / s1 0.13 / s2 0.12 | 通过 |

![等数据 ms（均值，误差线为种子间最小到最大）](figures/h20x8-p_stages-metrics.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `baseline` [s0](../../../results/h20x8/R05/baseline/s0-6c6d33/) · [s1](../../../results/h20x8/R05/baseline/s1-3c85f1/) · [s2](../../../results/h20x8/R05/baseline/s2-de361b/)

</details>

---

[← 端到端换稠密 Adam](10-e2e-dense.md) · [课程首页](../README.md) · [不开 worker →](12-workers0.md)
