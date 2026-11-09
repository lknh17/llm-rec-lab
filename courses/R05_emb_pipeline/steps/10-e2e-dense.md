<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← SparseAdam 对稠密 Adam](09-optim.md) · [课程首页](../README.md) · [一步的时间花在哪 →](11-stages.md)

# 第 10 步 / 共 17 步：端到端换稠密 Adam

> 阶段 4 / 7：**优化器** · 场景：`e2e_dense_adam`（对照 `baseline`）

端到端 DLRM（28 张表，每张 100 万 × 64，共 7.2 GB）的表从 SparseAdam 换成稠密 Adam。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：查表微基准（1000 万 × 64 fp32，bag 1 / 10 / 50，均匀与 Zipf）、SparseAdam 与 Adam、数据管线、端到端 DLRM | — | 3 |
| `e2e_dense_adam` | 本步 | 端到端改用稠密梯度 + Adam | `parts=e2e`, `emb_optim=adam` | 3 |

## 先预测

**端到端吞吐（样本/秒）是 SparseAdam 的多少？**

- **A.** 不到 30%
- **B.** 30% - 80%
- **C.** 80% 以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 微基准 + 28 张表的 DLRM · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← SparseAdam 对稠密 Adam](09-optim.md) · [课程首页](../README.md) · [一步的时间花在哪 →](11-stages.md)
