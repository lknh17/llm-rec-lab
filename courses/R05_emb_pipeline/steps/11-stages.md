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

<details><summary>展开实测：H20 · 微基准 + 28 张表的 DLRM · 3 种子（已有 0 / 3 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 端到端换稠密 Adam](10-e2e-dense.md) · [课程首页](../README.md) · [不开 worker →](12-workers0.md)
