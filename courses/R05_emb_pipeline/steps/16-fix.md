<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 诊断：GPU 在等什么](15-diagnose.md) · [课程首页](../README.md) · [回顾 →](17-keycard.md)

# 第 16 步 / 共 17 步：加 worker 能救吗

> 阶段 6 / 7：**故障** · 场景：`pyfeat_workers16`（对照 `fault_pyfeat`）

不改代码，Python 特征处理保持不变，把 worker 从 2 个加到 16 个。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `fault_pyfeat` | 对照 | 特征处理改成逐样本、逐特征的 Python 循环 | `parts=loader,e2e`, `feature_proc=python` | 3 |
| `baseline` | 参照 | 基线：查表微基准（1000 万 × 64 fp32，bag 1 / 10 / 50，均匀与 Zipf）、SparseAdam 与 Adam、数据管线、端到端 DLRM | — | 3 |
| `pyfeat_workers16` | 本步 | Python 特征处理，worker 加到 16 | `parts=loader,e2e`, `feature_proc=python`, `num_workers=16` | 3 |

## 先预测

**端到端吞吐能恢复到基线的多少？**

- **A.** 不到 50%
- **B.** 50% - 90%
- **C.** 90% 以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 微基准 + 28 张表的 DLRM · 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 诊断：GPU 在等什么](15-diagnose.md) · [课程首页](../README.md) · [回顾 →](17-keycard.md)
