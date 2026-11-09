<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 不锁页、阻塞拷贝](13-nopin.md) · [课程首页](../README.md) · [诊断：GPU 在等什么 →](15-diagnose.md)

# 第 14 步 / 共 17 步：同事改了特征处理

> 阶段 6 / 7：**故障** · 场景：`fault_pyfeat`（对照 `baseline`）

同事给特征处理加了一个新规则，顺手把整批 numpy 改成了"更好读"的逐样本、逐特征 Python 循环。结果在单元测试里完全一致。上线训练后，nvidia-smi 里 GPU 利用率时高时低。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：查表微基准（1000 万 × 64 fp32，bag 1 / 10 / 50，均匀与 Zipf）、SparseAdam 与 Adam、数据管线、端到端 DLRM | — | 3 |
| `fault_pyfeat` | 本步 | 特征处理改成逐样本、逐特征的 Python 循环 | `parts=loader,e2e`, `feature_proc=python` | 3 |

## 先预测

**端到端吞吐是基线的多少？**

- **A.** 不到 30%
- **B.** 30% - 70%
- **C.** 70% 以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 微基准 + 28 张表的 DLRM · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 不锁页、阻塞拷贝](13-nopin.md) · [课程首页](../README.md) · [诊断：GPU 在等什么 →](15-diagnose.md)
