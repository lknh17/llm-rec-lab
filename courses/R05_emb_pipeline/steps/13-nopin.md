<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 不开 worker](12-workers0.md) · [课程首页](../README.md) · [同事改了特征处理 →](14-fault.md)

# 第 13 步 / 共 17 步：不锁页、阻塞拷贝

> 阶段 5 / 7：**数据管线** · 场景：`e2e_nopin`（对照 `baseline`）

关掉锁页内存和 non_blocking：每次拷贝先从可分页内存拷到临时锁页缓冲，再拷到 GPU，并且阻塞。一批约 2 MB。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：查表微基准（1000 万 × 64 fp32，bag 1 / 10 / 50，均匀与 Zipf）、SparseAdam 与 Adam、数据管线、端到端 DLRM | — | 3 |
| `e2e_nopin` | 本步 | 端到端不锁页、阻塞拷贝 | `parts=e2e`, `pin_memory=False`, `non_blocking=False` | 3 |

## 先预测

**拷到 GPU 这一阶段的耗时是基线的几倍？**

- **A.** 不到 1.5 倍
- **B.** 1.5 - 5 倍
- **C.** 5 倍以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 微基准 + 28 张表的 DLRM · 3 种子（已有 6 / 6 个运行）</summary>

| 场景 | 类型 | 拷贝 ms | 样本/秒 | 拷贝 ms：与 `baseline` 差 | 拷贝 ms：各种子 | 断言 |
| - | - | - | - | - | - | - |
| `baseline` | 基线 | 0.121 ± 0.003 | 196,081 ± 4,531 | — | s0 0.124 / s1 0.121 / s2 0.119 | 通过 |
| `e2e_nopin` | 候选改动 | 0.591 ± 0.041 | 177,557 ± 3,734 | +0.470 | s0 0.548 / s1 0.595 / s2 0.630 |  |

![拷贝 ms（均值，误差线为种子间最小到最大）](figures/h20x8-p_nopin-metrics.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `baseline` [s0](../../../results/h20x8/R05/baseline/s0-6c6d33/) · [s1](../../../results/h20x8/R05/baseline/s1-3c85f1/) · [s2](../../../results/h20x8/R05/baseline/s2-de361b/)
- `e2e_nopin` [s0](../../../results/h20x8/R05/e2e_nopin/s0-1c194f/) · [s1](../../../results/h20x8/R05/e2e_nopin/s1-723f31/) · [s2](../../../results/h20x8/R05/e2e_nopin/s2-a15d94/)

</details>

---

[← 不开 worker](12-workers0.md) · [课程首页](../README.md) · [同事改了特征处理 →](14-fault.md)
