<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← leave-last-out：每个用户最后 5 条](06-user-last.md) · [课程首页](../README.md) · [诊断：4 个点从哪来 →](08-diagnose.md)

# 第 7 步 / 共 13 步：随机切分

> 阶段 3 / 5：**切分与验证虚高** · 场景：`fault_random`（对照 `baseline`）

同事觉得"时间切分让验证集太小、太偏"，把 t90 之前的行随机 8:1 分成训练和验证，其余代码不变。他的周报写的是"验证 AUC 提升 4 个点，建议上线"。留出集（t90 之后）没动。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：全局时间切分（训练 < t80 ≤ 验证 < t90 ≤ 留出），只用 ID，1 个 epoch | — | 3 |
| `split_user_last` | 参照 | t90 之前每个用户最后 5 条进验证（leave-last-out） | `split=user_last` | 3 |
| `fault_random` | 本步 | t90 之前的行随机 8:1 切分训练 / 验证（留出集不变） | `split=random` | 3 |

## 先预测

**随机切分的"验证 − 留出"AUC 差，比基线的这个差大多少？**

- **A.** 大不到 1 个点
- **B.** 大 1 - 3 个点
- **C.** 大 3 个点以上

> 先选定一个（记下来），再展开下面的实测。AUC 差异小于 0.3 个百分点或小于种子间波动，都按“差不多”算。不同切分的验证 AUC 不能直接比，要看固定留出集。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M 全量 · 用户 ID + 电影 ID 的 Embedding + MLP · 3 种子（已有 9 / 9 个运行）</summary>

| 场景 | 类型 | 验证 AUC | 留出 AUC | 验证 − 留出 | 验证 GAUC | 验证冷启动用户行 | 验证 AUC：与 `baseline` 差 | 验证 AUC：各种子 | 断言 |
| - | - | - | - | - | - | - | - | - | - |
| `baseline` | 基线 | 0.7447 ± 0.0010 | 0.7531 ± 0.0003 | -0.0084 ± +0.0011 | 0.7302 ± 0.0002 | 76.2% ± 0.0% | — | s0 0.7447 / s1 0.7437 / s2 0.7458 | 通过 |
| `split_user_last` | 候选改动 | 0.7841 ± 0.0007 | 0.7595 ± 0.0004 | +0.0247 ± +0.0010 | 0.6734 ± 0.0020 | 0.0% ± 0.0% | +0.0394 | s0 0.7844 / s1 0.7833 / s2 0.7847 | 通过 |
| `fault_random` | 教学故障 | 0.7899 ± 0.0008 | 0.7585 ± 0.0004 | +0.0314 ± +0.0010 | 0.7327 ± 0.0003 | 0.0% ± 0.0% | +0.0452 | s0 0.7895 / s1 0.7909 / s2 0.7894 | 通过 |

![验证 AUC（均值，误差线为种子间最小到最大）](figures/h20x8-p_fault-metrics.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `baseline` [s0](../../../results/h20x8/R0a/baseline/s0-461376/) · [s1](../../../results/h20x8/R0a/baseline/s1-71c585/) · [s2](../../../results/h20x8/R0a/baseline/s2-570ea7/)
- `split_user_last` [s0](../../../results/h20x8/R0a/split_user_last/s0-ca5532/) · [s1](../../../results/h20x8/R0a/split_user_last/s1-8d3a8f/) · [s2](../../../results/h20x8/R0a/split_user_last/s2-1ccd26/)
- `fault_random` [s0](../../../results/h20x8/R0a/fault_random/s0-2269fc/) · [s1](../../../results/h20x8/R0a/fault_random/s1-6c3980/) · [s2](../../../results/h20x8/R0a/fault_random/s2-b8315b/)

</details>

## 看完实测再想一步

三个场景的留出 AUC 差多少？验证 AUC 呢？

---

[← leave-last-out：每个用户最后 5 条](06-user-last.md) · [课程首页](../README.md) · [诊断：4 个点从哪来 →](08-diagnose.md)
