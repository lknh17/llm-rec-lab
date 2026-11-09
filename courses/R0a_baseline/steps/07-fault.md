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

<details><summary>展开实测：H20 · MovieLens-1M 全量 · 用户 ID + 电影 ID 的 Embedding + MLP · 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

三个场景的留出 AUC 差多少？验证 AUC 呢？

---

[← leave-last-out：每个用户最后 5 条](06-user-last.md) · [课程首页](../README.md) · [诊断：4 个点从哪来 →](08-diagnose.md)
