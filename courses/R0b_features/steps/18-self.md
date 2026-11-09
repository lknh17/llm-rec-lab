<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 诊断：全量统计的分数从哪来](17-diagnose.md) · [课程首页](../README.md) · [回顾 →](19-keycard.md)

# 第 18 步 / 共 19 步：只多算了本行的标签

> 阶段 6 / 7：**口径泄漏** · 场景：`stats_incl_self`（对照 `f_all`）

更隐蔽的版本：统计严格按时间截断（没有未来），只是"截止到本行"时把本行也算进去了——比如用 `<=` 而不是 `<`，或者曝光日志和点击日志 join 时把当前这次点击也计入了历史点击。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `f_all` | 对照 | 5 组特征全加（时点正确），5 个种子 | `feature_groups=['ids', 'user_profile', 'item_attr', 'user_stats', 'item_stats']` | 5 |
| `fault_stats_full` | 参照 | 全部特征，统计用全量日志按用户 / 电影聚合后拼回每一行 | `feature_groups=['ids', 'user_profile', 'item_attr', 'user_stats', 'item_stats']`, `stats_mode=full` | 3 |
| `stats_incl_self` | 本步 | 全部特征，统计里多算了本行自己的标签 | `feature_groups=['ids', 'user_profile', 'item_attr', 'user_stats', 'item_stats']`, `stats_mode=include_self` | 3 |

## 先预测

**同一批验证行，"场景口径 − 线上口径"的 AUC 差有多大？**

- **A.** 不到 0.3 个点
- **B.** 0.3 - 1 个点
- **C.** 1 个点以上

> 先选定一个（记下来），再展开下面的实测。AUC 差异小于 0.3 个百分点（GAUC 小于 0.5）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M 全量 · 在 R0a 基线上逐组加特征、换统计口径 · 3-5 种子（已有 0 / 11 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

分组表里，哪一组的"虚高"最大？

---

[← 诊断：全量统计的分数从哪来](17-diagnose.md) · [课程首页](../README.md) · [回顾 →](19-keycard.md)
