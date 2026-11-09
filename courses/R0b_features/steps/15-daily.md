<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 换成 DCN-v2](14-dcn.md) · [课程首页](../README.md) · [全量统计 →](16-fault.md)

# 第 15 步 / 共 19 步：T+1 天级统计

> 阶段 5 / 7：**统计的时效** · 场景：`stats_daily`（对照 `f_all`）

统计特征改成 T+1 天级：只用样本**前一天及更早**的评分（天级离线任务的产出）。不含未来，线上也能拿到——只是晚了一天。其余和"全部特征"相同。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `f_all` | 对照 | 5 组特征全加（时点正确），5 个种子 | `feature_groups=['ids', 'user_profile', 'item_attr', 'user_stats', 'item_stats']` | 5 |
| `stats_daily` | 本步 | 全部特征，统计改成 T+1 天级（只用样本前一天及更早的行为） | `feature_groups=['ids', 'user_profile', 'item_attr', 'user_stats', 'item_stats']`, `stats_mode=daily` | 3 |

## 先预测

**改成天级统计，"训练集没见过的用户"这一组 AUC 掉多少？**

- **A.** 掉不到 0.5 个点
- **B.** 掉 0.5 - 2 个点
- **C.** 掉 2 个点以上

> 先选定一个（记下来），再展开下面的实测。AUC 差异小于 0.3 个百分点（GAUC 小于 0.5）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M 全量 · 在 R0a 基线上逐组加特征、换统计口径 · 3-5 种子（已有 0 / 8 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

天级口径下，"全部特征"还比基线（只用 ID）好多少？

---

[← 换成 DCN-v2](14-dcn.md) · [课程首页](../README.md) · [全量统计 →](16-fault.md)
