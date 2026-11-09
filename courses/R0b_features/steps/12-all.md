<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 加电影历史统计](11-item-stats.md) · [课程首页](../README.md) · [去掉 ID →](13-no-ids.md)

# 第 12 步 / 共 19 步：5 组全加

> 阶段 4 / 7：**逐组加特征** · 场景：`f_all`（对照 `f_user_stats`）

ID + 画像 + 属性 + 用户统计 + 电影统计全部加上（5 个种子）。对照是"只加用户统计"——这是单变量对照：差的是画像、属性、电影统计三组。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `f_user_stats` | 对照 | + 用户历史统计（此前评分次数、平滑喜欢率，时点正确） | `feature_groups=['ids', 'user_stats']` | 3 |
| `baseline` | 参照 | 基线：只用用户 ID + 电影 ID（= R0a 基线），5 个种子 | — | 5 |
| `f_all` | 本步 | 5 组特征全加（时点正确），5 个种子 | `feature_groups=['ids', 'user_profile', 'item_attr', 'user_stats', 'item_stats']` | 5 |

## 先预测

**5 组全加，比只加用户统计又多多少？**

- **A.** 再涨 0.3 个点以上
- **B.** 差不多（±0.3）
- **C.** 反而跌 0.3 个点以上

> 先选定一个（记下来），再展开下面的实测。AUC 差异小于 0.3 个百分点（GAUC 小于 0.5）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M 全量 · 在 R0a 基线上逐组加特征、换统计口径 · 3-5 种子（已有 13 / 13 个运行）</summary>

| 场景 | 类型 | 验证 AUC | 验证 GAUC | 留出 AUC | MLP 输入维度 | 验证 AUC：与 `f_user_stats` 差 | 验证 AUC：各种子 | 断言 |
| - | - | - | - | - | - | - | - | - |
| `f_user_stats` | 候选改动 | 0.7708 ± 0.0007 | 0.7292 ± 0.0003 | 0.7681 ± 0.0010 | 98 ± 0 | — | s0 0.7713 / s1 0.7711 / s2 0.7700 | 通过 |
| `baseline` | 基线 | 0.7449 ± 0.0015 | 0.7304 ± 0.0004 | 0.7533 ± 0.0003 | 64 ± 0 | -0.0259 | s0 0.7447 / s1 0.7437 / s2 0.7458 / s3 0.7469 / s4 0.7433 | 通过 |
| `f_all` | 修复 | 0.7670 ± 0.0014 | 0.7330 ± 0.0005 | 0.7676 ± 0.0009 | 324 ± 0 | -0.0038 | s0 0.7675 / s1 0.7675 / s2 0.7651 / s3 0.7688 / s4 0.7663 | 通过 |

![验证 AUC（均值，误差线为种子间最小到最大）](figures/h20x8-p_all-metrics.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `f_user_stats` [s0](../../../results/h20x8/R0b/f_user_stats/s0-00e791/) · [s1](../../../results/h20x8/R0b/f_user_stats/s1-7d80a6/) · [s2](../../../results/h20x8/R0b/f_user_stats/s2-2b8070/)
- `baseline` [s0](../../../results/h20x8/R0b/baseline/s0-461376/) · [s1](../../../results/h20x8/R0b/baseline/s1-71c585/) · [s2](../../../results/h20x8/R0b/baseline/s2-570ea7/) · [s3](../../../results/h20x8/R0b/baseline/s3-a5952b/) · [s4](../../../results/h20x8/R0b/baseline/s4-e7e664/)
- `f_all` [s0](../../../results/h20x8/R0b/f_all/s0-55a121/) · [s1](../../../results/h20x8/R0b/f_all/s1-b243f6/) · [s2](../../../results/h20x8/R0b/f_all/s2-dd3fa1/) · [s3](../../../results/h20x8/R0b/f_all/s3-2ff37a/) · [s4](../../../results/h20x8/R0b/f_all/s4-7ddea0/)

</details>

## 看完实测再想一步

三个场景的留出 AUC 怎么排？

---

[← 加电影历史统计](11-item-stats.md) · [课程首页](../README.md) · [去掉 ID →](13-no-ids.md)
