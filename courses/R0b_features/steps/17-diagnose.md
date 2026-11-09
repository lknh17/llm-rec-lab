<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 全量统计](16-fault.md) · [课程首页](../README.md) · [只多算了本行的标签 →](18-self.md)

# 第 17 步 / 共 19 步：诊断：全量统计的分数从哪来

> 阶段 6 / 7：**口径泄漏** · 场景：`fault_stats_full`（对照 `f_all`）

配置只差统计口径一个参数，验证 AUC 多涨了一两个点，所有分组都在涨。按需逐级点开线索，然后点选最主要的原因。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `f_all` | 对照 | 5 组特征全加（时点正确），5 个种子 | `feature_groups=['ids', 'user_profile', 'item_attr', 'user_stats', 'item_stats']` | 5 |
| `fault_stats_full` | 本步 | 全部特征，统计用全量日志按用户 / 电影聚合后拼回每一行 | `feature_groups=['ids', 'user_profile', 'item_attr', 'user_stats', 'item_stats']`, `stats_mode=full` | 3 |

## 线索

逐条展开，看到能下判断为止。

<details><summary>线索 1</summary>

线上回放：固定留出集（线上口径特征）上，时点正确 〔待 H20 生成〕，全量统计 〔待 H20 生成〕。离线涨，线上回放持平还是下跌？

</details>

<details><summary>线索 2</summary>

口径体检：用户统计里数进了未来的行占比 〔待 H20 生成〕（时点正确为 〔待 H20 生成〕）。只用"用户喜欢率"排序的 AUC：训练集 〔待 H20 生成〕，同一批验证行换线上口径 〔待 H20 生成〕。

</details>

<details><summary>线索 3</summary>

同一个模型、同一批验证行，只换特征口径：〔待 H20 生成〕 → 〔待 H20 生成〕。样本之前没有任何历史的用户（u_new）：全量口径 〔待 H20 生成〕，时点正确 〔待 H20 生成〕。

</details>

## 先预测

**全量统计的验证 AUC 更高，最主要的原因是什么？**

- **A.** 统计量里含有本行的标签和未来的行为，线上拿不到（穿越）
- **B.** 全量统计样本多，估计更准
- **C.** 模型过拟合
- **D.** 验证集换了一批更容易的样本

> 先选定一个（记下来），再展开下面的实测。AUC 差异小于 0.3 个百分点（GAUC 小于 0.5）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M 全量 · 在 R0a 基线上逐组加特征、换统计口径 · 3-5 种子（已有 0 / 8 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 全量统计](16-fault.md) · [课程首页](../README.md) · [只多算了本行的标签 →](18-self.md)
