<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 5 组全加](12-all.md) · [课程首页](../README.md) · [换成 DCN-v2 →](14-dcn.md)

# 第 13 步 / 共 19 步：去掉 ID

> 阶段 4 / 7：**逐组加特征** · 场景：`f_no_ids`（对照 `f_all`）

在"全部特征"上只去掉用户 ID 和电影 ID 两张表，其余 4 组不变。embedding 参数少了九成以上。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `f_all` | 对照 | 5 组特征全加（时点正确），5 个种子 | `feature_groups=['ids', 'user_profile', 'item_attr', 'user_stats', 'item_stats']` | 5 |
| `f_no_ids` | 本步 | 全部特征去掉用户 ID 和电影 ID（其余 4 组不变） | `feature_groups=['user_profile', 'item_attr', 'user_stats', 'item_stats']` | 3 |

## 先预测

**去掉 ID，验证 AUC 会怎样？**

- **A.** 涨 0.3 个点以上
- **B.** 差不多（±0.3）
- **C.** 跌 0.3 个点以上

> 先选定一个（记下来），再展开下面的实测。AUC 差异小于 0.3 个百分点（GAUC 小于 0.5）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M 全量 · 在 R0a 基线上逐组加特征、换统计口径 · 3-5 种子（已有 0 / 8 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

AUC 和 GAUC 的方向一致吗？

---

[← 5 组全加](12-all.md) · [课程首页](../README.md) · [换成 DCN-v2 →](14-dcn.md)
