<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← GRPO，同分布 prompt](11-grpo.md) · [课程首页](../README.md) · [奖励只看召回 →](13-reward.md)

# 第 12 步 / 共 17 步：GRPO，一半排除型 prompt

> 阶段 5 / 7：**GRPO：可验证奖励** · 场景：`grpo_excl`（对照 `baseline`）

同样的 GRPO，prompt 里一半换成排除型条件。字段奖励里条件保留占 0.3：SFT 模型丢掉排除条件时，这条回答的奖励比保留条件的低。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 只做 SFT（与 L05 基线同配置），偏好阶段的起点和对照 | `pref_method=none` | 3 |
| `grpo` | 参照 | GRPO，字段奖励，prompt 同分布（每个 prompt 采样 4 条，组内归一化优势） | `pref_method=grpo`, `reward=field` | 3 |
| `dpo_rule_excl` | 参照 | DPO，规则偏好对，prompt 里一半是排除型条件（SFT 没见过的句式） | `pref_method=dpo`, `pref_pairs=rule`, `pref_exclusion_ratio=0.5` | 3 |
| `grpo_excl` | 本步 | GRPO，字段奖励，prompt 里一半是排除型条件 | `pref_method=grpo`, `reward=field`, `pref_exclusion_ratio=0.5` | 3 |

## 先预测

**排除型合成集的条件保留率，和只做 SFT 比会怎样？**

- **A.** 基本不变（±5 个点以内）
- **B.** 提升 5 - 20 个点
- **C.** 提升 20 个点以上
- **D.** 下降超过 5 个点

> 先选定一个（记下来），再展开下面的实测。dev 差异小于 6.7 个点（两条样本）、排除型合成集差异小于 5 个点（三条样本）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · Qwen2.5-7B · SFT + 偏好阶段 · 每个场景 3 种子（已有 0 / 12 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

无信号组比例和同分布 prompt 比少了多少？奖励曲线是在往上走吗？

---

[← GRPO，同分布 prompt](11-grpo.md) · [课程首页](../README.md) · [奖励只看召回 →](13-reward.md)
