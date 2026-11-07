<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← GRPO，一半排除型 prompt](12-grpo-excl.md) · [课程首页](../README.md) · [时间花在哪 →](14-cost.md)

# 第 13 步 / 共 17 步：奖励只看召回

> 阶段 5 / 7：**GRPO：可验证奖励** · 场景：`grpo_recall_excl`（对照 `grpo_excl`）

同样的 GRPO 和 prompt，只把奖励换成"只看召回"：卖点和优惠找全了就给分，**不看条件**，也不罚多写、不罚无依据。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `grpo_excl` | 对照 | GRPO，字段奖励，prompt 里一半是排除型条件 | `pref_method=grpo`, `reward=field`, `pref_exclusion_ratio=0.5` | 3 |
| `baseline` | 参照 | 只做 SFT（与 L05 基线同配置），偏好阶段的起点和对照 | `pref_method=none` | 3 |
| `grpo_recall_excl` | 本步 | GRPO，奖励只看召回（不看条件、不罚多写），prompt 里一半是排除型条件 | `pref_method=grpo`, `reward=recall_only`, `pref_exclusion_ratio=0.5` | 3 |

## 先预测

**和字段奖励比，排除型合成集的条件保留率会怎样？**

- **A.** 差不多（±5 个点以内）
- **B.** 明显更低（< −5 个点）
- **C.** 明显更高（> +5 个点）

> 先选定一个（记下来），再展开下面的实测。dev 差异小于 6.7 个点（两条样本）、排除型合成集差异小于 5 个点（三条样本）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · Qwen2.5-7B · SFT + 偏好阶段 · 每个场景 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

条件保留率停在哪？和 SFT 比呢？无依据率有没有往上漂？

---

[← GRPO，一半排除型 prompt](12-grpo-excl.md) · [课程首页](../README.md) · [时间花在哪 →](14-cost.md)
