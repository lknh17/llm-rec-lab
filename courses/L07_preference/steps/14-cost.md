<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 奖励只看召回](13-reward.md) · [课程首页](../README.md) · [训练日志一切正常，上线条件全丢 →](15-fault.md)

# 第 14 步 / 共 17 步：时间花在哪

> 阶段 5 / 7：**GRPO：可验证奖励** · 场景：`grpo_excl`

GRPO 每步先生成 32 条回答（8 prompt × 4，最长 160 token，逐 token 解码），再对这 32 条做一次前向 + 反向（并行处理整条序列）。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `dpo_rule_excl` | 参照 | DPO，规则偏好对，prompt 里一半是排除型条件（SFT 没见过的句式） | `pref_method=dpo`, `pref_pairs=rule`, `pref_exclusion_ratio=0.5` | 3 |
| `grpo_excl` | 本步 | GRPO，字段奖励，prompt 里一半是排除型条件 | `pref_method=grpo`, `reward=field`, `pref_exclusion_ratio=0.5` | 3 |

## 先预测

**偏好阶段的时间里，生成占多少？**

- **A.** 不到 40%
- **B.** 40% - 70%
- **C.** 70% 以上

> 先选定一个（记下来），再展开下面的实测。dev 差异小于 6.7 个点（两条样本）、排除型合成集差异小于 5 个点（三条样本）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · Qwen2.5-7B · SFT + 偏好阶段 · 每个场景 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

同样 512 个 prompt，GRPO 和 DPO 的偏好阶段各用了多久？差在哪？

---

[← 奖励只看召回](13-reward.md) · [课程首页](../README.md) · [训练日志一切正常，上线条件全丢 →](15-fault.md)
