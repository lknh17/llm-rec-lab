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

- **A.** 不到 25%
- **B.** 25% - 60%
- **C.** 60% 以上

> 先选定一个（记下来），再展开下面的实测。dev 差异小于 6.7 个点（两条样本）、排除型合成集差异小于 5 个点（三条样本）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · Qwen2.5-7B · SFT + 偏好阶段 · 每个场景 3 种子（已有 6 / 6 个运行）</summary>

| 场景 | 类型 | 生成耗时占比 | 偏好阶段秒数 | 其中生成秒数 | SFT 秒数 | 生成耗时占比：各种子 | 断言 |
| - | - | - | - | - | - | - | - |
| `dpo_rule_excl` | 候选改动 | — | 116 ± 0 | — | 907 ± 2 | — |  |
| `grpo_excl` | 候选改动 | 40.0% ± 0.2 | 528 ± 5 | 210 ± 3 | 908 ± 6 | s0 40.2% / s1 39.8% / s2 40.0% |  |

![生成耗时占比（均值，误差线为种子间最小到最大）](figures/h20x8-p_cost-metrics.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `dpo_rule_excl` [s0](../../../results/h20x8/L07/dpo_rule_excl/s0-e37ea1/) · [s1](../../../results/h20x8/L07/dpo_rule_excl/s1-ba5bcc/) · [s2](../../../results/h20x8/L07/dpo_rule_excl/s2-019344/)
- `grpo_excl` [s0](../../../results/h20x8/L07/grpo_excl/s0-69954e/) · [s1](../../../results/h20x8/L07/grpo_excl/s1-726303/) · [s2](../../../results/h20x8/L07/grpo_excl/s2-215ff4/)

</details>

## 看完实测再想一步

同样 512 个 prompt，GRPO 和 DPO 的偏好阶段各用了多久？差在哪？

---

[← 奖励只看召回](13-reward.md) · [课程首页](../README.md) · [训练日志一切正常，上线条件全丢 →](15-fault.md)
