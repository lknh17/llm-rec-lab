<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← chosen 的概率涨了吗](09-logp.md) · [课程首页](../README.md) · [GRPO，同分布 prompt →](11-grpo.md)

# 第 10 步 / 共 17 步：β 从 0.1 降到 0.01

> 阶段 4 / 7：**DPO：参考模型与 β** · 场景：`dpo_beta_001`（对照 `dpo_rule_excl`）

同样的偏好对（规则对、一半排除型 prompt），只把 β 从 0.1 降到 0.01。β 越小，loss 对"偏离参考模型多远"越不敏感，模型可以走得更远。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `dpo_rule_excl` | 对照 | DPO，规则偏好对，prompt 里一半是排除型条件（SFT 没见过的句式） | `pref_method=dpo`, `pref_pairs=rule`, `pref_exclusion_ratio=0.5` | 3 |
| `baseline` | 参照 | 只做 SFT（与 L05 基线同配置），偏好阶段的起点和对照 | `pref_method=none` | 3 |
| `dpo_beta_001` | 本步 | 同 dpo_rule_excl，β 从 0.1 降到 0.01（几乎不约束偏离参考模型） | `pref_method=dpo`, `pref_pairs=rule`, `pref_exclusion_ratio=0.5`, `dpo_beta=0.01` | 3 |

## 先预测

**和 β = 0.1 比，dev 全对率（3 种子均值）会怎样？**

- **A.** 差不多（±6.7 个点以内）
- **B.** 明显更好（> +6.7 个点）
- **C.** 明显更差（< −6.7 个点）

> 先选定一个（记下来），再展开下面的实测。dev 差异小于 6.7 个点（两条样本）、排除型合成集差异小于 5 个点（三条样本）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · Qwen2.5-7B · SFT + 偏好阶段 · 每个场景 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

β 小 10 倍，rejected 被压低了多少倍？目标集合和 dev 是各往哪边走的？

---

[← chosen 的概率涨了吗](09-logp.md) · [课程首页](../README.md) · [GRPO，同分布 prompt →](11-grpo.md)
