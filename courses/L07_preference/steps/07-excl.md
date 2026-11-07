<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 在线采样偏好对，同分布 prompt](06-onpolicy.md) · [课程首页](../README.md) · [在线采样 + 排除型 prompt →](08-onpolicy-excl.md)

# 第 7 步 / 共 17 步：prompt 换到模型会错的地方

> 阶段 3 / 7：**DPO：偏好对从哪来** · 场景：`dpo_rule_excl`（对照 `baseline`）

还是规则偏好对，但偏好阶段的 512 个 prompt 里**一半换成排除型条件**（"满 199 减 30，活动款除外"）。SFT 训练集里没有这种句式。这些 prompt 的 rejected 就是"把排除条件删掉"——正好是 SFT 模型在这类句式上犯的错。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 只做 SFT（与 L05 基线同配置），偏好阶段的起点和对照 | `pref_method=none` | 3 |
| `dpo_rule` | 参照 | DPO，规则构造的偏好对（删条件 / 塞无依据卖点），prompt 与 SFT 训练集同分布 | `pref_method=dpo`, `pref_pairs=rule` | 3 |
| `dpo_rule_excl` | 本步 | DPO，规则偏好对，prompt 里一半是排除型条件（SFT 没见过的句式） | `pref_method=dpo`, `pref_pairs=rule`, `pref_exclusion_ratio=0.5` | 3 |

## 先预测

**排除型合成集（60 条）的条件保留率，和只做 SFT 比会怎样？**

- **A.** 基本不变（±5 个点以内）
- **B.** 提升 5 - 20 个点
- **C.** 提升 20 个点以上
- **D.** 下降超过 5 个点

> 先选定一个（记下来），再展开下面的实测。dev 差异小于 6.7 个点（两条样本）、排除型合成集差异小于 5 个点（三条样本）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · Qwen2.5-7B · SFT + 偏好阶段 · 每个场景 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

目标集合涨了多少？dev 的排除型条件和合成验证集（看有没有弄坏）分别怎样？

---

[← 在线采样偏好对，同分布 prompt](06-onpolicy.md) · [课程首页](../README.md) · [在线采样 + 排除型 prompt →](08-onpolicy-excl.md)
