<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 规则偏好对，同分布 prompt](05-rule.md) · [课程首页](../README.md) · [prompt 换到模型会错的地方 →](07-excl.md)

# 第 6 步 / 共 17 步：在线采样偏好对，同分布 prompt

> 阶段 3 / 7：**DPO：偏好对从哪来** · 场景：`dpo_onpolicy`（对照 `baseline`）

换一种负例：SFT 模型对每个 prompt 用温度 1 采样 4 条，rejected = 4 条里奖励最低、且不全对的一条；**4 条全对的 prompt 不产生偏好对**。prompt 还是 512 个同分布合成样本。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 只做 SFT（与 L05 基线同配置），偏好阶段的起点和对照 | `pref_method=none` | 3 |
| `dpo_rule` | 参照 | DPO，规则构造的偏好对（删条件 / 塞无依据卖点），prompt 与 SFT 训练集同分布 | `pref_method=dpo`, `pref_pairs=rule` | 3 |
| `dpo_onpolicy` | 本步 | DPO，在线采样的偏好对（标准答案 vs SFT 自己采样出的最差回答），prompt 同分布 | `pref_method=dpo`, `pref_pairs=onpolicy` | 3 |

## 先预测

**512 个 prompt 里，大约多少比例能产出偏好对（至少有一条采样不全对）？**

- **A.** 不到 10%
- **B.** 10% - 50%
- **C.** 50% 以上

> 先选定一个（记下来），再展开下面的实测。dev 差异小于 6.7 个点（两条样本）、排除型合成集差异小于 5 个点（三条样本）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · Qwen2.5-7B · SFT + 偏好阶段 · 每个场景 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

产出率和规则偏好对（512 个，100%）比差多少？采样的算力花在了哪里？

---

[← 规则偏好对，同分布 prompt](05-rule.md) · [课程首页](../README.md) · [prompt 换到模型会错的地方 →](07-excl.md)
