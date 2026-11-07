<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 时间花在哪](14-cost.md) · [课程首页](../README.md) · [诊断：偏好对方向 →](16-diagnose.md)

# 第 15 步 / 共 17 步：训练日志一切正常，上线条件全丢

> 阶段 6 / 7：**诊断故障** · 场景：`fault_flip`（对照 `baseline`）

同事复现 dpo_rule（规则偏好对、同分布 prompt）。训练日志看起来很健康：DPO loss 稳定下降，偏好准确率很快到 1。评估时发现问题。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 只做 SFT（与 L05 基线同配置），偏好阶段的起点和对照 | `pref_method=none` | 3 |
| `fault_flip` | 本步 | 规则偏好对的 chosen / rejected 对调（"删掉条件"成了被偏好的一方） | `pref_method=dpo`, `pref_pairs=rule`, `pref_flip=True` | 3 |

## 先预测

**合成验证集（SFT 是满分）的条件保留率会怎样？**

- **A.** 和 SFT 差不多（±10 个点以内）
- **B.** 掉 10 - 30 个点
- **C.** 掉 30 个点以上

> 先选定一个（记下来），再展开下面的实测。dev 差异小于 6.7 个点（两条样本）、排除型合成集差异小于 5 个点（三条样本）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · Qwen2.5-7B · SFT + 偏好阶段 · 每个场景 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 时间花在哪](14-cost.md) · [课程首页](../README.md) · [诊断：偏好对方向 →](16-diagnose.md)
