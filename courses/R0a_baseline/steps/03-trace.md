<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 13 步：追踪一条样本

> 阶段 1 / 5：**理解任务**

基线运行的真实追踪：
- ratings.csv 开头的原文（引号里带换行的 timestamp）和解析结果；
- 训练第 1 个 batch 的第 1 条样本：原始字段、编码后的下标；
- 一条冷启动验证行：用户编码成 0，训练集里编码为 0 的行数；
- 第一步更新：用户表 V 行里有梯度的行数、被追踪那一行更新前后的值；
- 一个验证用户的 GAUC 怎么算：他的几行按分数排好，正例排在负例前面的比例。

来源：H20 · MovieLens-1M 全量 · 用户 ID + 电影 ID 的 Embedding + MLP · 3 种子，`baseline` s0 的 [`trace.json`](../../../results/h20x8/R0a/baseline/s0-461376/trace.json)。

### 样本

- 文案：
- 标注：`null`

### 第 1 个 batch 的张量

| 张量 | 形状 | dtype | 备注 |
| - | - | - | - |
| `mlp_input` | `[2048, 64]` | float32 |  |


### 保存 → 重新加载核对

重新加载基础模型 + 适配器后，抽查 None 条，输出完全一致 None 条。

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
