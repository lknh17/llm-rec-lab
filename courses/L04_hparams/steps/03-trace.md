<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 16 步：追踪一步训练

> 阶段 1 / 7：**理解任务**

基线的一次真实运行：第一个 batch 的输入和目标、一层注意力的张量形状、训练了多少步。打开文档 mask 的场景可以对照看同一行里有几篇文档。

来源：H20 · 小 GPT（1060 万主体参数）从零训练 100 MB 中文维基 · 每个场景只改一个训练参数 · 3 种子，`baseline` s0 的 [`trace.json`](../../../results/h20x8/L04/baseline/s0-39279f/trace.json)。

### 样本

- 文案：
- 标注：`null`

### 保存 → 重新加载核对

重新加载基础模型 + 适配器后，抽查 None 条，输出完全一致 None 条。

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
