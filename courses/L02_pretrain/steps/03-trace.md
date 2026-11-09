<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 16 步：追踪一个 batch

> 阶段 1 / 8：**理解任务**

基线的一次真实运行：第一个 batch 的输入和目标（目标是右移一位的输入）、一层注意力里各张量的形状、训练了多少步多少 token、每 token 的 FLOPs 拆分。

来源：H20 · 小 GPT（1060 万主体参数）从零训练中文维基 · 每个场景 3 种子，`baseline` s0 的 [`trace.json`](../../../results/h20x8/L02/baseline/s0-39279f/trace.json)。

### 样本

- 文案：
- 标注：`null`

### 保存 → 重新加载核对

重新加载基础模型 + 适配器后，抽查 None 条，输出完全一致 None 条。

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
