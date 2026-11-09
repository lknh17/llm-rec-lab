<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 15 步：追踪一层注意力

> 阶段 1 / 7：**理解任务**

各变体的真实运行：一层注意力里 q、k、v 的形状、每 token 每层缓存几个元素；MoE 场景里验证集前 16 个 token 在第 1 层选中了哪两个专家。

来源：H20 · 小 GPT（6 层 d=384）从零训练 100 MB 中文维基 · 每个场景只换一处结构 · 3 种子，`baseline` s0 的 [`trace.json`](../../../results/h20x8/L03/baseline/s0-39279f/trace.json)。

### 样本

- 文案：
- 标注：`null`

### 保存 → 重新加载核对

重新加载基础模型 + 适配器后，抽查 None 条，输出完全一致 None 条。

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
