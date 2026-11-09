<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 15 步：追踪一段文本

> 阶段 1 / 6：**理解任务**

基线（16K BPE）的一次真实运行：一条广告文案和一段维基文本被切成了哪些 token，哪些常用汉字没被合并（回退成字节），训练 token 流长什么样，保存的权重重新加载后结果是否一致。

来源：H20 · 小 GPT（1060 万主体参数）从零训练 100 MB 中文维基 · 每个场景 3 种子，`baseline` s0 的 [`trace.json`](../../../results/h20x8/L01/baseline/s0-39279f/trace.json)。

### 样本

- 文案：
- 标注：`null`

### 保存 → 重新加载核对

重新加载基础模型 + 适配器后，抽查 None 条，输出完全一致 None 条。

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
