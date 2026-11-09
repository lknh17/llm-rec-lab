<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 19 步：追踪

> 阶段 1 / 8：**理解任务**

基线 rank0 的真实追踪：逐阶段计时窗口前 3 步的数据 / 前向 / 反向 / 优化器毫秒数；显存快照序列和解析估算对照。

来源：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子），`baseline` s0 的 [`trace.json`](../../../results/h20x8/L08/baseline/s0-cb39a6/trace.json)。

### 样本

- 文案：
- 标注：`null`

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
