<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 18 步：追踪

> 阶段 1 / 7：**理解任务**

基线运行的真实追踪：第 0 条请求解码回文本的开头和结尾（聊天模板 + 内容 + 指令）、前两条请求开头相同的 token 数；前 5 条请求各自的 TTFT（含排队）、输出 token 数、分几次收到、命中缓存的 token 数。

来源：H20 · vLLM 0.11 · Qwen2.5-7B-Instruct bf16 · 输入 4096 / 输出 50 token · 3 种子（每个运行 1 张卡），`baseline` s0 的 [`trace.json`](../../../results/h20x8/L09/baseline/s0-627ae3/trace.json)。

### 样本

- 文案：
- 标注：`null`

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
