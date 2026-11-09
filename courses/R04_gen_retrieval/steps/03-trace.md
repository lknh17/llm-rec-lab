<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 16 步：追踪一个用户

> 阶段 1 / 8：**理解任务**

基线运行的真实追踪：一个测试用户最近 5 部电影和它们的语义 ID token；encoder 输入的形状；根节点允许的 token 数；beam 10 解码出的 10 个 token 元组、分数和对应的电影；目标电影是否命中。另有"保存后重新加载，Top-10 是否一致"。

来源：H20 · ML-1M · 语义 ID 3 × 64 + 去重位 · encoder-decoder（4 + 2 层）· 3 种子，`baseline` s0 的 [`trace.json`](../../../results/h20x8/R04/baseline/s0-1eaff1/trace.json)。

### 样本

- 文案：
- 标注：`null`

### 保存 → 重新加载核对

重新加载基础模型 + 适配器后，抽查 None 条，输出完全一致 None 条。

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
