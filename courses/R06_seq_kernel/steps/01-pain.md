<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[课程首页](../README.md) · [流程结构 →](02-flow.md)

# 第 1 步 / 共 15 步：场景与痛点

> 阶段 1 / 6：**理解任务**

R02 里 HSTU 的 attention 在 10K 序列上占了 97% 以上的计算，PyTorch 逐步实现又要物化 L × L。三条路：

- **torch.compile**：不改代码，看编译器能不能把 L × L 融合掉；
- **手写 Triton**：照着 FlashAttention 的结构写，逐点打分反而比 softmax 简单；
- **用现成的**：HSTU 官方有 Triton kernel，但要知道它在自己的形状上快多少。

写 kernel 的风险是"快了但错了"：因果 mask 的边界只在部分位置出错，平均误差可能看不出来。这一课把四种实现在 H20 上逐项量出来，并且每个结果都和 fp32 参照对拍。

---

[课程首页](../README.md) · [流程结构 →](02-flow.md)
