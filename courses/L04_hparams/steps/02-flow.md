<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一步训练 →](03-trace.md)

# 第 2 步 / 共 16 步：流程结构

> 阶段 1 / 7：**理解任务**

一步训练的完整路径。每个环节对应这一课的一个参数，点击看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. packing | 文档 + EOS 首尾相接 → [B, T + 1] 块 | doc_mask 打开时按 EOS 分段：块对角因果 mask，位置从 0 重新计。 | [`tinylm/model.py#L62-L72`](../../../projects/tinylm/model.py#L62-L72) |
| 2. 梯度累积 | [64, 512] → 4 × [16, 512] | 每份 loss ÷ 4 再反向，梯度在参数上累加；4 份做完才更新一次。 | [`tinylm/train.py#L163-L295`](../../../projects/tinylm/train.py#L163-L295) |
| 3. 16 位前向 | autocast bf16 / fp16 | 矩阵乘用 16 位，归一化和 softmax 内部用 fp32；参数和优化器状态始终是 fp32。 |  |
| 4. 交叉熵 | logits [B, T, V] → fp32 → 标量 | 对 logits 的梯度 = (softmax − onehot) / (B × T)，绝大多数分量很小。 |  |
| 5. 反向 + GradScaler | loss × scale → 反向 → 梯度 ÷ scale | 只有 fp16 需要：把小梯度抬进 fp16 的可表示范围。出现 inf 时跳过这一步并把 scale 减半。 |  |
| 6. 梯度裁剪 | 全局范数 > 1 时整体缩放 | 防止单步更新过大；梯度范数曲线是稳定性的第一信号。 |  |
| 7. 调度器 | warmup 2% → 余弦到 10% | warmup_frac = 0 时第一步就用峰值学习率。 | [`tinylm/train.py#L32-L40`](../../../projects/tinylm/train.py#L32-L40) |
| 8. AdamW | fp32 参数 + 两份 fp32 状态 | 每个参数 16 字节：权重 4 + 梯度 4 + 一阶矩 4 + 二阶矩 4。 |  |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一步训练 →](03-trace.md)
