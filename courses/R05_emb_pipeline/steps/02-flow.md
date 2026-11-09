<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪 →](03-trace.md)

# 第 2 步 / 共 17 步：流程结构

> 阶段 1 / 7：**理解任务**

一步训练的四个阶段，和三组微基准。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 原始样本 | 13 数值 + 26 类别 + 2 × 50 多值 | 类别是 40 位整数，按表行数取模。 | [`recops/embops.py#L125-L138`](../../../projects/recops/embops.py#L125-L138) |
| 2. DataLoader | 每个 __getitem__ 返回一整批 4096 | worker 进程里做特征处理，锁页内存。 |  |
| 3. 拷到 GPU | 约 2 MB / 批 | 锁页内存 + non_blocking 时可以和计算重叠。 |  |
| 4. DLRM | 28 个 [B, 64] embedding + 数值塔 → 两两内积 → MLP | 表用稀疏梯度。 | [`recops/embops.py#L191-L214`](../../../projects/recops/embops.py#L191-L214) |
| 5. 优化器 | MLP 用 Adam；表用 SparseAdam（或稠密 Adam） | 稀疏只更新本批碰到的行。 | [`recops/embops.py#L75-L107`](../../../projects/recops/embops.py#L75-L107) |
| 6. 微基准 | 查表 / 表规模 / 优化器 | CUDA event 计时、L2 冲刷、取中位数。 | [`recops/bench.py#L31-L71`](../../../projects/recops/bench.py#L31-L71) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪 →](03-trace.md)
