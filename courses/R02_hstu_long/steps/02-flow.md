<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一个用户 →](03-trace.md)

# 第 2 步 / 共 17 步：流程结构

> 阶段 1 / 6：**理解任务**

一条合成长序列从生成到按类型评估。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 合成长序列 | 每用户 1K-10K 个 (物品, 时间) | 会话 + 簇偏好的背景；每隔约 P 步回访一次回访簇。 | [`recseq/synth.py#L38-L89`](../../../projects/recseq/synth.py#L38-L89) |
| 2. 训练窗口 | 随机终点 → 最近 L+1 个 → 左补齐 [B, L] | L = 10240 时每步 6 条序列（61440 个 token）。 | [`recseq/data.py#L56-L67`](../../../projects/recseq/data.py#L56-L67) |
| 3. 相对位置 / 时间桶 | [q, k] 的距离和 Δt → 桶号 → 每头一个偏置 | 位置 0-15 精确、之后对数；时间 ⌊log2(1+Δt)⌋。 | [`recseq/models.py#L21-L32`](../../../projects/recseq/models.py#L21-L32) |
| 4. HSTU 层 | [B, L, d] → U, V, Q, K → A [B, H, q, k] → [B, L, d] | 逐点 SiLU、除以真实长度 n；按 query 分块，每块 checkpoint。 | [`recseq/models.py#L161-L203`](../../../projects/recseq/models.py#L161-L203) |
| 5. mask | 因果 ∧ key 是真实行为 ∧ query 是真实行为 | 左补齐时 padding 在过去一侧，因果 mask 挡不住。 | [`recseq/models.py#L37-L48`](../../../projects/recseq/models.py#L37-L48) |
| 6. 按类型评估 | 长程 / 近期 / 新会话，长程再按距离分桶 | 全量排序 Recall@10。 | [`recseq/train.py#L39-L92`](../../../projects/recseq/train.py#L39-L92) |
| 7. 成本探针 | L = 512 / 2K / 8K / 10K，手写 vs 分块 / SDPA | 一条序列前向 + 反向的毫秒和峰值显存；jagged 对补齐。 | [`recseq/cost.py#L86-L123`](../../../projects/recseq/cost.py#L86-L123) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一个用户 →](03-trace.md)
