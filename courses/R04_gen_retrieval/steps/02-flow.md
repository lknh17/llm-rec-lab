<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一个用户 →](03-trace.md)

# 第 2 步 / 共 16 步：流程结构

> 阶段 1 / 8：**理解任务**

一个用户从历史到 Top-10。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 语义 ID | 物品 → 3 位码 + 1 位去重 → 4 个 token | R03 基线的做法，量化种子固定。 | [`genrec/sid.py#L79-L97`](../../../projects/genrec/sid.py#L79-L97) |
| 2. encoder | 最近 200 个物品 × 4 个 token → 每个物品一个位置 [B, 200, d] | 每个物品的输入 = 4 个 token 的 embedding 之和 + 位置编码。 | [`genrec/model.py#L74-L110`](../../../projects/genrec/model.py#L74-L110) |
| 3. decoder | BOS → t1 → t2 → t3 → t4，每步 [B·W, V] 的 logits | cross-attend encoder 输出；W 条 beam 共享同一份 memory。 |  |
| 4. trie 掩码 | 节点 [B, W] → 合法掩码 [B, W, V] | 稠密转移表，一次 gather。 | [`genrec/model.py#L35-L69`](../../../projects/genrec/model.py#L35-L69) |
| 5. beam search | [B, W·V] 取全局 top-W | 初始只有第 0 条 beam 分数为 0。 | [`genrec/model.py#L115-L143`](../../../projects/genrec/model.py#L115-L143) |
| 6. 反查物品 | token 元组 → 物品 → Top-10 | 没有去重位时一个元组对应一桶物品，按热度排开。 | [`genrec/train.py#L78-L103`](../../../projects/genrec/train.py#L78-L103) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一个用户 →](03-trace.md)
