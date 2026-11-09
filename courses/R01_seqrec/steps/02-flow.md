<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一个用户 →](03-trace.md)

# 第 2 步 / 共 14 步：流程结构

> 阶段 1 / 6：**理解任务**

一个用户的评分序列从 CSV 走到 Recall@10。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 读序列 | 100 万条评分 → 6040 条按时间排的序列 | 复用 recbase 的 CSV 解析；同一秒的评分按电影编号排。 | [`recseq/data.py#L37-L43`](../../../projects/recseq/data.py#L37-L43) |
| 2. leave-one-out | 训练 seq[:-2]；验证目标 seq[-2]；测试目标 seq[-1] | 评估输入是目标之前的全部历史，再截断到 L。 |  |
| 3. 训练窗口 | 随机终点 e → 截断到 L+1 → x [B, L]、y [B, L]，左补齐 | x 和 y 错开一位：每个位置预测下一个物品。 | [`recseq/data.py#L56-L67`](../../../projects/recseq/data.py#L56-L67) |
| 4. 序列模型 | [B, L] → 每个位置的用户表示 [B, L, d] → 对全部物品打分 [B, L, V] | mean / DIN / SASRec 三选一，输出层和物品 embedding 共享。 | [`recseq/models.py#L122-L156`](../../../projects/recseq/models.py#L122-L156) |
| 5. 全量 softmax | [B·L, V] 交叉熵，pad 位置不监督 | 物品只有 3706 个，全量算得起。 | [`recseq/train.py#L97-L138`](../../../projects/recseq/train.py#L97-L138) |
| 6. 全量排序 / 采样 100 | 最后一个位置的 [V] 打分 → 目标的名次 | 同时算全量 Recall@10 和"目标 + 100 个随机负例"的 Recall@10。 | [`recseq/train.py#L39-L92`](../../../projects/recseq/train.py#L39-L92) |
| 7. 成本 | 解析 FLOPs / 推理乘加 + 实测探针 | 注意力占比 L / (L + 12d)；DIN 推理 ≈ 2·L·V·d。 | [`recseq/cost.py#L19-L41`](../../../projects/recseq/cost.py#L19-L41) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一个用户 →](03-trace.md)
