<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一条样本 →](03-trace.md)

# 第 2 步 / 共 13 步：流程结构

> 阶段 1 / 5：**理解任务**

一条样本从 CSV 走到 AUC。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 读评分日志 | ratings.csv → 100 万行 (用户, 电影, 评分, 时间) | timestamp 被引号包住还带换行，按字符串读再去空白转整数；按 (时间, 用户, 电影) 排序。 | [`recbase/data.py#L34-L64`](../../../projects/recbase/data.py#L34-L64) |
| 2. 标签与切分 | 每行 0 = 训练 / 1 = 验证 / 2 = 留出 | 标签 = 评分 ≥ 4。留出集固定为时间 ≥ t90；切分方式只决定 t90 之前怎么分。 | [`recbase/data.py#L78-L103`](../../../projects/recbase/data.py#L78-L103) |
| 3. 训练集词表编码 | 原始 ID → [0, V)，0 = 没见过 | 词表只用训练集建；验证 / 留出里新出现的用户编成 0。 | [`recbase/data.py#L120-L130`](../../../projects/recbase/data.py#L120-L130) |
| 4. Embedding + MLP | [B] × 2 个 ID → [B, 2, 32] → [B, 64] → 256 → 128 → [B] | 每个字段一张 [V, d] 的表，拼接后过 MLP，输出一个 logit。 | [`recbase/model.py#L36-L96`](../../../projects/recbase/model.py#L36-L96) |
| 5. BCE | [B] logit、[B] 标签 → 标量 | with_logits 版本，batch 内平均。 | [`recbase/model.py#L101-L103`](../../../projects/recbase/model.py#L101-L103) |
| 6. 训练循环 | 每 epoch 约 390 步 × 2048 行 | AdamW，embedding 不做 weight decay；每个 epoch 末评估验证和留出。 | [`recbase/train.py#L96-L139`](../../../projects/recbase/train.py#L96-L139) |
| 7. AUC / GAUC / 分组 | [N] 分数 + [N] 标签 + [N] 用户 → 指标 | AUC 并列取平均秩；GAUC 按用户算再按行数加权；按用户 bootstrap 估抽样误差。 | [`recbase/metrics.py#L22-L72`](../../../projects/recbase/metrics.py#L22-L72) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一条样本 →](03-trace.md)
