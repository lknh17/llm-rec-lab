<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一个 batch →](03-trace.md)

# 第 2 步 / 共 16 步：流程结构

> 阶段 1 / 8：**理解任务**

语料和分词器直接复用 L01 的缓存。和 L01 不同的只有"训练流怎么用"：取前缀、重复几遍、跑多少步。点击每个环节看对应代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. token 流 | 100 MB 文本 → [约 2700 万 token] | L01 的产物：文档首尾相接，中间插分隔符 EOS。 |  |
| 2. 取前缀 × 遍数 | 前 data_frac 的块 × epochs 遍 | 每遍重新打乱；给了算力预算时，步数 = 预算 ÷ 每步 FLOPs。 | [`tinylm/train.py#L142-L158`](../../../projects/tinylm/train.py#L142-L158) |
| 3. 切块 + 右移 | [B, T + 1] → x [B, T]，y [B, T] | y 是 x 右移一位：位置 t 的目标是 x[t+1]。 | [`tinylm/train.py#L72-L76`](../../../projects/tinylm/train.py#L72-L76) |
| 4. 前向 | [B, T] → [B, T, V] | 每 token 前向约 2N FLOPs + 注意力。 | [`tinylm/model.py#L201-L243`](../../../projects/tinylm/model.py#L201-L243) |
| 5. 交叉熵 + 反向 + AdamW | [B × T, V] → 标量 | 反向约是前向的 2 倍算力；梯度裁剪 1.0。 | [`tinylm/train.py#L163-L295`](../../../projects/tinylm/train.py#L163-L295) |
| 6. 学习率调度 | warmup 2% → 余弦降到 10% | 调度的总步数跟着 token 预算走，每个预算都完整跑完一个余弦。 | [`tinylm/train.py#L32-L40`](../../../projects/tinylm/train.py#L32-L40) |
| 7. 评估 | 验证集 8 个点 + 留出集 + 已见数据 | 已见数据 = 训练前缀开头的一段，用来量模型背下了多少。 | [`tinylm/train.py#L81-L118`](../../../projects/tinylm/train.py#L81-L118) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一个 batch →](03-trace.md)
