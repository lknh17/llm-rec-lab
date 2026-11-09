# R01 行为序列建模：mean pooling / DIN / SASRec，序列长度与截断方向，全量排序与采样评估

在 MovieLens-1M 的评分序列上做"下一个物品"预测：mean pooling、DIN（target attention）、SASRec（因果自注意力）三种模型，序列长度 20 / 50 / 200，对全部 3706 部电影排序算 Recall@10，同时看"目标 + 100 个随机负例"的采样指标为什么虚高。最后诊断一次"保留最早 50 个行为"：训练 loss 更低，测试却掉了一大截。

## 怎么学（只看 GitHub 就够）

按下面“步骤”表从第 1 步往下点。每一页先给问题和选项，**先选定一个再展开“实测”**；实测里有结果表、证据图和运行目录链接。作业进度见 [作业看板](../../jobs/README.md)。参考答案和解析只在私有工作台里。

## 任务

**出发点**：用户行为越来越长，"把序列加长到 200、1000、10K 值不值"是生成式推荐绕不开的问题。在问"值不值"之前，先要知道三件事：不同的序列模型差多少；加长序列的收益落在哪类用户上；评估口径会不会把差距放大或抹平。

做法：MovieLens-1M 每个用户按时间排的评分序列（中位长度约 96），leave-one-out：倒数第 2 个做验证、最后 1 个做测试，训练只用前面的。模型都是 d = 64、2 层、2 头，输出层和物品 embedding 共享。

| 维度 | 场景 |
| - | - |
| 模型 | mean pooling、DIN、SASRec（L = 50） |
| 长度 | SASRec L = 20 / 50 / 200；mean、DIN 也跑 L = 200 |
| 截断方向 | 保留最近 50 个（基线）vs 保留最早 50 个（教学故障） |

主指标是对**全部物品**排序的 Recall@10；另报"目标 + 100 个随机负例"的采样 Recall@10 作对照。

## 规模与边界

- **H20 档**：ML-1M 全量（6040 用户、3706 部电影）；每步 128 条序列（随机终点的训练窗口），4000 步，每 500 步在 1/3 验证用户上评估、取最好的权重；bf16；每个场景 3 个种子。
- **保留了**：leave-one-out 切分、训练窗口的左补齐、全量 softmax 训练和全量排序评估、target attention 和因果自注意力的结构、截断方向。
- **缩小了**：d = 64、2 层；DIN 只用双线性打分（原版是 MLP 注意力单元），没有位置信息；物品只有 3706 个，全量打分很便宜。
- **未验证**：ML-1M 是评分日志，同一秒的评分顺序是任意的，"序列"的意义比点击日志弱；工业场景的物品数是百万到亿级，全量 softmax 要换成采样 softmax。

## 步骤

<!-- labkit:path -->
| # | 阶段 | 步骤 | 这一步回答的问题 | 场景 | H20 结果 |
| - | - | - | - | - | - |
| 1 | 理解任务 | [场景与痛点](steps/01-pain.md) | 把行为序列加长到底值不值？不同序列模型差多少、贵多少？ | — | — |
| 2 | 理解任务 | [流程结构](steps/02-flow.md) |  | — | — |
| 3 | 理解任务 | [追踪一个用户](steps/03-trace.md) |  | — | — |
| 4 | 理解任务 | [代码导读](steps/04-code.md) |  | — | — |
| 5 | 先估算 | [先估算](steps/05-estimate.md) | 验证目标是每个用户的倒数第 2 个行为。训练序列最多能用到哪？ | — | — |
| 6 | 三种模型 | [mean pooling](steps/06-mean.md) | mean pooling 和 SASRec 比，测试 Recall@10 怎样？ | `mean_l50` | 0 / 3 |
| 7 | 三种模型 | [DIN](steps/07-din.md) | DIN 和 SASRec 比，测试 Recall@10 怎样？ | `din_l50` | 0 / 3 |
| 8 | 序列长度 | [只看最近 20 个](steps/08-len20.md) | L 从 50 降到 20，测试 Recall@10 怎样？ | `sasrec_l20` | 0 / 3 |
| 9 | 序列长度 | [加长到 200](steps/09-len200.md) | L 从 50 加到 200，测试 Recall@10 怎样？ | `sasrec_l200` | 0 / 3 |
| 10 | 序列长度 | [mean pooling 加长到 200](steps/10-pool-len.md) | mean pooling 的 L 从 50 加到 200，测试 Recall@10 怎样？ | `mean_l200` | 0 / 3 |
| 11 | 序列长度 | [加长的代价](steps/11-cost.md) | L = 200 的训练样本/秒是 L = 50 的多少？ | `sasrec_l200` | 0 / 3 |
| 12 | 截断方向 | [同事改了截断](steps/12-fault.md) | 测试 Recall@10 会怎样？ | `fault_earliest` | 0 / 3 |
| 13 | 截断方向 | [诊断：loss 更低，测试为什么大跌](steps/13-diagnose.md) | 最主要的原因是什么？ | `fault_earliest` | 0 / 3 |
| 14 | 带走 | [回顾](steps/14-keycard.md) |  | — | — |
<!-- /labkit:path -->

## 场景与结果

<!-- labkit:results -->
<details><summary>展开结果（先自己预测，再看）</summary>

**h20x8**：8 × NVIDIA H20，每张卡一个运行。MovieLens-1M 评分序列，leave-one-out（倒数第 2 个验证、最后 1 个测试），对全部 3706 部电影排序；d=64、2 层、2 头，4000 步 × 128 条序列（已有 0 / 24 个运行）

_还没有结果：等待批量运行。_

</details>
<!-- /labkit:results -->

## 证据图

<!-- labkit:figures -->
<details><summary>展开证据图（先自己预测，再看）</summary>

- **h20x8**：还没有结果，图在结果推回后自动生成。

</details>
<!-- /labkit:figures -->

## 代码

<!-- labkit:code -->
| 代码段 | 位置 |
| - | - |
| `estimate` | [projects/recseq/cost.py#L18](../../projects/recseq/cost.py#L18) |
| `probe` | [projects/recseq/cost.py#L85](../../projects/recseq/cost.py#L85) |
| `truncate` | [projects/recseq/data.py#L36](../../projects/recseq/data.py#L36) |
| `windows` | [projects/recseq/data.py#L55](../../projects/recseq/data.py#L55) |
| `rab` | [projects/recseq/models.py#L20](../../projects/recseq/models.py#L20) |
| `mask` | [projects/recseq/models.py#L36](../../projects/recseq/models.py#L36) |
| `din` | [projects/recseq/models.py#L70](../../projects/recseq/models.py#L70) |
| `sasrec` | [projects/recseq/models.py#L121](../../projects/recseq/models.py#L121) |
| `hstu_attn` | [projects/recseq/models.py#L160](../../projects/recseq/models.py#L160) |
| `generate` | [projects/recseq/synth.py#L37](../../projects/recseq/synth.py#L37) |
| `evaluate` | [projects/recseq/train.py#L38](../../projects/recseq/train.py#L38) |
| `train` | [projects/recseq/train.py#L96](../../projects/recseq/train.py#L96) |
<!-- /labkit:code -->
