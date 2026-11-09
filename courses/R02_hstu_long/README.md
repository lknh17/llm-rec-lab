# R02 HSTU 与 10K 长序列：长度收益落在哪类请求、O(L²) 的显存与耗时、SDPA 与手写注意力、padding 与 jagged、左补齐的 mask

用一份"远处行为确实有用"的合成长序列（1K-10K，用户每隔几百到几千步回访一个只在回访时出现的簇），比较 HSTU 截断到 512 / 2K / 8K / 10K 的效果，看长度的收益只落在哪一类请求上；实测 10K 序列的显存和耗时，比较 SDPA 和手写注意力、padding 的浪费；对照 SASRec、去掉时间偏置、换回 ML-1M。最后诊断一次"左补齐只用因果 mask"。

## 怎么学（只看 GitHub 就够）

按下面“步骤”表从第 1 步往下点。每一页先给问题和选项，**先选定一个再展开“实测”**；实测里有结果表、证据图和运行目录链接。作业进度见 [作业看板](../../jobs/README.md)。参考答案和解析只在私有工作台里。

## 任务

**出发点**：生成式推荐把用户行为序列推到 10K，代价是注意力的 O(L²)。"值不值"取决于两件事：远处的行为对这一次请求有没有用；模型能不能用上它。这一课把收益和成本分开量。

**数据**：合成长序列，每个用户 1K-10K 个行为（对数均匀）。背景行为是一个个会话：会话内大多停留在同一个兴趣簇，新会话从用户的长期偏好里重抽。另外每个用户有一个"回访簇"——背景里几乎不出现，只在周期性回访时出现：每隔 P 步（P 在 256 到 9500 之间），用户先点一个"回访入口"，再看 8 个回访簇里的物品。测试请求分三类：

| 类型 | 占比 | 要预测的 | 需要多远的历史 |
| - | - | - | - |
| 长程 | 40% | 回访入口之后的第一个物品 | 至少回看到上一次回访（约 P 步） |
| 近期 | 40% | 会话中间的下一个物品 | 最近几个行为 |
| 新会话 | 20% | 隔了 1 小时以上的新会话的第一个物品 | 长期偏好 |

**模型**：HSTU（d = 64、2 层、2 头），每步 61440 个 token（L = 10240 时每步 6 条序列），3000 步。

## 规模与边界

- **H20 档**：16384 个训练用户、2048 个测试用户（另 512 个验证）、4105 个物品；HSTU 按 query 分块计算注意力（每块 1024 行、checkpoint）；bf16；每个场景 3 个种子。
- **保留了**：HSTU 的逐点 SiLU 注意力、U 门控、相对位置桶和相对时间桶；左补齐 + 因果 + padding mask；SDPA 走 flash 时的右补齐转换；O(L²) 的显存和耗时；padding 浪费。
- **缩小了**：d = 64、2 层，远小于工业 HSTU；长程信号是人为构造的"回访簇"，真实数据里远处信息的价值要实测。
- **未验证**：jagged（不补齐）的融合 kernel、长序列的推理（KV cache / M-FALCON）、多卡序列并行。

## 步骤

<!-- labkit:path -->
| # | 阶段 | 步骤 | 这一步回答的问题 | 场景 | H20 结果 |
| - | - | - | - | - | - |
| 1 | 理解任务 | [场景与痛点](steps/01-pain.md) | 行为序列从 512 推到 10K，收益落在谁身上、成本是多少？ | — | — |
| 2 | 理解任务 | [流程结构](steps/02-flow.md) |  | — | — |
| 3 | 理解任务 | [追踪一个用户](steps/03-trace.md) |  | — | — |
| 4 | 理解任务 | [代码导读](steps/04-code.md) |  | — | — |
| 5 | 先估算 | [先估算](steps/05-estimate.md) | 手写注意力，L = 10240、2 头、B = 1，一个 bf16 的 [B, H, L, L] 张量多大？ | — | — |
| 6 | 长度的收益与成本 | [截断到 512](steps/06-len512.md) | 长程题的 Recall@10，L = 512 比 L = 10240 低多少？ | `hstu_512` | 0 / 3 |
| 7 | 长度的收益与成本 | [2K 和 8K](steps/07-len-mid.md) | 长程题的 Recall@10，L = 2048 比 L = 10240 低多少？ | `hstu_2k` | 0 / 3 |
| 8 | 长度的收益与成本 | [整体收益被稀释](steps/08-benefit.md) | 只有 5% 的请求依赖远处行为时，L = 10240 比 L = 512 的整体 Recall@10 高多少？ | `baseline` | 0 / 3 |
| 9 | 长度的收益与成本 | [长度的成本](steps/09-cost-len.md) | 每个 token 的训练耗时，L = 10240 是 L = 512 的几倍？ | `hstu_512` | 0 / 3 |
| 10 | 结构与实现 | [SASRec（softmax）对 HSTU](steps/10-softmax.md) | 长程题的 Recall@10，SASRec 和 HSTU 比怎样？ | `sasrec_10k` | 0 / 3 |
| 11 | 结构与实现 | [手写注意力对 SDPA](steps/11-manual.md) | 训练峰值显存，手写是 SDPA 的几倍？ | `sasrec_10k_manual` | 0 / 3 |
| 12 | 结构与实现 | [去掉时间偏置](steps/12-timebias.md) | 新会话题的 Recall@10 会怎样？ | `hstu_notime` | 0 / 3 |
| 13 | 结构与实现 | [回到 ML-1M](steps/13-ml1m.md) | ML-1M 上 HSTU 和 SASRec 比，测试 Recall@10 怎样？ | `ml1m_hstu` | 0 / 3 |
| 14 | padding | [补齐 vs jagged](steps/14-jagged.md) | jagged 比补齐快几倍？ | `baseline` | 0 / 3 |
| 15 | padding | [同事简化了 mask](steps/15-fault.md) | 短用户（总长度 < 2048）的测试 Recall@10 会怎样？ | `fault_padmask` | 0 / 3 |
| 16 | padding | [诊断：短用户为什么掉得多](steps/16-diagnose.md) | 根因是什么？ | `fault_padmask` | 0 / 3 |
| 17 | 带走 | [回顾](steps/17-keycard.md) |  | — | — |
<!-- /labkit:path -->

## 场景与结果

<!-- labkit:results -->
<details><summary>展开结果（先自己预测，再看）</summary>

**h20x8**：8 × NVIDIA H20，每张卡一个运行。合成长序列（1K-10K，对数均匀；周期性回访串作长程信号），16384 个训练用户、2048 个测试用户；HSTU d=64、2 层、2 头，每步 61440 个 token，3000 步（已有 0 / 30 个运行）

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
