# R03 语义 ID：残差量化、表示来源与归一化、码本形状与规模、碰撞率与均匀基准、RQ-VAE，残差没减的故障

把物品向量逐层残差量化成几个离散码（语义 ID），让千万级物品可以用几百个 token 拼出来。在 ML-1M 上比较表示来源（协同 / 内容 / 随机）和归一化；在 100 万 / 1000 万合成物品上看码本形状、层数和规模对碰撞率的影响，对照"均匀撒点"的基准；RQ-VAE 的码本坍缩。最后诊断一次"残差没减"。

## 怎么学（只看 GitHub 就够）

按下面“步骤”表从第 1 步往下点。每一页先给问题和选项，**先选定一个再展开“实测”**；实测里有结果表、证据图和运行目录链接。作业进度见 [作业看板](../../jobs/README.md)。参考答案和解析只在私有工作台里。

## 任务

**出发点**：生成式召回要让模型"写出"一个物品的 ID。千万级物品如果每个 ID 一个 token，输出层就是 1000 万 × d 的矩阵，冷启动物品也没有可学的向量。语义 ID 把每个物品压成 3-4 个离散码（每个码从几百个里选），**相似的物品共享前缀**：词表只要几百到一千个 token，组合数却有上千万。

这一课量的是"码好不好"：

| 维度 | 场景 |
| - | - |
| 表示来源 | 协同（SASRec 物品向量）/ 内容（Qwen2.5-0.5B 读片名和类型）/ 随机；是否 L2 归一化 |
| 码本形状 | ML-1M 3 × 64 / 3 × 256；合成 100 万物品 4 × 256 / 4 × 64 |
| 规模 | 合成 1000 万物品 |
| 量化器 | RQ-KMeans（逐层 k-means）/ RQ-VAE（端到端） |

主要指标：**碰撞率**（和别的物品码完全相同的比例）和它的**均匀基准**（N 个物品均匀撒进 K^L 个格子时的碰撞率）、每层的码字利用率和困惑度、重建误差、层间互信息、同前缀物品的相似度和跨视角检验。

## 规模与边界

- **H20 档**：ML-1M 3706 部电影；合成物品 100 万 / 1000 万（三层高斯树：100 个顶簇 × 50 个子簇 × 20 个微簇，加噪声）；RQ-KMeans 每层 30 轮，码本在至多 100 万个样本上拟合；每个场景 3 个种子（种子决定量化的初始化）。
- **保留了**：逐层残差、k-means++ 初始化、空簇用最远点重置、去重位、碰撞率与均匀基准、利用率与困惑度。
- **缩小了**：ML-1M 只有 3706 个物品，3 层 × 64 码的容量（26 万）远大于物品数；合成数据的簇结构是人为构造的。
- **未验证**：工业语义 ID 的均衡约束（每个码分到的物品数受控）、多模态内容向量、码本随物品更新的增量维护。

## 步骤

<!-- labkit:path -->
| # | 阶段 | 步骤 | 这一步回答的问题 | 场景 | H20 结果 |
| - | - | - | - | - | - |
| 1 | 理解任务 | [场景与痛点](steps/01-pain.md) | 千万级物品怎么用几百个 token 拼出 ID？拼出来的码好不好怎么量？ | — | — |
| 2 | 理解任务 | [流程结构](steps/02-flow.md) |  | — | — |
| 3 | 理解任务 | [追踪一部电影](steps/03-trace.md) |  | — | — |
| 4 | 理解任务 | [代码导读](steps/04-code.md) |  | — | — |
| 5 | 先估算 | [先估算](steps/05-estimate.md) | 100 万个物品均匀撒进 3 层 × 256 码（1677 万个格子），一个物品和别的物品撞码的概率大约是？ | — | — |
| 6 | 表示来源 | [内容向量](steps/06-content.md) | 内容 ID 的"相邻行为同 c1 比例"是协同 ID 的几倍？ | `content` | 0 / 3 |
| 7 | 表示来源 | [随机向量](steps/07-random.md) | 随机 ID 的碰撞率和协同 ID 比怎样？ | `random` | 0 / 3 |
| 8 | 表示来源 | [不做归一化](steps/08-unnorm.md) | 第 1 层码的困惑度 / K（分布有多均匀）会怎样？ | `unnorm` | 0 / 3 |
| 9 | 码本形状与规模 | [100 万物品，3 × 256](steps/09-depth.md) | 前 3 层的实测碰撞率大约多少？ | `synth` | 0 / 3 |
| 10 | 码本形状与规模 | [同样容量，4 × 64](steps/10-k64.md) | 同容量（1677 万）时，4 × 64 的碰撞率和 3 × 256 比怎样？ | `synth_k64` | 0 / 3 |
| 11 | 码本形状与规模 | [1000 万物品](steps/11-scale.md) | 前 3 层的碰撞率上升多少？ | `synth_10m` | 0 / 3 |
| 12 | RQ-VAE | [RQ-VAE](steps/12-rqvae.md) | RQ-VAE 第 1 层实际用到的码字占多少？ | `rqvae` | 0 / 3 |
| 13 | 故障 | [同事简化了量化代码](steps/13-fault.md) | 碰撞率会上升多少？ | `fault_noresid` | 0 / 3 |
| 14 | 故障 | [诊断：利用率 100%，为什么撞得厉害](steps/14-diagnose.md) | 根因是什么？ | `fault_noresid` | 0 / 3 |
| 15 | 带走 | [回顾](steps/15-keycard.md) |  | — | — |
<!-- /labkit:path -->

## 场景与结果

<!-- labkit:results -->
<details><summary>展开结果（先自己预测，再看）</summary>

**h20x8**：8 × NVIDIA H20，每张卡一个运行。ML-1M 的 3706 部电影（协同向量来自和 R01 同一套代码训的 SASRec），以及合成的 100 万 / 1000 万物品；RQ-KMeans 逐层 30 轮（已有 0 / 30 个运行）

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
| `cost` | [projects/genrec/cost.py#L10](../../projects/genrec/cost.py#L10) |
| `content_emb` | [projects/genrec/items.py#L64](../../projects/genrec/items.py#L64) |
| `synth` | [projects/genrec/items.py#L91](../../projects/genrec/items.py#L91) |
| `trie` | [projects/genrec/model.py#L34](../../projects/genrec/model.py#L34) |
| `encdec` | [projects/genrec/model.py#L73](../../projects/genrec/model.py#L73) |
| `beam` | [projects/genrec/model.py#L114](../../projects/genrec/model.py#L114) |
| `kmeans` | [projects/genrec/sid.py#L25](../../projects/genrec/sid.py#L25) |
| `rq` | [projects/genrec/sid.py#L52](../../projects/genrec/sid.py#L52) |
| `collide` | [projects/genrec/sid.py#L78](../../projects/genrec/sid.py#L78) |
| `metrics` | [projects/genrec/sid.py#L101](../../projects/genrec/sid.py#L101) |
| `evaluate` | [projects/genrec/train.py#L77](../../projects/genrec/train.py#L77) |
| `train` | [projects/genrec/train.py#L107](../../projects/genrec/train.py#L107) |
<!-- /labkit:code -->
