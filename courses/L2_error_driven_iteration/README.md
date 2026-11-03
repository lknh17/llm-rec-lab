# L2 围绕错误做一次迭代：针对性补数据、多种子与最终测试

从 L1 的错例里挑一类错误，只改训练数据这一个变量，用种子噪声和逐样本修好/弄坏判断是否真的修好了；顺带看 loss 和任务指标为什么不同步；诊断一次"离线 67%、上线 0%"的训练推理不一致；最后在保留的 test 集上做一次阶段性检验。

- **项目代码**：[`projects/adextract/`](../../projects/adextract/)（重点：`synth.py` 的 `exclusion_ratio`、`infer.py` 的推理模板）
- **场景清单**：[`scenarios.yaml`](scenarios.yaml)
- **前置**：[L1](../L1_lora_extraction/)

## 任务

L1 的微调基线在 dev 上的错误高度集中：

| 错误类型 | dev 样本数 | 例子 |
| - | - | - |
| 排除型条件 | 3 | 店铺活动满399减40，**特价商品除外** |
| 多优惠 | 2 | 满300减40，**再送**收纳包 |
| 否定句 | 2 | 本品**不参与**满减 |

模型把"特价商品除外"丢掉，等于告诉下游"全店满减"——业务上最危险的错误。训练集里**根本没有**这种句式。

本课只改一个变量 `exclusion_ratio`：把一定比例的训练样本**替换**（不是追加）成排除型条件样本，样本数和训练 token 基本不变。排除词表刻意避开 dev/test 里出现过的词：dev 上的提升如果存在，说明模型学到的是句式而不是背下了评估集。

## 学习路径

| 步 | 做什么 | 看哪里 |
| - | - | - |
| 1 任务 | 错误分布、为什么先修排除型条件 | 本页 |
| 2 基线与噪声 | 基线 3 个种子的全对率和标准差 | `baseline` |
| 3 看清改了哪些数据 | 替换了哪些样本、新样本长什么样 | `exclusion` 运行的 `trace.json` |
| 4 关键代码 | 合成生成器的替换逻辑、训练循环、评测规则 | 下方代码表 |
| 5 假设 | 15% 能修好几条？30% 会不会弄坏别的？多训两轮呢？ | `scenarios.yaml` |
| 6 运行与对照 | 逐样本：修好了哪几条、弄坏了哪几条，是否超过种子噪声 | `predictions.jsonl` |
| 7 最终检验 | 只在决定好之后看一次 test | `final_base` vs `final_exclusion` |
| 8 诊断 | `fault_plain_template`：同一个模型，格式合法率从 100% 掉到 0。为什么？ | `predictions.jsonl` 里的原始输出 |
| 9 记录 | 三句话结论 + 下一轮要做的事 | — |

## 关键代码

<!-- labkit:code -->
| 代码段 | 位置 |
| - | - |
| `encode` | [projects/adextract/data.py#L57](../../projects/adextract/data.py#L57) |
| `collate` | [projects/adextract/data.py#L88](../../projects/adextract/data.py#L88) |
| `generate` | [projects/adextract/infer.py#L13](../../projects/adextract/infer.py#L13) |
| `lora` | [projects/adextract/modeling.py#L55](../../projects/adextract/modeling.py#L55) |
| `messages` | [projects/adextract/prompts.py#L29](../../projects/adextract/prompts.py#L29) |
| `score_one` | [projects/adextract/schema.py#L82](../../projects/adextract/schema.py#L82) |
| `aggregate` | [projects/adextract/schema.py#L137](../../projects/adextract/schema.py#L137) |
| `make_ad` | [projects/adextract/synth.py#L85](../../projects/adextract/synth.py#L85) |
| `generate` | [projects/adextract/synth.py#L134](../../projects/adextract/synth.py#L134) |
| `loss` | [projects/adextract/train.py#L14](../../projects/adextract/train.py#L14) |
| `train_loop` | [projects/adextract/train.py#L72](../../projects/adextract/train.py#L72) |
<!-- /labkit:code -->

## 场景与结果

<!-- labkit:results -->
<details><summary>展开结果（先自己预测，再看）</summary>

**mac-m4pro**：Apple M4 Pro（MPS），Qwen2.5-0.5B-Instruct，400 条合成训练样本（已有 10 / 10 个运行）

| 场景 | 类型 | dev 全对率 | dev 格式合法率 | 合成验证全对率 | test 全对率 | 断言 |
| - | - | - | - | - | - | - |
| `baseline` | 基线 | 0.6445 ± 0.0385 (n=3) | 1.0000 ± 0.0000 (n=3) | 0.9444 ± 0.0096 (n=3) | — |  |
| `exclusion` | 候选改动 | 0.7333 ± 0.0000 (n=2) | 1.0000 ± 0.0000 (n=2) | 0.8750 ± 0.0117 (n=2) | — |  |
| `exclusion30` | 候选改动 | 0.5667 | 1.0000 | 0.8667 | — |  |
| `epochs4` | 候选改动 | 0.7000 | 1.0000 | 0.9500 | — |  |
| `fault_plain_template` | 教学故障 | 0.0000 | 0.0000 | 0.0000 | — | 通过 |
| `final_base` | 最终测试 | 0.6667 | 1.0000 | 0.9500 | 0.7000 |  |
| `final_exclusion` | 最终测试 | 0.7333 | 1.0000 | 0.8667 | 0.8000 |  |

**h20x8**：8 × NVIDIA H20，Qwen2.5-7B-Instruct（bf16），4000 条合成训练样本，每个场景 3 个种子（已有 0 / 21 个运行）

_还没有结果：等待批量运行。_

</details>
<!-- /labkit:results -->

## 放大后要回答的问题

1. 7B 零样本就会处理排除型条件吗？如果会，补数据在 7B 上还有没有必要？
2. 3 个种子下，`exclusion` 相对基线的提升是否超过 2 倍标准差？
3. 推理模板不一致的故障，在 7B 上是否同样严重？大模型对格式漂移更鲁棒吗？
4. 4 轮训练时 dev loss 和全对率是否同向？
