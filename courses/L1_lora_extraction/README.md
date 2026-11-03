# L1 广告元素抽取微调：LoRA、assistant-only 监督与截断

先评估基础模型，再追踪一条样本从 messages 到 labels 的全过程，做一次 LoRA 微调并保存、重新加载，用同一评估集比较；最后诊断一次"训练正常结束但什么都没学到"的故障。

- **项目代码**：[`projects/adextract/`](../../projects/adextract/)（合成数据 `synth.py` → 编码 `data.py` → 模型 `modeling.py` → 训练 `train.py` → 推理 `infer.py` → 评测 `schema.py`，入口 `experiment.py`）
- **场景清单**：[`scenarios.yaml`](scenarios.yaml)
- **前置**：了解 causal LM 的 next-token loss、LoRA 的 \(y = Wx + \frac{\alpha}{r} BAx\)

## 任务

从广告文案中抽取商品品类、卖点、优惠及其适用条件，输出结构化 JSON。原文没写的信息必须为空，不能编造。

| 输入 | 期望输出 |
| - | - |
| 跑鞋轻量透气，第二件半价，指定款参加 | `{"category":"跑鞋","selling_points":["轻量","透气"],"promotions":[{"offer":"第二件半价","condition":"指定款"}]}` |

难点在条件：必须保留"指定款"，不能写成所有跑鞋都半价。

| 集合 | 条数 | 来源 |
| - | - | - |
| 训练集 | 400（Mac）/ 4000（H20） | 模板生成的教学合成数据，16 个品类、虚构品牌 |
| dev | 30 | 人工撰写并逐条核对（[`evalsets/dev.jsonl`](../../projects/adextract/evalsets/)） |
| 合成验证集 | 60 | 同一生成器、不同种子 |
| test | 30 | 人工撰写，保留到阶段性最终检验 |

指标（规则评测，见 `schema.py` 的 `score_one`）：**全对率**（所有字段都对）、格式合法率、品类准确率、卖点 F1、优惠召回、条件保留率、无依据率（编造）。dev 只有 30 条，一条样本 = 3.3 个百分点。

## 学习路径

| 步 | 做什么 | 看哪里 |
| - | - | - |
| 1 任务 | 输入、输出、评测规则 | 本页 |
| 2 基础模型 | 不训练的 Qwen2.5 零样本能做到多少？错在哪？ | `base_model` 的 `predictions.jsonl` |
| 3 微调 | 基线 LoRA：r=16、挂注意力 + MLP、2 轮 | `baseline` |
| 4 追踪样本 | 一条样本的 messages → 模板文本 → token → labels（哪些位置是 -100） | `trace.json` 的 `sample` 和 `batch` |
| 5 关键代码 | 编码与监督 mask、loss 对齐、训练循环、评测规则 | 下方代码表 |
| 6 假设 | 难例加倍、few-shot、整段监督、多训两轮、降秩：各自效果和代价？ | `scenarios.yaml` |
| 7 运行与对照 | 预先跑好的结果；逐样本看修好了哪些、弄坏了哪些 | 结果表、`predictions.jsonl` |
| 8 诊断 | `fault_truncation`：loss 正常下降、训练正常结束，但全对率回到基础模型水平。为什么？ | `run.json` → `data.tokenized`、`summary.train.lora_B_norm_after` |
| 9 记录 | 三句话结论 | — |

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

- **mac-m4pro**：Qwen2.5-0.5B-Instruct，400 条训练样本，1 个种子。
- **h20x8**：Qwen2.5-7B-Instruct（bf16），4000 条训练样本，每个场景 3 个种子。7B 和 0.5B 用同一套 tokenizer，prompt 长度完全相同，截断故障在两档上应同样出现（由期望断言 `supervised_tokens_total == 0` 自动确认）。

<!-- labkit:results -->
<details><summary>展开结果（先自己预测，再看）</summary>

**mac-m4pro**：Apple M4 Pro（MPS），Qwen2.5-0.5B-Instruct，400 条合成训练样本。课程原始数字的来源（已有 9 / 9 个运行）

| 场景 | 类型 | dev 全对率 | dev 格式合法率 | 合成验证全对率 | 断言 |
| - | - | - | - | - | - |
| `base_model` | 参照 | 0.0333 | 1.0000 | 0.0333 |  |
| `baseline` | 基线 | 0.6667 | 1.0000 | 0.9500 | 通过 |
| `hard04` | 候选改动 | 0.6000 | 1.0000 | 0.9833 |  |
| `few_shot` | 候选改动 | 0.6333 | 1.0000 | 0.9167 |  |
| `all_tokens` | 候选改动 | 0.3000 | 1.0000 | 0.7667 |  |
| `epochs4` | 候选改动 | 0.7000 | 1.0000 | 0.9500 |  |
| `lora_r4` | 候选改动 | 0.5667 | 1.0000 | 0.7667 |  |
| `fault_truncation` | 教学故障 | 0.0333 | 1.0000 | 0.0333 | 通过 |
| `fix_truncation` | 修复 | 0.6667 | 1.0000 | 0.9500 | 通过 |

**h20x8**：8 × NVIDIA H20，Qwen2.5-7B-Instruct（bf16），4000 条合成训练样本，每个场景 3 个种子（已有 0 / 25 个运行）

_还没有结果：等待批量运行。_

</details>
<!-- /labkit:results -->

## 放大后要回答的问题

1. 7B 零样本（`base_model`）比 0.5B 好多少？微调后的差距是变大还是变小？
2. few-shot 在 0.5B 上的收益，到 7B 上还在吗？它的代价（每条样本多约 110 个 token）值不值？
3. 3 个种子的标准差是多少？哪些候选改动在 1 个种子时看起来有效，3 个种子后就不显著了？
4. 截断故障在 7B 上的表现是否和 0.5B 一样？LoRA B 矩阵的范数为什么正好是 0？
