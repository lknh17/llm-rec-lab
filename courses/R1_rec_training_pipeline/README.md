# R1 推荐训练全流程：从评分日志到可复现的 AUC

从一条评分记录出发，走完样本构造、时间切分、特征编码、Embedding + MLP、BCE、AdamW 更新、评估、保存与重新加载；最后判断一次"提升 4.6 个点"的结果能不能信。

- **项目代码**：[`projects/rec/`](../../projects/rec/)（数据 `data.py` → 模型 `model.py` → 训练 `train.py` → 评估 `evaluate.py`，入口 `experiment.py`）
- **场景清单**：[`scenarios.yaml`](scenarios.yaml)（每个场景改了哪些参数、跑几个种子、期望断言）
- **前置**：熟悉 CTR 精排的"离散特征 → embedding → MLP → logit → BCE"结构

## 任务

给定用户和电影，预测用户会不会喜欢它，用概率给候选电影排序。MovieLens-1M 只有"用户给电影打了分"，**没有曝光日志**：任务是"已知用户在时间 t 给电影 i 打了分，这个分数是否 ≥ 4"。负样本是"看过但打了低分"，不是"曝光未点击"，离线指标不能直接换算成点击率收益。

验收指标只看验证集：**AUC**（排序）、**LogLoss**（概率质量）、**ECE**（校准）。测试集留到阶段性最终检验。

## 9 步学习路径

| 步 | 做什么 | 看哪里 |
| - | - | - |
| 1 任务 | 弄清输入、输出、标签定义、验收指标 | 本页"任务"一节 |
| 2 基线 | 基线 AUC 是多少、3 个种子之间差多少（这就是"噪声"） | 结果表 `baseline` 行 |
| 3 追踪样本 | 一条评分记录变成哪些张量、经过哪些层、第一步更新后概率怎么变 | 任一运行目录的 `trace.json` |
| 4 关键代码 | 时间切分、模型、loss、训练循环、指标 | 下方代码表 |
| 5 假设 | 先写下：4 个候选改动里哪个会涨、涨多少（相对噪声） | `scenarios.yaml` 里的 `variant` |
| 6 运行 | 预先跑好的结果 | 结果表 |
| 7 对照 | 改动是否超过种子噪声？收益落在哪些用户分组？ | `summary.json` 的 `groups_valid` |
| 8 诊断 | `fault_random_split` 的 AUC 高了 4.6 个点，为什么不可信？ | 该运行的 `run.json` → `checks.data` |
| 9 记录 | 三句话结论：改了什么、证据是什么、下一步做什么 | — |

教学网页（作者本地）会在第 5 步要求先写下假设，再揭晓结果；只看 GitHub 时，请先自己写下预测再展开结果表。

## 关键代码

<!-- labkit:code -->
| 代码段 | 位置 |
| - | - |
| `load_raw` | [projects/rec/data.py#L76](../../projects/rec/data.py#L76) |
| `load_other` | [projects/rec/data.py#L92](../../projects/rec/data.py#L92) |
| `split` | [projects/rec/data.py#L161](../../projects/rec/data.py#L161) |
| `stats` | [projects/rec/data.py#L180](../../projects/rec/data.py#L180) |
| `encode` | [projects/rec/data.py#L224](../../projects/rec/data.py#L224) |
| `leakage_checks` | [projects/rec/data.py#L309](../../projects/rec/data.py#L309) |
| `metrics` | [projects/rec/evaluate.py#L10](../../projects/rec/evaluate.py#L10) |
| `group_metrics` | [projects/rec/evaluate.py#L64](../../projects/rec/evaluate.py#L64) |
| `model` | [projects/rec/model.py#L31](../../projects/rec/model.py#L31) |
| `loss` | [projects/rec/model.py#L65](../../projects/rec/model.py#L65) |
| `optimizer` | [projects/rec/train.py#L27](../../projects/rec/train.py#L27) |
| `train_loop` | [projects/rec/train.py#L89](../../projects/rec/train.py#L89) |
| `checkpoint` | [projects/rec/train.py#L152](../../projects/rec/train.py#L152) |
<!-- /labkit:code -->

## 场景与结果

两个档位：

- **mac-m4pro**：Apple M4 Pro CPU，ML-1M。课程讲解里的数字来自这一档。
- **h20x8**：8 × NVIDIA H20。ML-1M 场景每个 5 个种子（噪声估计更准），另加三个放大场景：MovieLens-20M（2000 万条，无用户画像）、KuaiRand（真实曝光日志，标签是 is_click）、20M 行合成数据（已知真实点击概率，能算出理论最优 AUC）。

<!-- labkit:results -->
<details><summary>展开结果（先自己预测，再看）</summary>

**mac-m4pro**：Apple M4 Pro，CPU 训练（小 MLP 在 MPS 上反而慢）。课程原始数字的来源（已有 11 / 11 个运行）

| 场景 | 类型 | 验证 AUC | 验证 LogLoss | 断言 |
| - | - | - | - | - |
| `baseline` | 基线 | 0.7475 ± 0.0015 (n=3) | 0.5809 ± 0.0004 (n=3) | 通过 |
| `emb32` | 候选改动 | 0.7478 | 0.5801 |  |
| `epochs4` | 候选改动 | 0.7485 | 0.5790 |  |
| `lr3e-3` | 候选改动 | 0.7538 | 0.5746 |  |
| `dropout02` | 候选改动 | 0.7489 | 0.5810 |  |
| `fault_random_split` | 教学故障 | 0.7950 | 0.5351 | 通过 |
| `synth_baseline` | 候选改动 | 0.7177 ± 0.0022 (n=3) | 0.5885 ± 0.0008 (n=3) | 通过 |

**h20x8**：8 × NVIDIA H20。ML-1M 场景每个跑 5 个种子，另加 ML-20M、KuaiRand、合成数据三个放大场景（已有 0 / 42 个运行）

_还没有结果：等待批量运行。_

</details>
<!-- /labkit:results -->

## 放大后要回答的问题

1. ML-1M 上 5 个种子的标准差是多少？4 个候选改动里，哪些在 3 个种子下"显著"、5 个种子下就不显著了？
2. 随机切分的虚高在 ML-20M 上是变大还是变小？（提示：看验证集冷启动用户占比 `cold_user_share_valid`）
3. 合成数据上，模型 AUC 和理论上限（`checks.data.oracle_auc_valid`）差多少？这个差距说明了什么？
4. KuaiRand 的负样本是"曝光未点击"，同一个模型结构的 AUC 为什么比 MovieLens 低得多？

## 怎么读一个运行目录

```
results/<档位>/R1/<场景>/s<种子>-<配置哈希>/
  run.json        元数据：场景、配置相对基线改了什么、代码哈希、数据检查、期望断言结果
  config.json     完整配置（所有参数）
  summary.json    指标：valid.auc / logloss / ece、分组指标、参照线
  trace.json      一条样本的完整追踪：原始字段 → 张量 → logit → loss → 梯度 → 更新后
  metrics.jsonl   训练曲线
  log.txt         运行日志
  code/           这次运行实际执行的代码快照
```
