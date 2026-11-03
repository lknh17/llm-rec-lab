# R2 特征改进与评估诊断：统计特征的收益和泄漏

在 R1 基线上增删一组特征，用种子噪声和分组对照判断收益从哪里来；再识别一次"改一个参数涨 2.5 个点"的标签泄漏。

- **项目代码**：[`projects/rec/`](../../projects/rec/)（和 R1 同一个项目，只改配置）
- **场景清单**：[`scenarios.yaml`](scenarios.yaml)
- **前置**：[R1](../R1_rec_training_pipeline/)

## 任务

R1 的模型只比"按电影平均喜欢率排序"高 1.5 个点，个性化很薄。R1 留下的线索：按时间切分后，验证集 **76%** 的行来自训练集里没出现过的用户，他们的 user_id embedding 只能用 OOV 那一行。

| 特征组 | 字段 | 基线里有吗 |
| - | - | - |
| ids | user_id、movie_id embedding | 有 |
| user_profile | 性别、年龄段、职业、邮编首位 | 有 |
| item_attr | 上映年代、题材（多值 mean pooling） | 有 |
| user_stats | 用户历史评分数（log2 分桶 embedding）+ 平滑喜欢率 | 没有 |
| item_stats | 电影历史评分数（分桶）+ 平滑喜欢率 | 没有 |

统计特征必须**时点正确**（point-in-time）：每条样本只能用它发生时刻**之前**的行为来算。`stats_mode: full_data` 是教学故障：用全量数据算，特征里含有样本自己的标签和未来行为。

## 9 步学习路径

| 步 | 做什么 | 看哪里 |
| - | - | - |
| 1 任务 | 为什么统计特征可能帮到冷启动用户 | 本页 |
| 2 基线与噪声 | 基线 3 个种子的标准差 | 结果表 `baseline` |
| 3 追踪统计特征 | 一条样本的"历史评分数""历史喜欢率"是怎么算出来的 | `trace.json` |
| 4 关键代码 | `history_stats`、泄漏检查、分组评估 | 下方代码表 |
| 5 假设 | user_stats、item_stats、两者都加、去掉 ID：各涨多少？收益落在哪类用户？ | `scenarios.yaml` |
| 6 运行 | 预先跑好的结果 | 结果表 |
| 7 对照 | 按"用户历史评分数"分组，收益集中在哪一档？ | `summary.json` → `groups_valid` |
| 8 诊断 | `fault_stats_leak` 涨了 2.5 个点，怎么从数据检查里一眼看出来？ | `run.json` → `checks.data.user_stats_rows_at_or_after_sample_share` |
| 9 记录 | 三句话结论 | — |

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

<!-- labkit:results -->
<details><summary>展开结果（先自己预测，再看）</summary>

**mac-m4pro**：Apple M4 Pro，CPU 训练。课程原始数字的来源（已有 10 / 10 个运行）

| 场景 | 类型 | 验证 AUC | 验证 LogLoss | 断言 |
| - | - | - | - | - |
| `baseline` | 基线 | 0.7475 ± 0.0015 (n=3) | 0.5809 ± 0.0004 (n=3) | 通过 |
| `user_stats` | 候选改动 | 0.7671 | 0.5659 |  |
| `item_stats` | 候选改动 | 0.7481 | 0.5841 |  |
| `both_stats` | 修复 | 0.7670 ± 0.0010 (n=3) | 0.5658 ± 0.0047 (n=3) | 通过 |
| `no_ids` | 候选改动 | 0.6531 | 0.6449 |  |
| `fault_stats_leak` | 教学故障 | 0.7906 | 0.5439 | 通过 |

**h20x8**：8 × NVIDIA H20。ML-1M 场景每个 5 个种子，另在 KuaiRand 和 ML-20M 上复现统计特征收益和泄漏（已有 0 / 45 个运行）

_还没有结果：等待批量运行。_

</details>
<!-- /labkit:results -->

## 放大后要回答的问题

1. KuaiRand 是真实曝光日志：统计特征（用户历史点击率）的收益比 MovieLens 大还是小？为什么？
2. 泄漏故障在 KuaiRand 上虚高多少？同一用户反复曝光同一视频时，`full_data` 口径会泄漏得更厉害吗？
3. ML-20M 没有用户画像：统计特征能不能部分替代画像？
