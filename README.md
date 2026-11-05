# llm-rec-lab

推荐模型与 LLM 微调的循序渐进实践课。每门课都是"一个真实可跑的小项目 + 一组预先定义好的场景（基线、候选改动、教学故障、修复、多种子）+ 已经跑好的结果"。

- 只看 GitHub 也能学：按课程 README 的步骤走，代码链接直达锚点，结果表和证据图（`figures/*.svg`）默认折叠——**先想好自己的预测，再展开**。
- 所有数据和模型都是公开的（MovieLens、KuaiRand、Qwen2.5、合成数据），不含任何业务数据。
- 结果档位：`h20x8`（8 卡 H20：7B 模型、更大的数据、3 个种子，L1 的默认档位）、`mac-m4pro`（Mac 小规模（历史），可在 CPU/MPS 复现）、`mac-smoke`（Mac 冒烟，只验证流程，不进结果表）。

## 课程

| 课程 | 主题 | 项目 | 你会回答的问题 |
| - | - | - | - |
| [R1](courses/R1_rec_training_pipeline/) | 推荐训练全流程 | `projects/rec` | 一条评分如何变成张量和梯度？"AUC +4.6 点"能不能信？ |
| [R2](courses/R2_features_and_leakage/) | 特征与泄漏 | `projects/rec` | 统计特征的收益来自哪类用户？用了全量数据算统计量会怎样？ |
| [L1](courses/L1_lora_extraction/) | LoRA 结构化抽取 | `projects/adextract` | 0.5B/7B 做 LoRA 能把广告元素抽对多少？截断为什么让训练"什么都没学到"？ |
| [L2](courses/L2_error_driven_iteration/) | 错误驱动迭代 | `projects/adextract` | 从错例出发改数据/模板，改进是否超过种子噪声？最终测试能不能反复看？ |

推荐顺序：R1 → R2 → L1 → L2。R 课和 L 课互相独立。

## 仓库分层

```
courses/<课程>/          每门课的入口：README（步骤、代码表、结果表、证据图）+ scenarios.yaml（全部场景定义）+ figures/（labkit docs 生成的 SVG）
projects/rec/           推荐项目：data（加载/切分/特征）→ model（Embedding+MLP）→ train → evaluate；synth 合成数据
projects/adextract/     LLM 项目：schema/prompts/data（样本与模板）→ modeling（模型加载/LoRA）→ train → infer（生成式评估）
labkit/                 执行内核：场景展开、单作业 worker、多卡批量调度、证据摘要、SVG 图、脱敏导出、作业单、常驻 agent、发布
results/<档位>/<课程>/<场景>/s<种子>-<配置哈希>/   已跑好的结果（脱敏后），每个目录是一次运行
jobs/{pending,running,done,cancelled}/   作业单：要在 H20 上跑什么、跑得怎么样；作废的放在 cancelled/
status/h20x8.json       H20 agent 心跳（在线状态、当前作业、队列长度；只含白名单字段）
deploy/                 H20 agent 的 systemd 用户服务模板
AGENT_RUNBOOK.md        H20 端 AI agent 的执行手册
```

一个结果目录里有：

| 文件 | 内容 |
| - | - |
| `run.json` | 场景、种子、配置哈希、代码哈希、git commit、检查项（数据泄漏/截断等）、断言结果 |
| `config.json` | 完整参数（场景参数 = 档位基础参数 + 场景改动） |
| `summary.json` | 最终指标，含分组指标 |
| `trace.json` | 追踪样本：一条输入经过每一层的张量形状和数值 |
| `evidence.json` | 证据摘要（≤200KB）：可训练参数量、显存构成、吞吐、loss 曲线摘要 |
| `metrics.jsonl` | 训练曲线 |
| `predictions.jsonl` | 评估集逐样本预测（截断到前 200 条） |
| `log.txt` | 运行日志 |
| `code/` | 运行时的项目代码快照（与 `run.json` 里的代码哈希对应） |

## 本地复现

```bash
uv sync                                                      # Python 3.12；Linux 上 PyPI 的 torch 自带 CUDA 12 运行时
uv run python -m labkit assets ml-1m                         # 下载公开数据（qwen2.5-0.5b 用于 L 课）
uv run python -m labkit scenarios R1                         # 看 R1 有哪些场景
uv run python -m labkit run R1 baseline --profile mac-m4pro  # 前台跑一个场景
uv run python -m labkit batch R1 --profile mac-m4pro         # 跑完整门课（已有结果会跳过）
uv run python -m labkit run L1 baseline --profile mac-smoke  # 冒烟：只验证 L1 流程能跑通
uv run pytest -q                                             # 测试
uv run python -m labkit docs --check                         # 课程 README 结果表和证据图是否最新
```

H20 端：`uv run python -m labkit agent` 常驻领取 `jobs/pending/`（见 `AGENT_RUNBOOK.md` 和 `deploy/labkit-agent.service`）；作废作业单用 `uv run python -m labkit jobs cancel <id> --note 原因`。

自己跑出的结果在 `.work/data/batch/` 下，不会覆盖 `results/`。同一配置、同一代码得到的运行目录名相同，可以直接和 `results/` 里的对应目录对照。

## 场景、断言和可比性

- 场景只写"相对基线改了什么"，档位决定规模（数据集、模型、训练集大小、种子数）。运行目录名由场景、种子和配置哈希决定，重跑落到同一个目录。
- 教学故障带 `expect` 断言，例如截断故障要求 `supervised_tokens_total == 0`。规模放大后故障如果不再出现，断言失败，结果会标为"未通过"，不会被当作正常结果使用。
- 每个结果记录代码哈希；代码改过之后，旧结果会被标为"过期"，需要重跑。

## 数据与模型来源

| 名称 | 来源 | 许可 |
| - | - | - |
| MovieLens-1M / 20M | [GroupLens](https://grouplens.org/datasets/movielens/) | 仅限研究使用，不可再分发（本仓库只放指标，不放数据） |
| KuaiRand-Pure | [KuaiRand](https://kuairand.com/)（Zenodo 10439422） | CC BY-SA 4.0 |
| Qwen2.5-0.5B / 7B-Instruct | [Hugging Face](https://huggingface.co/Qwen) / [ModelScope](https://modelscope.cn/organization/qwen) | Apache-2.0 |
| 合成推荐数据 | `projects/rec/synth.py`，潜在因子生成，已知理论最优 AUC | 本仓库 |
| 合成广告文案 | `projects/adextract/synth.py` | 本仓库 |

## 脱敏

`results/`、`jobs/`、`status/` 和课程 `figures/` 只允许文本文件，推送前由 `scripts/pre-push` 运行 `labkit redact --check`：主机名、用户名、家目录、IP、MAC、邮箱、令牌、`LABKIT_REDACT_EXTRA` 登记的字符串、模型权重和超大文件都会被拦下。GitHub Actions 再复检一次。
