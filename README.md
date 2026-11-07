# llm-rec-lab

LLM 与推荐的循序渐进实践课。每门课 = 一个真实可跑的小项目 + 一组预先定义好的场景（基线、候选改动、教学故障、修复、多种子）+ 在 8 × H20 上跑好的结果和证据图。

课程按学习路线逐门上线，做好一门再加一门（作业进度见 [`jobs/README.md`](jobs/README.md)）：

| 课程 | 主题 | 项目 | 你会回答的问题 |
| - | - | - | - |
| [L05 SFT 与 LoRA](courses/L1_lora_extraction/)（课程编号 `L1`） | 7B 上做结构化抽取的 LoRA SFT | [`projects/adextract`](projects/adextract/) | 一条样本从 messages 到 labels mask 到 loss 怎么走？LoRA 的 rank、挂载层对参数、显存和效果各有多大影响？截断为什么让训练"什么都没学到"？ |
| [L06 围绕错误迭代](courses/L06_error_iteration/) | 错例归类 → 只改一个变量 → 多种子与逐样本判断 → 保留集检验 | [`projects/adextract`](projects/adextract/) | 5 个种子的波动有多大？针对错例补数据和泛泛加难样本哪个有效？dev loss 和任务指标同步吗？训练推理模板不一致会掉多少？ |
| [L07 偏好优化](courses/L07_preference/) | 在 L05 的 SFT 模型上做 DPO 与 GRPO | [`projects/adpref`](projects/adpref/)（复用 adextract） | 偏好对从规则构造还是在线采样？prompt 落在哪里才有信号？β 管住了什么？可验证奖励没覆盖的维度会怎样？时间花在生成还是训练？ |

- 只看 GitHub 也能学：按课程 README 的步骤走，代码链接直达锚点，结果表和证据图默认折叠，**先想好自己的预测，再展开**。
- 所有数据和模型都是公开的（Qwen2.5、合成广告文案、人工撰写的评测集），不含任何业务数据。
- 档位：`h20x8`（8 × H20，7B、4000 条训练样本、3 个种子，**课程数字以它为准**）、`mac-m4pro`（Mac 小规模（历史），0.5B、400 条）、`mac-smoke`（Mac 冒烟，只验证流程，不进结果表）。

> **H20 机器**：直接看 [`AGENT_RUNBOOK.md`](AGENT_RUNBOOK.md) 开头的"现在要做什么"。

## 仓库结构

```
courses/<课程>/               课程入口：README（步骤、代码表、结果表、证据图）、steps/（一步一页）、lesson.yaml（公开讲义）、scenarios.yaml（全部场景）、figures/（labkit docs 生成的 SVG）
projects/adextract/           项目代码：synth/schema/prompts/data（样本与模板）→ modeling（模型加载/LoRA）→ train → infer → experiment（入口）
projects/adpref/              偏好优化：pairs（偏好对与在线采样）→ reward（可验证奖励）→ optim（DPO / GRPO）→ experiment；
                              deps.txt 声明复用 adextract，依赖代码一起计入代码哈希、一起拍快照
labkit/                       执行内核：场景展开、单作业 worker、多卡批量调度、证据摘要、SVG 图、脱敏导出、作业单、常驻 agent、发布
results/<档位>/<课程>/<场景>/s<种子>-<配置哈希>/   跑好的结果（脱敏后），每个目录是一次运行
jobs/{pending,running,done,cancelled}/   作业单：Mac 端下发 → H20 领取执行 → 回报；作废的放 cancelled/
status/h20x8.json             H20 agent 心跳（在线状态、当前作业、进度；只含白名单字段）
deploy/                       H20 agent 的 systemd 用户服务模板
AGENT_RUNBOOK.md              H20 端执行手册
tests/                        测试（tests/fixtures/projects/toy 是批量调度测试用的玩具项目）
```

一个结果目录里有：

| 文件 | 内容 |
| - | - |
| `run.json` | 场景、种子、配置哈希、代码哈希、git commit、检查项（监督 token 数、截断等）、断言结果 |
| `config.json` | 完整参数（档位基础参数 + 场景改动） |
| `summary.json` | 最终指标，含分字段指标 |
| `evidence.json` | 证据摘要（≤ 200 KB）：可训练参数量、显存构成、吞吐、loss 曲线摘要 |
| `trace.json` | 追踪样本：一条输入经过每一步的 token、mask 和张量形状 |
| `metrics.jsonl` | 训练曲线 |
| `predictions.jsonl` | 评估集逐样本预测 |
| `log.txt` | 运行日志 |
| `code/` | 运行时的项目代码快照（与 `run.json` 里的代码哈希对应） |

## 作业流程（Mac ↔ GitHub ↔ H20）

1. Mac 端新建作业单写进 `jobs/pending/` 并推送；
2. H20 上的常驻 agent（`python -m labkit agent`）领取 → 校验 → 8 卡执行 → 断言 → 导出脱敏结果 → 更新课程 README 结果表和证据图 → 推回，并定期推送心跳；
3. Mac 端拉取，在教学网页上展示。

作废作业单：`uv run python -m labkit jobs cancel <id> --note 原因`，然后推送。

## 本地复现

```bash
uv sync                                                        # Python 3.12；Linux 上 PyPI 的 torch 自带 CUDA 12 运行时
uv run python -m labkit assets qwen2.5-0.5b                    # 下载小模型（Mac 档和冒烟档用）
uv run python -m labkit scenarios L1 --profile h20x8           # 看有哪些场景
uv run python -m labkit run L1 baseline --profile mac-smoke    # 冒烟：几分钟内验证流程能跑通
uv run python -m labkit batch L1 --profile mac-m4pro           # Mac 小规模跑完整门课（已有结果会跳过）
uv run pytest -q                                               # 测试
uv run python -m labkit docs --check                           # 课程 README 结果表和证据图是否最新
```

自己跑出的结果在 `.work/data/batch/` 下，不会覆盖 `results/`。同一配置、同一代码得到的运行目录名相同，可以直接和 `results/` 里的对应目录对照。

## 场景、断言和可比性

- 场景只写"相对基线改了什么"，档位决定规模（模型、训练集大小、种子数）。运行目录名由场景、种子和配置哈希决定，重跑落到同一个目录。
- 教学故障带 `expect` 断言，例如截断故障要求 `supervised_tokens_total == 0`。规模放大后故障如果不再出现，断言失败，结果标为"未通过"，不会被当作正常结果使用。
- 每个结果记录代码哈希；代码改过之后，旧结果标为"过期"，需要重跑。

## 数据与模型来源

| 名称 | 来源 | 许可 |
| - | - | - |
| Qwen2.5-0.5B / 7B-Instruct | [Hugging Face](https://huggingface.co/Qwen) / [ModelScope](https://modelscope.cn/organization/qwen) | Apache-2.0 |
| 合成广告文案（训练集、合成验证集） | `projects/adextract/synth.py` | 本仓库 |
| dev / test 评测集 | `projects/adextract/evalsets/`，人工撰写并逐条核对 | 本仓库 |

## 脱敏

`results/`、`jobs/`、`status/` 和课程 `figures/` 只允许文本文件。推送前由 `scripts/pre-push` 运行 `labkit redact --check`：主机名、用户名、家目录、IP、MAC、邮箱、令牌、`LABKIT_REDACT_EXTRA` 登记的字符串、模型权重和超大文件都会被拦下。GitHub Actions 再复检一次。
