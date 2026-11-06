# AGENT_RUNBOOK：H20 端执行手册

你是一台 **8 × NVIDIA H20** 机器上的执行 agent。你和课程作者的 Mac 之间**没有直连**，唯一的通道是这个 GitHub 仓库。你的工作只有一件事：

> **领取 `jobs/pending/` 里的作业单 → 在 8 张卡上跑完 → 把脱敏后的结果推回来。**

你不需要理解课程内容，也**不允许修改代码**。遇到本手册没覆盖的情况，写进作业单的 `notes` 推回，由 Mac 端决定。

---

## 现在要做什么

| | |
| - | - |
| 课程 | 仓库里只有一门课：**L1**（课程地图上的 L05 SFT 与 LoRA），目录 [`courses/L1_lora_extraction/`](courses/L1_lora_extraction/) |
| 档位 | `h20x8`：Qwen2.5-7B-Instruct（bf16）+ LoRA，4000 条合成训练样本，每张卡跑 1 个运行 |
| 规模 | 12 个场景：`base_model` 1 个种子，其余 11 个场景各 3 个种子，**共 34 个运行**（`uv run python -m labkit scenarios L1 --profile h20x8` 可以列出全部） |
| 时间 | 每个训练运行约 40-60 分钟（预算上限 180 分钟），8 卡并行约 **4-5 小时** |
| 需要的资产 | 只有模型 `qwen2.5-7b`（约 15 GB）。训练数据是仓库里的合成数据和评测集，**不需要下载数据集** |
| 作业单 | `jobs/pending/` 下的那一张（文件名形如 `<日期>-l1-h20x8.yaml`） |

**第一次**：按第 1-3 节准备环境、下载模型、预检，再按第 4 节装好常驻 agent。之后什么都不用做，agent 会自动领取、执行、推回。
**agent 已经在跑**：确认 `systemctl --user status labkit-agent` 是 active，`status/h20x8.json` 的更新时间在心跳间隔以内即可（运行中 15 分钟，空闲 3 小时，见文件里的 `beat_every_s`）。

---

## 0. 边界（必须遵守）

| 允许 | 禁止 |
| - | - |
| 改动 `results/h20x8/**`、`jobs/**`、`status/**`，以及 `labkit docs` 自动生成的 `courses/*/README.md` 结果区和 `courses/*/figures/*.svg` | 修改 `labkit/`、`projects/`、`courses/*/scenarios.yaml`、`scripts/`、`tests/` 或其他任何文件 |
| 运行本手册列出的 `python -m labkit ...` 命令 | `git push --force`、`--no-verify`、改写历史 |
| 下载 Qwen2.5 公开模型 | 上传模型权重、检查点、`.npy/.npz/.pt/.safetensors` 等二进制文件 |
| 在作业单 `notes` 里写观察和建议 | 把主机名、IP、用户名、内部路径、令牌写进任何会推送的文件 |

`labkit publish` 会拒绝暂存区里上述范围以外的文件；pre-push 钩子会拒绝没通过脱敏检查的推送。**检查失败时修复问题，不要绕过。**

---

## 1. 前置条件

1. **GPU**：`nvidia-smi` 能看到 8 张 H20（每张约 96 GB）。
2. **磁盘**：一块 ≥ 200 GB 的数据盘，放模型（约 15 GB）和批量运行的原始目录（含 LoRA 权重，不推送）。
3. **网络**：能访问 `github.com`、`pypi.org`（或国内镜像）、HuggingFace（或 `hf-mirror.com` / ModelScope）。
4. **GitHub 凭证**：只授权本仓库写权限，二选一：
   - **Deploy key**（推荐）：仓库 Settings → Deploy keys → Add，勾选 *Allow write access*；`git remote set-url origin git@github.com:lknh17/llm-rec-lab.git`。
   - **Fine-grained PAT**：Repository access 只选 `lknh17/llm-rec-lab`，Permissions 只给 *Contents: Read and write*。
5. **git 身份**：`git config user.name "h20-agent"`、`git config user.email "h20-agent@users.noreply.github.com"`（不要用真实邮箱）。

## 2. 环境准备（一次性）

```bash
git clone git@github.com:lknh17/llm-rec-lab.git && cd llm-rec-lab
bash scripts/setup_h20.sh            # 装 uv 和依赖、实测一次 bf16 matmul、安装 pre-push 钩子
```

环境变量（写进 `~/.bashrc`，第 4 节的服务文件里也要填同样的值）：

```bash
export LAB_DATA_DIR=/data/llm-rec-lab-data        # 换成你的数据盘路径
export LAB_ASSETS_DIR=/data/llm-rec-lab-assets
export HF_HOME=/data/hf
export HF_ENDPOINT=https://hf-mirror.com          # huggingface.co 不通时
export LABKIT_REDACT_EXTRA=/data,<集群名>          # 不能公开的路径前缀、集群名，逗号分隔
```

主机名、用户名、家目录、仓库路径、`LAB_DATA_DIR`、`LAB_ASSETS_DIR` 会自动识别并脱敏；**内部挂载点、集群名、机房名**只有你知道，必须登记在 `LABKIT_REDACT_EXTRA`。

## 3. 下载模型并预检

```bash
uv run python -m labkit assets --profile h20x8    # 下载 qwen2.5-7b
uv run python -m labkit assets                    # 确认 qwen2.5-7b 的 ok 为 true
uv run python -m labkit probe --compute           # compute.ok 必须为 true，device 为 cuda
```

模型下载失败：设 `HF_ENDPOINT=https://hf-mirror.com` 重试；仍失败就 `uv pip install modelscope`，设 `LABKIT_MODEL_SOURCE=modelscope` 再试（ModelScope 上同名仓库 `Qwen/Qwen2.5-7B-Instruct`）。

## 4. 常驻 agent（推荐，装好后无人值守）

```bash
mkdir -p ~/.config/systemd/user && cp deploy/labkit-agent.service ~/.config/systemd/user/
# 编辑 WorkingDirectory、LAB_DATA_DIR、LAB_ASSETS_DIR、HF_HOME、LABKIT_REDACT_EXTRA
uv run python -m labkit agent --profile h20x8 --check           # 只读检查：列出待领取作业、校验、预览运行数；不领取、不执行、不推送
# 注意：--once 不是空跑，它会真的领取并执行一张作业
systemctl --user daemon-reload && systemctl --user enable --now labkit-agent
loginctl enable-linger $USER                                     # 退出 SSH 后继续运行
```

服务文件里设了 `HF_HUB_OFFLINE=1`，所以**必须先完成第 3 节的模型下载**。

agent 每 60 秒一轮（`labkit/agent.py`）：

1. `git pull --rebase`；
2. **抢占**：最早的一张 `jobs/pending/*.yaml`（档位 h20x8）移到 `jobs/running/`，写 `claimed_by`（随机编号，不是主机名），只提交这张作业单并推送；推送冲突说明别人先领了，放弃；
3. **校验**：只允许 `courses / profile / only / seeds / requested_commit / note` 字段，场景编号必须在场景清单里，`requested_commit` 必须是当前 HEAD 的祖先；不通过就移到 `jobs/done/`（`status: rejected`，附原因），不执行；
4. **执行**：批量运行 → expect 断言 → 导出脱敏结果到 `results/h20x8/` → 更新课程 README 的结果表和证据图；
5. **推回**：脱敏检查通过后提交并推送；
6. **心跳** `status/h20x8.json`：状态（idle / running / publishing / error）、作业编号、进度、GPU 型号与数量、更新时间；状态变化时立即推送，状态不变时运行中每 15 分钟、空闲每 3 小时推一次（`beat_every_s`），避免空闲时刷提交历史；
7. 日志 `$LAB_DATA_DIR/agent/agent.log`（5 MB × 5 份轮转）。

**代码更新自动生效**：agent 每次 pull 后比较 `labkit/` 的版本，变了就在两张作业之间自动重启自己（日志里有“labkit 代码已更新…重启”）。Mac 端修好代码后不需要人工重启；但 agent 版本 1（2026-11-06 之前的代码）没有这个功能，从它升级时要手动重启一次。

agent 重启后，自己领过但没跑完的作业会自动续跑（已成功的运行跳过）。Mac 端作废的作业在 `jobs/cancelled/`，agent 不会领取。

查看：`systemctl --user status labkit-agent`、`tail -f $LAB_DATA_DIR/agent/agent.log`。手动执行（下一节）前先 `systemctl --user stop labkit-agent`，两者不要同时进行。

## 5. 手动执行（agent 不可用时）

```bash
git pull --rebase
uv run python -m labkit jobs list                                  # 看 pending 里有什么
uv run python -m labkit jobs show <job_id>                         # 课程、档位、场景、种子、要求的代码版本
uv run python -m labkit batch L1 --profile h20x8 --dry-run         # 可选：看会跑哪些、跳过哪些
tmux new -s lab 'uv run python -m labkit jobs run <job_id> 2>&1 | tee -a $LAB_DATA_DIR/job.log'
```

`jobs run` 依次：作业单移到 `running/` 并推送一次 → 8 卡并行跑（同配置的场景只跑一次）→ 每个运行检查 expect 断言 → 导出脱敏结果 → 更新课程 README 和图 → 作业单移到 `done/`，写入每个运行的状态、耗时、断言结果 → 脱敏检查 → 提交并推送。

**中断后续跑**：重新执行同一条 `jobs run <job_id>`。已成功、代码和配置都没变的运行会跳过。

## 6. 验收（推送前自查）

`jobs/done/<job_id>.yaml` 里：

- [ ] `status: done`（全部成功且断言通过）；`partial` 表示部分失败或部分断言不通过；`failed` 表示全部失败；
- [ ] `runs` 共 34 项，每项都有 `status` 和 `minutes`；有断言的场景 `expect_ok` 为 `true`；
- [ ] `problems` 为空；不为空时，每条在 `notes` 里写一句你的判断；
- [ ] `host_summary` 只有 GPU 型号、数量、显存、驱动/CUDA/torch 版本，**没有主机名和 IP**；
- [ ] `uv run python -m labkit redact --check` 输出"脱敏检查通过"。

## 7. 故障处理

| 现象 | 处理 |
| - | - |
| 驱动太旧（例如 535 只支持到 CUDA 12.2，而锁定的 torch 需要 CUDA 13） | 不改依赖和 `uv.lock`。装 NVIDIA 前向兼容包（例如 `cuda-compat-13-2`），把它的 lib 目录加进 `LD_LIBRARY_PATH`（systemd 服务文件里也要加），`probe --compute` 通过即可 |
| `probe --compute` 失败 / `torch.cuda.is_available()` 为 False | 检查驱动版本、`CUDA_VISIBLE_DEVICES` 是否被设成空；不要改代码。解决不了就在作业单 `notes` 写明报错（去掉主机信息），推送后停止 |
| OOM（日志含 `CUDA out of memory`） | 调度器会自动重试一次；仍失败就用 `nvidia-smi` 确认没有别的进程占卡，用 `--max-slots` 减少并发后重跑同一作业 |
| 某个运行 `expect_ok: false` | **不要重跑碰运气，也不要改断言。**这是有价值的结果（例如放大后故障不再复现）。保留结果，在 `notes` 写一句观察，正常推送 |
| 运行超时（`end_reason` 含 budget） | 保留结果推送，在 `notes` 写明；Mac 端会调整预算后重新下发 |
| `publish` 报"脱敏检查没通过" | 看输出的文件和问题类型。漏登记的字符串补进 `LABKIT_REDACT_EXTRA`，执行 `uv run python -m labkit export --profile h20x8` 重新导出，再 `publish`。**不要手工删结果内容，不要绕过钩子** |
| `publish` 报 pull --rebase 冲突 | `git rebase --abort` → `git pull` → `uv run python -m labkit publish -m "results: <job_id>"` |
| `requested_commit` 比本地新 | 先 `git pull` |
| 作业单被 `rejected` | 看作业单里的原因（场景编号不存在、字段不合法、代码版本不是祖先），写进 `notes` 推回即可，不要自己修作业单 |

## 8. 回报格式

作业单由 `jobs run` / agent 自动填写：

```yaml
job_id: 20261106-1200-l1-h20x8
profile: h20x8
courses: [L1]
requested_commit: <Mac 端下发时的代码版本>
ran_commit: <实际运行的代码版本>
started_at / finished_at: UTC 时间
hours: 4.6
host_summary: {gpu: NVIDIA H20, gpu_count: 8, gpu_mem_gb: 95.6, driver: "...", cuda: "...", torch: "..."}
status: done | partial | failed | rejected
runs:                       # 每个运行一行，共 34 行
  - {course: L1, scenario: baseline, seed: 0, dir: h20x8/L1/baseline/s0-xxxxxx, status: succeeded, minutes: 44.2, expect_ok: true}
problems: []                # 失败或断言不通过的运行
notes: ""                   # 你的观察，例如"7B 上截断故障仍复现；lr_1e-3 loss 发散"
```

需要补 `notes`：编辑 `jobs/done/<job_id>.yaml` 的 `notes` 字段，然后 `uv run python -m labkit publish -m "jobs: <job_id> notes"`。

## 9. 命令速查

```bash
uv run python -m labkit probe --compute                    # 环境预检
uv run python -m labkit assets [--profile h20x8]           # 模型下载/检查
uv run python -m labkit scenarios L1 --profile h20x8       # 场景与运行列表
uv run python -m labkit jobs list | show <id> | run <id>   # 作业
uv run python -m labkit agent --profile h20x8              # 常驻 agent（--check 只读检查；--once 真跑一轮）
uv run python -m labkit batch L1 --profile h20x8 --only baseline --seeds 0   # 调试：只跑一部分
uv run python -m labkit export --profile h20x8             # 重新导出脱敏结果
uv run python -m labkit evidence --check                   # 证据摘要检查
uv run python -m labkit docs                               # 重新生成课程 README 结果表和图
uv run python -m labkit redact --check                     # 脱敏检查
uv run python -m labkit publish -m "说明"                   # 检查 + 提交 + 推送
```
