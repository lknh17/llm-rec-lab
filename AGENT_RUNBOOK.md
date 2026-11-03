# AGENT_RUNBOOK：给 H20 端 AI agent 的执行手册

你是在一台 **8 × NVIDIA H20** 机器上执行实验的 agent。这台机器和课程作者的 Mac 之间**没有直连**，唯一的通道是这个 GitHub 仓库：

- Mac 端把要跑的作业写进 `jobs/pending/`，推上来；
- 你拉下来，照本手册执行，把**脱敏后的**结果写进 `results/h20x8/`，作业单移到 `jobs/done/`，推回去；
- Mac 端拉取结果，在教学网页上展示。

你不需要理解课程内容，也**不需要、不允许修改代码**。遇到本手册没覆盖的情况，按"故障处理"一节记录到作业单里并推回，由 Mac 端决定下一步。

---

## 0. 角色与边界（必须遵守）

| 允许 | 禁止 |
| - | - |
| 修改 `results/h20x8/**`、`jobs/**` | 修改 `labkit/`、`projects/`、`courses/`、`scripts/`、`tests/` 或任何其他文件 |
| 运行本手册列出的 `python -m labkit ...` 命令 | 用 `git push --force`、`--no-verify`、改写历史 |
| 下载本手册列出的公开数据集和模型 | 上传模型权重、检查点、`.npy/.npz/.pt/.safetensors` 等二进制文件 |
| 在作业单 `notes` 里写你的观察和建议 | 把主机名、IP、用户名、内部路径、令牌写进任何会推送的文件 |

`python -m labkit publish` 会拒绝暂存区里 `results/`、`jobs/` 以外的文件，pre-push 钩子会拒绝没通过脱敏检查的推送。**这两道检查失败时，修复问题，不要绕过。**

---

## 1. 前置条件（第一次执行前确认）

1. **GPU**：`nvidia-smi` 能看到 8 张 H20（每张约 96 GB）。驱动版本要满足 PyTorch 自带 CUDA 运行时的最低要求（`setup_h20.sh` 会实测一次 bf16 matmul）。
2. **磁盘**：一块 ≥ 200 GB 的数据盘，用来放数据集、模型（约 20 GB）和批量运行的原始目录（含 LoRA 权重，不推送）。
3. **网络**：能访问 `github.com`、`pypi.org`（或国内 PyPI 镜像）、模型和数据集的下载地址（见第 3 节）。
4. **GitHub 凭证**：只授权本仓库的写权限。推荐二选一：
   - **Deploy key**（推荐）：仓库 Settings → Deploy keys → Add，勾选 *Allow write access*；本机 `git remote set-url origin git@github.com:lknh17/llm-rec-lab.git`。
   - **Fine-grained PAT**：Repository access 只选 `lknh17/llm-rec-lab`，Permissions 只给 *Contents: Read and write*。
   - 不要使用对其他仓库有权限的个人令牌；不要把令牌写进任何文件。
5. **git 身份**：`git config user.name "h20-agent"`、`git config user.email "h20-agent@users.noreply.github.com"`（不要用真实邮箱）。

---

## 2. 一次性环境准备

```bash
git clone git@github.com:lknh17/llm-rec-lab.git && cd llm-rec-lab
bash scripts/setup_h20.sh            # 装 uv、依赖，检查 torch.cuda，安装 pre-push 钩子
```

然后按脚本最后的提示设置环境变量（建议写进 `~/.bashrc`）：

```bash
export LAB_DATA_DIR=/data/llm-rec-lab-data        # 换成你的数据盘路径
export LAB_ASSETS_DIR=/data/llm-rec-lab-assets
export HF_HOME=/data/hf
export HF_ENDPOINT=https://hf-mirror.com          # huggingface.co 不通时
export LABKIT_REDACT_EXTRA=/data,<集群名>          # 不能公开的路径前缀、集群名，逗号分隔
```

`LABKIT_REDACT_EXTRA` 很重要：本机主机名、用户名、家目录、仓库路径、`LAB_DATA_DIR`、`LAB_ASSETS_DIR` 会被自动识别并替换，但**内部挂载点、集群名、机房名**这类只有你知道的字符串，必须登记在这里。

预检（必须通过才能继续）：

```bash
uv run python -m labkit probe --compute   # 输出里 compute.ok 必须是 true，device 是 cuda
```

---

## 3. 下载公开资产

```bash
uv run python -m labkit assets --profile h20x8   # 下载 h20x8 档位所有课程需要的数据集和模型
uv run python -m labkit assets                   # 查看是否齐全（每项 ok: true）
```

| 资产 | 地址 | 大小 | 许可 |
| - | - | - | - |
| `ml-1m` | https://files.grouplens.org/datasets/movielens/ml-1m.zip | 6 MB | GroupLens 研究许可，不得再分发 |
| `ml-20m` | https://files.grouplens.org/datasets/movielens/ml-20m.zip（md5 `cd245b17a1ae2cc31bb14903e1204af3`） | 190 MB | 同上 |
| `kuairand-pure` | https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz（md5 `0820331067a3784d9691136f772b35a7`） | 47 MB | CC BY-SA 4.0 |
| `qwen2.5-7b` | HuggingFace `Qwen/Qwen2.5-7B-Instruct` | 15 GB | Apache-2.0 |
| `qwen2.5-0.5b` | HuggingFace `Qwen/Qwen2.5-0.5B-Instruct` | 1 GB | Apache-2.0 |

下载失败时：

- 模型：设 `HF_ENDPOINT=https://hf-mirror.com` 重试；仍失败则 `uv pip install modelscope` 后设 `LABKIT_MODEL_SOURCE=modelscope` 重试（ModelScope 上同名仓库 `Qwen/Qwen2.5-7B-Instruct`）。
- KuaiRand：Zenodo 不通时，从作者提供的国内镜像网页 https://rec.ustc.edu.cn/share/7f5a44c0-c196-11ed-b626-81cd6a642112 手动下载 `KuaiRand-Pure.tar.gz`，放到 `$LAB_ASSETS_DIR/datasets/` 后重跑 `labkit assets kuairand-pure`（会校验 md5 并解压）。
- MovieLens：GroupLens 偶尔限速，重试即可。**不要从第三方网站下载替代文件**——md5 不一致会被拒绝。

---

## 4. 执行作业（每次的标准流程）

```bash
git pull --rebase
uv run python -m labkit jobs list                 # 看 pending 里有什么
uv run python -m labkit jobs show <job_id>        # 看作业内容：课程、档位、场景、种子、要求的代码版本
uv run python -m labkit batch <课程...> --profile h20x8 --dry-run   # 可选：先看会跑哪些、跳过哪些
uv run python -m labkit jobs run <job_id>         # 执行（见下）
```

`jobs run` 会依次：

1. 把作业单移到 `jobs/running/`，写入开始时间、实际代码版本和硬件摘要，**推送一次**（Mac 端据此知道作业开始了）；
2. 批量运行：推荐课每张卡并发 4 个、LLM 课每张卡 1 个；同一份预处理数据先跑一个，其余并行；同配置的场景只跑一次；
3. 每个运行结束后检查场景的**期望断言**（例如"截断故障的监督 token 数必须为 0"）；
4. 导出脱敏结果到 `results/h20x8/`；
5. 作业单移到 `jobs/done/`，写入每个运行的状态、耗时、断言结果；
6. 跑脱敏检查，通过后提交并推送。

**时间预估**（8 卡并行）：R1 + R2 约 1-2 小时（ML-20M 预处理每份约 5-10 分钟）；L1 + L2 每个 7B 运行约 40-60 分钟，共约 50 个运行，约 6-8 小时。长作业请用 `tmux` 或 `nohup` 运行，避免 SSH 断开时中断：

```bash
tmux new -s lab 'uv run python -m labkit jobs run <job_id> 2>&1 | tee -a $LAB_DATA_DIR/job.log'
```

**中断后续跑**：直接重新执行同一条 `jobs run <job_id>`（作业单在 `running/` 也可以）。已经成功、且代码和配置都没变的运行会自动跳过。

---

## 5. 验收清单（推送前自查，结果写进作业单）

`jobs/done/<job_id>.yaml` 里：

- [ ] `status: done`（全部成功且断言全部通过）；`partial` 表示部分失败或部分断言不通过；`failed` 表示全部失败。
- [ ] `runs` 里每一项都有 `status` 和 `minutes`；有断言的场景 `expect_ok` 应为 `true`。
- [ ] `problems` 为空；不为空时，每条问题在 `notes` 里写一句你的判断（见第 6 节）。
- [ ] `host_summary` 只有 GPU 型号、数量、显存、驱动/CUDA/torch 版本，**没有主机名和 IP**。
- [ ] `uv run python -m labkit redact --check` 输出"脱敏检查通过"。

---

## 6. 故障处理

| 现象 | 处理 |
| - | - |
| `probe --compute` 失败 / `torch.cuda.is_available()` 为 False | 检查驱动版本、`CUDA_VISIBLE_DEVICES` 是否被设成空；不要改代码。无法解决时，在 `jobs/running/<id>.yaml` 的 `notes` 写明报错（去掉主机信息），推送后停止 |
| OOM（日志含 `CUDA out of memory`） | 调度器会自动重试一次。仍失败：确认没有其他进程占用这张卡（`nvidia-smi`），用 `--max-slots` 减少并发后重跑同一作业 |
| 某个运行 `expect_ok: false`（断言不通过） | **不要重跑去"碰运气"**，也不要改断言。这是有价值的结果（例如放大后故障不再复现），保留结果，在 `notes` 里写一句观察，正常推送 |
| 运行超时（`end_reason` 含 budget） | 保留结果推送，在 `notes` 写明；Mac 端会调整时间预算后重新下发 |
| 下载失败 | 见第 3 节的备用地址；仍失败就在 `notes` 写明哪个资产、什么报错 |
| `publish` 报"脱敏检查没通过" | 看输出里的文件和问题类型。如果是 `LABKIT_REDACT_EXTRA` 漏登记的字符串，补上后执行 `uv run python -m labkit export --profile h20x8` 重新导出，再 `publish`。**不要手工编辑结果文件删内容，也不要绕过钩子** |
| `publish` 报 pull --rebase 冲突 | 只会发生在 `jobs/` 同一个文件。执行 `git rebase --abort`，`git pull`，再重跑 `uv run python -m labkit publish -m "results: <job_id>"` |
| 作业要求的代码版本（`requested_commit`）比本地新 | 先 `git pull` |
| 作业要求的代码版本比本地旧 | 正常执行，结果里会记录实际运行的版本，Mac 端据此判断 |

---

## 7. 回报格式

作业单（`jobs/done/<job_id>.yaml`）由 `jobs run` 自动填写，字段含义：

```yaml
job_id: 20261103-1400-l1-l2-h20x8
profile: h20x8
courses: [L1, L2]
requested_commit: <Mac 端下发时的代码版本>
ran_commit: <实际运行的代码版本>
started_at / finished_at: UTC 时间
hours: 6.8
host_summary: {gpu: NVIDIA H20, gpu_count: 8, gpu_mem_gb: 95.6, driver: "...", cuda: "...", torch: "..."}
status: done | partial | failed
runs:                       # 每个运行一行
  - {course: L1, scenario: baseline, seed: 0, dir: h20x8/L1/baseline/s0-xxxxxx, status: succeeded, minutes: 44.2, expect_ok: true}
problems: []                # 失败或断言不通过的运行，一行一条
notes: ""                   # 你的观察：异常、建议（例如"7B 上截断故障仍复现；epochs4 超时"）
```

`notes` 需要手工补充时：编辑 `jobs/done/<job_id>.yaml` 里的 `notes` 字段，然后 `uv run python -m labkit publish -m "jobs: <job_id> notes"`。

---

## 8. 命令速查

```bash
uv run python -m labkit probe --compute                       # 环境预检
uv run python -m labkit assets [--profile h20x8]              # 资产下载/检查
uv run python -m labkit scenarios L1 --profile h20x8          # 列出场景与运行
uv run python -m labkit jobs list | show <id> | run <id>      # 作业
uv run python -m labkit batch R1 --profile h20x8 --only baseline --seeds 0   # 手动跑一部分（调试用）
uv run python -m labkit export --profile h20x8                # 重新导出脱敏结果
uv run python -m labkit redact --check                        # 脱敏检查
uv run python -m labkit publish -m "说明"                      # 检查 + 提交 + 推送
```
