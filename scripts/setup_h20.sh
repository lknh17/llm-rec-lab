#!/usr/bin/env bash
# H20（或任何 Linux + NVIDIA GPU）上的一次性环境准备。可重复执行。
#
#   bash scripts/setup_h20.sh
#
# 做的事：
#   1. 检查 nvidia-smi 和驱动；
#   2. 安装 uv（如果没有）；用 uv 建 .venv，装依赖（Linux 上 PyPI 的 torch 自带 CUDA 运行时）；
#   3. 确认 torch.cuda 可用、GPU 数量；
#   4. 安装 git pre-push 钩子（推送前强制跑脱敏检查）；
#   5. 打印建议的环境变量（数据目录、模型缓存、HF 镜像）。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 GPU 与驱动"
if ! command -v nvidia-smi >/dev/null; then
  echo "找不到 nvidia-smi：这台机器没有装 NVIDIA 驱动，或不在 PATH 里。停止。" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

echo "== 2/5 uv 与依赖"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
# 国内网络可以先设：export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
uv sync --group dev

echo "== 3/5 torch.cuda"
uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch.cuda 不可用：检查驱动版本是否满足 torch 的 CUDA 运行时要求"
n = torch.cuda.device_count()
print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  GPU × {n}: {torch.cuda.get_device_name(0)}")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("bf16 matmul ok:", float((x @ x).float().abs().mean()))
EOF

echo "== 4/5 pre-push 钩子"
install -m 0755 scripts/pre-push .git/hooks/pre-push
echo "已安装 .git/hooks/pre-push"

echo "== 5/5 建议的环境变量（写进 ~/.bashrc 或每次执行前 export）"
cat <<EOF
export LAB_DATA_DIR=/path/to/big/disk/llm-rec-lab-data      # 批量运行原始目录和缓存，建议 >= 200 GB
export LAB_ASSETS_DIR=/path/to/big/disk/llm-rec-lab-assets  # 数据集和模型，约 20 GB
export HF_HOME=/path/to/big/disk/hf                          # HuggingFace 缓存
# 访问 huggingface.co 慢或不通时二选一：
export HF_ENDPOINT=https://hf-mirror.com
# export LABKIT_MODEL_SOURCE=modelscope   # 并 uv pip install modelscope
# 内部挂载前缀、集群名等不能公开的字符串（逗号分隔），脱敏时一并替换和检查：
export LABKIT_REDACT_EXTRA=/mnt/xxx,cluster-name
EOF
echo "环境准备完成。下一步：uv run python -m labkit probe --compute"
