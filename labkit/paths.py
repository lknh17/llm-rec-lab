"""目录约定。

仓库里（进 git）：
- projects/   被执行的项目代码（每次运行都会快照一份到运行目录）
- courses/    每门课的场景清单 scenarios.yaml 和 README
- jobs/       作业单：pending → running → done
- results/    脱敏后的运行结果，按 <档位>/<课程>/<场景>/s<种子>-<配置哈希> 存放

仓库外（不进 git，路径可用环境变量改）：
- LAB_DATA_DIR（默认 <repo>/.work/data）：批量运行的原始目录（含模型产物）、缓存、锁
- LAB_ASSETS_DIR（默认 <repo>/.work/assets）：数据集和本地模型
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"
COURSES_DIR = ROOT / "courses"
RESULTS_DIR = ROOT / "results"
JOBS_DIR = ROOT / "jobs"


def data_dir() -> Path:
    path = Path(os.environ.get("LAB_DATA_DIR", ROOT / ".work" / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def assets_dir() -> Path:
    return Path(os.environ.get("LAB_ASSETS_DIR", ROOT / ".work" / "assets"))


def sub(name: str) -> Path:
    path = data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir() -> Path:
    """本机临时运行（网页或命令行单次提交）。"""
    return sub("runs")


def batch_dir() -> Path:
    """批量运行的原始目录。结构和 results/ 相同，但包含模型产物和未脱敏的日志，不进 git。"""
    return sub("batch")
