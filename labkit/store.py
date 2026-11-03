"""运行记录存储：一个运行一个目录，全部是普通文件，网站之外也能直接读。

<run_dir>/
  run.json          元数据和状态（服务进程与 worker 都会改，读改写在文件锁里做）
  config.json       完整配置（resolve 后的）
  code/             本次执行的源码快照，worker 从这里 import，网页从这里展示
  env.json          运行开始时的环境探测（硬件、软件版本、设备）
  log.txt           worker 的 stdout/stderr 原文
  metrics.jsonl     按 step 和时间记录的指标
  trace.json        选定调试样本和单个 batch 的追踪（字段、张量形状、中间值）
  predictions.jsonl 评估样本的预测与错误类型
  summary.json      最终指标
  artifacts/        模型和检查点（体积大，不进回放和离线结果包的必选部分）
  heartbeat         worker 每几秒写一次时间戳
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Iterator

from . import paths

# 任务生命周期。终态之后不再变化
STATUSES = ["queued", "preparing", "running", "evaluating", "succeeded", "failed", "cancelled", "interrupted"]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
ACTIVE = {"queued", "preparing", "running", "evaluating"}

# 执行来源。网页必须持续显示，回放和离线导入不能冒充在线新结果
SOURCES = {
    "online": "在线真实运行",
    "offline": "离线执行并导入",
    "replay": "已录制真实回放",
    "batch": "批量预生成",
}


def now() -> float:
    return time.time()


def new_run_id(course_id: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{course_id}-{secrets.token_hex(2)}"


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                    if limit and len(rows) >= limit:
                        break
    except FileNotFoundError:
        pass
    return rows


class Run:
    """对一个运行目录的薄封装。"""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def meta_path(self) -> Path:
        return self.dir / "run.json"

    def meta(self) -> dict:
        return read_json(self.meta_path, {}) or {}

    @contextlib.contextmanager
    def locked(self) -> Iterator[dict]:
        """读改写 run.json。服务进程和 worker 进程都可能同时改，用 flock 串行。"""
        lock_path = self.dir / "run.lock"
        with open(lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            meta = self.meta()
            yield meta
            write_json(self.meta_path, meta)

    def update(self, **fields: Any) -> dict:
        with self.locked() as meta:
            meta.update(fields)
        return meta

    def set_status(self, status: str, note: str = "") -> dict:
        assert status in STATUSES, status
        with self.locked() as meta:
            old = meta.get("status")
            if old in TERMINAL and status != old:
                # 终态不可逆：比如已取消的任务，迟到的"运行中"心跳不能把它改回来
                return meta
            meta["status"] = status
            meta.setdefault("status_history", []).append({"status": status, "time": now(), "note": note})
            if status in ("preparing",) and not meta.get("started_at"):
                meta["started_at"] = now()
            if status in TERMINAL:
                meta["ended_at"] = now()
                if note:
                    meta["end_reason"] = note
        return meta

    def config(self) -> dict:
        return read_json(self.dir / "config.json", {}) or {}

    def summary(self) -> dict:
        return read_json(self.dir / "summary.json", {}) or {}

    def env(self) -> dict:
        return read_json(self.dir / "env.json", {}) or {}

    def trace(self) -> dict:
        return read_json(self.dir / "trace.json", {}) or {}

    def metrics(self) -> list[dict]:
        return read_jsonl(self.dir / "metrics.jsonl")

    def predictions(self, limit: int | None = None) -> list[dict]:
        return read_jsonl(self.dir / "predictions.jsonl", limit)

    def log_tail(self, max_bytes: int = 20000) -> str:
        path = self.dir / "log.txt"
        if not path.exists():
            return ""
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="replace")

    def heartbeat_age(self) -> float | None:
        try:
            return now() - float((self.dir / "heartbeat").read_text().strip())
        except (FileNotFoundError, ValueError):
            return None


# 额外的运行目录来源（例如教学平台把 results/ 各档位、旧回放挂进来）。每项返回 [(目录, origin)]
ROOT_PROVIDERS: list = []


def run_roots() -> list[tuple[Path, str]]:
    """运行目录的来源：本机临时运行 + 注册进来的其他来源。"""
    roots = [(paths.runs_dir(), "local")]
    for provider in ROOT_PROVIDERS:
        roots += provider()
    return roots


def _run_dirs(root: Path) -> list[Path]:
    """root 下的全部运行目录。本机运行是一层（runs/<run_id>），预生成结果是多层
    （<档位>/<课程>/<场景>/s<种子>-<哈希>），统一按"含 run.json 的目录"查找。"""
    if not root.exists():
        return []
    return sorted(p.parent for p in root.rglob("run.json") if "code" not in p.parent.parts[len(root.parts):])


_INDEX: dict = {"t": 0.0, "map": {}}


def _index(max_age: float = 2.0) -> dict[str, Path]:
    """run_id -> 目录。几百个运行时全量扫描也只要几十毫秒，缓存 2 秒避免网页轮询反复扫。"""
    if time.time() - _INDEX["t"] > max_age:
        m: dict[str, Path] = {}
        for root, _ in run_roots():
            for d in _run_dirs(root):
                rid = (read_json(d / "run.json") or {}).get("run_id") or d.name
                m.setdefault(rid, d)
        _INDEX.update(t=time.time(), map=m)
    return _INDEX["map"]


def get_run(run_id: str) -> Run | None:
    for root, _ in run_roots():
        d = root / run_id
        if (d / "run.json").exists():
            return Run(d)
    d = _index().get(run_id) or _index(max_age=0).get(run_id)
    return Run(d) if d else None


def list_runs() -> list[dict]:
    out = []
    for root, origin in run_roots():
        for d in _run_dirs(root):
            meta = read_json(d / "run.json")
            if not meta:
                continue
            meta.setdefault("run_id", d.name)
            meta["_origin"] = origin
            meta["_dir"] = str(d)
            out.append(meta)
    out.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return out
