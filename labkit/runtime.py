"""项目代码和平台之间唯一的接口：RunContext。

项目的 experiment.run(ctx, cfg) 通过 ctx 记录一切证据：
  ctx.stage("train")                     -> 状态变成 running，网页显示当前阶段
  ctx.metric("loss", 0.52, step=10)      -> metrics.jsonl 一行（带时间）
  ctx.trace("sample", {...})             -> trace.json 的一个条目（样本、张量形状、中间值）
  ctx.predictions(rows)                  -> predictions.jsonl
  ctx.summary({...})                     -> summary.json
  ctx.check()                            -> 检查取消和时间预算，超了就抛异常
原始证据只来自这里；AI 解释另存，不写进这些文件。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .store import Run, read_json, write_json


class Cancelled(Exception):
    pass


class BudgetExceeded(Exception):
    pass


STAGE_TO_STATUS = {"prepare": "preparing", "train": "running", "evaluate": "evaluating"}


def _jsonable(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item") and getattr(x, "ndim", 1) == 0:
        return _jsonable(x.item())
    return x


def tensor_info(t, max_items: int = 8) -> dict:
    """张量的形状、类型和前几个值。大张量只记摘要，不全量落盘。"""
    import torch

    if not isinstance(t, torch.Tensor):
        return {"value": _jsonable(t)}
    flat = t.detach().flatten()
    info = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "device": str(t.device)}
    head = flat[:max_items].float().cpu().tolist() if flat.numel() else []
    info["head"] = [round(v, 5) for v in head]
    if t.is_floating_point() and flat.numel():
        f = flat.float()
        info["mean"] = round(f.mean().item(), 5)
        info["std"] = round(f.std().item(), 5) if flat.numel() > 1 else 0.0
    return info


class RunContext:
    def __init__(self, run: Run, cfg: dict, cancel_flag: list[bool] | None = None):
        self.run = run
        self.run_dir: Path = run.dir
        self.run_id = run.id
        self.cfg = cfg
        self.t0 = time.time()
        self._cancel_flag = cancel_flag if cancel_flag is not None else [False]
        self.max_minutes = float(cfg.get("max_minutes", 0) or 0)
        self.current_stage = "prepare"
        (self.run_dir / "artifacts").mkdir(exist_ok=True)

    # ---------- 状态与日志 ----------
    def stage(self, name: str) -> None:
        self.current_stage = name
        status = STAGE_TO_STATUS.get(name, "running")
        self.run.set_status(status, f"阶段 {name}")
        self.log(f"===== 阶段：{name} =====")

    def log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {msg}", flush=True)

    def elapsed(self) -> float:
        return time.time() - self.t0

    def check(self) -> None:
        if self._cancel_flag[0]:
            raise Cancelled("收到取消请求")
        if self.max_minutes and self.elapsed() > self.max_minutes * 60:
            raise BudgetExceeded(f"超过时间预算 {self.max_minutes} 分钟")

    # ---------- 证据 ----------
    def metric(self, name: str, value: float, step: int | None = None, split: str = "train", **extra: Any) -> None:
        row = {"name": name, "value": _jsonable(float(value)), "step": step, "split": split,
               "time": round(self.elapsed(), 3), "stage": self.current_stage}
        row.update(_jsonable(extra))
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def trace(self, key: str, payload: dict) -> None:
        path = self.run_dir / "trace.json"
        data = read_json(path, {}) or {}
        data[key] = _jsonable(payload)
        write_json(path, data)

    def predictions(self, rows: list[dict], append: bool = False) -> None:
        mode = "a" if append else "w"
        with open(self.run_dir / "predictions.jsonl", mode) as f:
            for r in rows:
                f.write(json.dumps(_jsonable(r), ensure_ascii=False) + "\n")

    def summary(self, data: dict) -> None:
        path = self.run_dir / "summary.json"
        cur = read_json(path, {}) or {}
        cur.update(_jsonable(data))
        write_json(path, cur)

    def info(self, section: str, data: dict) -> None:
        """补充运行元数据：data（数据来源/版本/切分）、model、budget、checks 等。"""
        with self.run.locked() as meta:
            meta.setdefault(section, {}).update(_jsonable(data))

    def note_cache(self, stage: str, digest: str, key: dict, hit: bool, source_run: str | None) -> None:
        with self.run.locked() as meta:
            meta.setdefault("cache", []).append({
                "stage": stage, "key_hash": digest, "key": key, "hit": hit, "source_run": source_run,
            })
        self.log(f"缓存 {stage} key={digest}: {'命中，复用运行 ' + str(source_run) + ' 的产物' if hit else '未命中，重新生成'}")

    def warn(self, code: str, message: str) -> None:
        """运行中发现的异常信号（例如有效监督 token 为 0）。只记录事实，不替用户下结论。"""
        self.log(f"[WARN {code}] {message}")
        with self.run.locked() as meta:
            meta.setdefault("warnings", []).append({"code": code, "message": message, "time": round(self.elapsed(), 2)})

    def artifact(self, name: str) -> Path:
        return self.run_dir / "artifacts" / name

    def register_artifact(self, name: str, kind: str, note: str = "") -> None:
        path = self.artifact(name)
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else (path.stat().st_size if path.exists() else 0)
        with self.run.locked() as meta:
            meta.setdefault("artifacts", {})[name] = {"kind": kind, "bytes": size, "note": note}
