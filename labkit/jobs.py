"""作业单：Mac 端和 H20 端通过仓库里的 YAML 文件交换"要跑什么、跑得怎么样"。

    jobs/pending/<job_id>.yaml   Mac 端写入：要跑哪些课程、哪个档位、哪些场景/种子，要求的代码版本
    jobs/running/<job_id>.yaml   H20 端领取后移到这里，写明开始时间（推送一次，Mac 端能看到"在跑"）
    jobs/done/<job_id>.yaml      H20 端跑完后移到这里，附每个运行的状态、耗时、断言结果和硬件摘要

为什么用文件而不是 GitHub Issues：AI agent 读写文件最稳，能 diff、能审计，不需要额外的 API 权限。
状态只能往前走：pending → running → done；失败的作业也是 done（status: failed/partial），附原因。
"""

from __future__ import annotations

import datetime as dt
import shutil
import time
from pathlib import Path

import yaml

from . import paths

STATES = ("pending", "running", "done")


def _dir(state: str) -> Path:
    d = paths.JOBS_DIR / state
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_job(courses: list[str], profile: str, only: list[str] | None = None, seeds: list[int] | None = None,
            note: str = "", requested_commit: str = "", job_id: str | None = None) -> Path:
    job_id = job_id or f"{time.strftime('%Y%m%d-%H%M')}-{'-'.join(c.lower() for c in courses)}-{profile}"
    job = {
        "job_id": job_id, "created_at": _now(), "profile": profile, "courses": courses,
        "only": only or [], "seeds": seeds or [], "requested_commit": requested_commit, "note": note,
        "how_to_run": f"python -m labkit jobs run {job_id}",
    }
    path = _dir("pending") / f"{job_id}.yaml"
    if any((_dir(s) / f"{job_id}.yaml").exists() for s in STATES):
        raise FileExistsError(f"作业 {job_id} 已存在")
    path.write_text(yaml.safe_dump(job, allow_unicode=True, sort_keys=False))
    return path


def find(job_id: str) -> tuple[str, Path] | None:
    for s in STATES:
        p = _dir(s) / f"{job_id}.yaml"
        if p.exists():
            return s, p
    return None


def load(job_id: str) -> dict:
    hit = find(job_id)
    if not hit:
        raise KeyError(f"找不到作业 {job_id}")
    data = yaml.safe_load(hit[1].read_text())
    data["_state"] = hit[0]
    return data


def list_jobs() -> list[dict]:
    out = []
    for s in STATES:
        for p in sorted(_dir(s).glob("*.yaml")):
            d = yaml.safe_load(p.read_text()) or {}
            d["_state"] = s
            out.append(d)
    return out


def move(job_id: str, to: str, **updates) -> Path:
    state, path = find(job_id) or (None, None)
    if state is None:
        raise KeyError(f"找不到作业 {job_id}")
    if STATES.index(to) < STATES.index(state):
        raise ValueError(f"作业状态只能往前走：{state} → {to} 不允许")
    data = yaml.safe_load(path.read_text())
    data.update(updates)
    dst = _dir(to) / path.name
    dst.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    if dst != path:
        path.unlink()
    return dst


def host_summary() -> dict:
    """硬件摘要：只有型号、数量和版本号，不含主机名、IP、用户名。"""
    out: dict = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_count"] = torch.cuda.device_count()
            out["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        import subprocess

        r = subprocess.run([smi, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True)
        out["driver"] = (r.stdout.strip().splitlines() or [""])[0]
    return out


def run_job(job_id: str, max_slots: int | None = None, publish_each: bool = True) -> dict:
    """H20 端执行一个作业：领取 → 批量运行 → 导出脱敏结果 → 写回报。publish 由调用方决定是否推送。"""
    from . import batch, redact

    job = load(job_id)
    if job["_state"] == "done":
        raise ValueError(f"作业 {job_id} 已完成")
    commit = batch.git_commit()
    if job.get("requested_commit") and not commit["commit"].startswith(job["requested_commit"][:7]):
        print(f"[注意] 作业要求代码 {job['requested_commit'][:8]}，当前是 {commit['commit'][:8]}。"
              f"先 git pull；如果当前更新，结果会记录实际运行的版本", flush=True)
    move(job_id, "running", started_at=_now(), ran_commit=commit["commit"], host_summary=host_summary())
    if publish_each:
        from .publish import publish

        publish(f"jobs: 开始 {job_id}", paths_to_add=["jobs"])
    t0 = time.time()
    report = batch.run_batch(job["courses"], job["profile"], job.get("only") or None, job.get("seeds") or None,
                             max_slots=max_slots)
    exported = redact.export_all(job["profile"], job["courses"])
    bad = [r for r in report if r["status"] not in ("succeeded", "skipped")]
    expect_bad = [r for r in report if r.get("expect_ok") is False]
    status = "done" if not bad and not expect_bad else ("failed" if len(bad) == len(report) else "partial")
    runs = [{k: r.get(k) for k in ("course", "scenario", "seed", "dir", "status", "minutes", "expect_ok", "note",
                                   "end_reason") if r.get(k) is not None} for r in report]
    dst = move(job_id, "done", status=status, finished_at=_now(), hours=round((time.time() - t0) / 3600, 2),
               runs=runs, n_exported=len(exported),
               problems=[f"{r['course']}/{r['scenario']} s{r['seed']}: {r['status']} expect_ok={r.get('expect_ok')} "
                         f"{r.get('end_reason') or ''}".strip() for r in bad + expect_bad])
    return {"status": status, "job_file": str(dst), "runs": len(report), "problems": len(bad) + len(expect_bad)}
