"""批量运行：把一门或几门课某个档位的全部场景跑完。

    python -m labkit batch L1 --profile h20x8                 # 跑 L1 在 h20x8 档位的全部场景
    python -m labkit batch L1 --profile h20x8 --only baseline fault_truncation --seeds 0
    python -m labkit batch --all --profile mac-m4pro --dry-run

调度：
- 每个"槽位"一次跑一个运行。槽位数 = 可用 GPU 数 × jobs_per_device（GPU 档位），或 jobs_per_device（CPU/MPS 档位）；
  每个槽位是一个线程，启动 `python -m labkit.worker <run_dir>` 子进程，设置 CUDA_VISIBLE_DEVICES 和按槽位的设备锁；
- prepare_first：同一份预处理数据（data_group 相同）的第一个运行先跑完，其余再并行，避免多个进程同时重复做同一份预处理；
- 续跑：结果目录里已有"同配置 + 同代码哈希 + 成功"的运行就跳过；失败、中断或代码变了的重跑；
- OOM 自动重试一次；超过时间预算 ×1.3 + 5 分钟强制结束；
- 每个运行结束后检查场景的 expect 断言，结果写进 run.json（expect_results / expect_ok）。
"""

from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import paths, scenarios, snapshot
from .store import Run, read_json, write_json

OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "MPS backend out of memory")


def git_commit() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "labkit", "projects"))}


def detect_gpus() -> list[str]:
    env = os.environ.get("LABKIT_GPUS")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        return []


@dataclass
class Slot:
    name: str
    gpu: str | None


def make_slots(profile: dict, max_slots: int | None = None) -> list[Slot]:
    """gpus_per_job = 1：每张卡一个槽位（× jobs_per_device）；n > 1：每 n 张卡一组，一个运行独占一组（多卡训练）。"""
    sl = profile.get("slots", {})
    per = int(sl.get("jobs_per_device", 1))
    n = int(sl.get("gpus_per_job", 0))
    if n > 0:
        gpus = detect_gpus()
        if not gpus:
            raise RuntimeError("该档位需要 GPU，但 torch.cuda.device_count() = 0。先运行 python -m labkit probe --compute 排查")
        if n == 1:
            slots = [Slot(f"gpu{g}-{j}", g) for g in gpus for j in range(per)]
        else:
            if len(gpus) < n:
                raise RuntimeError(f"该档位每个运行要 {n} 张卡，本机只有 {len(gpus)} 张")
            groups = [gpus[i:i + n] for i in range(0, len(gpus) - n + 1, n)]
            slots = [Slot(f"gpu{grp[0]}-{grp[-1]}-{j}", ",".join(grp)) for grp in groups for j in range(per)]
    else:
        slots = [Slot(f"local-{j}", None) for j in range(per)]
    return slots[:max_slots] if max_slots else slots


# ------------------------------------------------------------------ 单个运行
def baseline_cfg(course_id: str, profile: str, seed: int) -> dict:
    course = scenarios.load_course(course_id)
    base = next(s for s in course["scenarios"] if s["role"] == "baseline")
    prof = course["profiles"][profile]
    merged = dict(prof.get("base", {}))
    merged.update(base.get("params", {}))
    merged.update((base.get("profile_params") or {}).get(profile, {}))
    merged["seed"] = seed
    return scenarios.param_space(course["project"]).resolve(merged)


def existing_status(spec: scenarios.RunSpec, code_hash: str) -> str | None:
    """'published'：results/ 里已有成功结果；'raw'：batch/ 里跑完还没导出；None：需要跑。"""
    for root, tag in ((paths.RESULTS_DIR, "published"), (paths.batch_dir(), "raw")):
        meta = read_json(root / spec.rel_dir / "run.json")
        if meta and meta.get("status") == "succeeded" and meta.get("code_hash") == code_hash:
            return tag
    return None


def prepare_run(spec: scenarios.RunSpec, commit: dict) -> Run:
    run_dir = paths.batch_dir() / spec.rel_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    manifest = snapshot.snapshot(spec.project, run_dir)
    write_json(run_dir / "code" / "MANIFEST.json", manifest)
    write_json(run_dir / "config.json", spec.cfg)
    base = baseline_cfg(spec.course, spec.profile, spec.seed)
    space = scenarios.param_space(spec.project)
    now = time.time()
    meta = {
        "run_id": spec.run_id, "course_id": spec.course, "project": spec.project, "entry": spec.entry,
        "profile": spec.profile, "scenario": spec.scenario, "role": spec.role, "title": spec.title, "seed": spec.seed,
        "variant": spec.scenario, "config_hash": spec.config_hash, "code_hash": manifest["code_hash"],
        "git": commit, "changed_from_baseline": {k: v for k, v in spec.cfg.items() if v != base.get(k)},
        "param_affects": {p.name: p.affects for p in space.params}, "env_id": spec.profile, "source": "batch",
        "status": "queued", "status_history": [{"status": "queued", "time": now, "note": "批量运行排队"}],
        "created_at": now, "budget": {"max_minutes": spec.cfg.get("max_minutes")}, "expect": spec.expect,
    }
    write_json(run_dir / "run.json", meta)
    return Run(run_dir)


def _log_has_oom(run: Run) -> bool:
    tail = run.log_tail(200_000)
    return any(m in tail for m in OOM_MARKERS)


def execute(run: Run, slot: Slot, stop: threading.Event) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["LABKIT_DEVICE_LOCK"] = slot.name
    if slot.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = slot.gpu
    max_min = float(run.config().get("max_minutes") or 0)
    limit = (max_min * 1.3 + 5) * 60 if max_min else None
    with open(run.dir / "log.txt", "ab") as log:
        proc = subprocess.Popen([sys.executable, "-m", "labkit.worker", str(run.dir)], cwd=paths.ROOT, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True, env=env)
        run.update(pid=proc.pid, spawned_at=time.time(), slot=slot.name)
        t0 = time.time()
        while proc.poll() is None:
            if stop.is_set() or (limit and time.time() - t0 > limit):
                why = "批量运行被中止" if stop.is_set() else f"超过硬性时间预算 {max_min} 分钟 ×1.3 + 5"
                run.update(cancel_reason=why)
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                if run.meta().get("status") not in ("succeeded", "failed", "cancelled", "interrupted"):
                    run.set_status("cancelled" if stop.is_set() else "failed", why)
                break
            time.sleep(2)
    status = run.meta().get("status")
    if status not in ("succeeded", "failed", "cancelled", "interrupted"):
        run.set_status("interrupted", "worker 退出但没有写入终态")
        status = "interrupted"
    return status


# ------------------------------------------------------------------ 断言
def find_base(spec: scenarios.RunSpec, code_hash: str) -> dict | None:
    """同档位、同种子、同代码的 baseline 运行：先找 batch/（本次刚跑的），再找 results/。"""
    course = scenarios.load_course(spec.course)
    base_id = next(s["id"] for s in course["scenarios"] if s["role"] == "baseline")
    for root in (paths.batch_dir(), paths.RESULTS_DIR):
        for d in sorted((root / spec.profile / spec.course / base_id).glob(f"s{spec.seed}-*")):
            meta = read_json(d / "run.json") or {}
            if meta.get("status") == "succeeded" and meta.get("code_hash") == code_hash:
                return {"summary": read_json(d / "summary.json", {}), "meta": meta}
    return None


def check_expect(run: Run, spec: scenarios.RunSpec, allow_missing_base: bool) -> bool | None:
    meta, summary = run.meta(), run.summary()
    base = None
    if any(scenarios.uses_base(e) for e in spec.expect):
        b = find_base(spec, meta.get("code_hash"))
        base = {"summary": b["summary"], "meta": b["meta"]} if b else None
        if base is None and allow_missing_base:
            return None  # 等 baseline 跑完再检查
    # 断言里用 base.summary.valid.auc 这样的写法引用 baseline
    results = [scenarios.evaluate(e, summary, meta, base) for e in spec.expect]
    oks = [r["ok"] for r in results]
    ok = None if any(o is None for o in oks) else all(oks)
    run.update(expect_results=results, expect_ok=ok if spec.expect else True)
    return ok


def recheck(courses: list[str], profile: str, root: Path | None = None) -> list[dict]:
    """场景清单里的断言改了之后，对已有结果（默认 results/）按当前清单重新求值，写回 run.json。
    只重算断言，不动任何测量值；没有结果或运行失败的跳过。"""
    root = root or paths.RESULTS_DIR
    out = []
    for spec in plan(courses, profile):
        d = root / spec.rel_dir
        meta = read_json(d / "run.json") or {}
        if meta.get("status") != "succeeded":
            continue
        before = meta.get("expect_ok")
        after = check_expect(Run(d), spec, allow_missing_base=False)
        out.append({"course": spec.course, "scenario": spec.scenario, "seed": spec.seed, "before": before, "after": after})
    return out


# ------------------------------------------------------------------ 主流程
def plan(courses: list[str], profile: str, only: list[str] | None = None, seeds: list[int] | None = None) -> list:
    specs = []
    for c in courses:
        course = scenarios.load_course(c)
        if profile not in course["profiles"]:
            print(f"[跳过] {c} 没有定义档位 {profile}", flush=True)
            continue
        specs += scenarios.expand(c, profile, only, seeds)
    return specs


def find_same_config(spec: scenarios.RunSpec, code_hash: str) -> Path | None:
    """同档位里同配置、同代码的成功结果（例如修复场景和基线恰好是同一个配置），找到就复用，不重跑。"""
    for root in (paths.batch_dir(), paths.RESULTS_DIR):
        for d in (root / spec.profile).glob(f"*/*/s{spec.seed}-{spec.config_hash[:6]}"):
            meta = read_json(d / "run.json") or {}
            if (meta.get("project") == spec.project and meta.get("config_hash") == spec.config_hash
                    and meta.get("code_hash") == code_hash and meta.get("status") == "succeeded"):
                return d
    return None


def copy_alias(src: Path, spec: scenarios.RunSpec) -> Run:
    """把同配置的运行复制一份到另一个课程/场景下，元数据改成新场景，并注明复用自哪里。"""
    dst = paths.batch_dir() / spec.rel_dir
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("heartbeat", "*.lock"))
    run = Run(dst)
    src_meta = read_json(src / "run.json") or {}
    run.update(run_id=spec.run_id, course_id=spec.course, scenario=spec.scenario, variant=spec.scenario, role=spec.role,
               title=spec.title, expect=spec.expect, alias_of=src_meta.get("run_id"),
               changed_from_baseline={k: v for k, v in spec.cfg.items()
                                      if v != baseline_cfg(spec.course, spec.profile, spec.seed).get(k)})
    return run


def _order(specs: list[scenarios.RunSpec]) -> list[scenarios.RunSpec]:
    """baseline 先跑（其他场景的断言要引用它），其余按课程、场景、种子排序。"""
    rank = {r: i for i, r in enumerate(("baseline", "reference", "fault", "fix", "variant", "final"))}
    return sorted(specs, key=lambda s: (rank.get(s.role, 9), s.seed, s.course, s.scenario))


def _slot_key(course: str, profile: str) -> tuple[int, int]:
    sl = scenarios.profile_info(course, profile).get("slots", {})
    return int(sl.get("gpus_per_job", 0)), int(sl.get("jobs_per_device", 1))


def run_batch(courses: list[str], profile: str, only: list[str] | None = None, seeds: list[int] | None = None,
              dry_run: bool = False, max_slots: int | None = None, force: bool = False) -> list[dict]:
    specs = plan(courses, profile, only, seeds)
    commit = git_commit()
    code_hash = {p: snapshot.tree_manifest(snapshot.project_dir(p))["code_hash"] for p in {s.project for s in specs}}
    report: list[dict] = []

    def row(spec, status, **kw) -> dict:
        return {"course": spec.course, "scenario": spec.scenario, "seed": spec.seed, "dir": str(spec.rel_dir),
                "run_id": spec.run_id, "status": status, **kw}

    todo, aliases, primary = [], [], {}
    for s in specs:
        st = None if force else existing_status(s, code_hash[s.project])
        if st:
            report.append(row(s, "skipped", minutes=0,  # 本次作业没花时间；作业单要求每个运行都有 minutes
                              note="已有结果（同配置、同代码）" if st == "published" else "已跑完，待导出"))
            continue
        same = None if force else find_same_config(s, code_hash[s.project])
        if same:
            if dry_run:
                report.append(row(s, "skipped", note=f"将复用同配置结果 {same.name}"))
                continue
            run = copy_alias(same, s)
            ok = check_expect(run, s, allow_missing_base=False)
            report.append(row(s, "succeeded", note=f"复用同配置结果 {same.relative_to(same.parents[3])}", expect_ok=ok,
                              minutes=0))
            continue
        key = (s.project, s.config_hash)
        if key in primary:
            aliases.append((s, primary[key]))
        else:
            primary[key] = s
            todo.append(s)
    todo = _order(todo)
    reused = sum(1 for r in report if r["status"] != "skipped")
    print(f"档位 {profile}：共 {len(specs)} 个运行，需要跑 {len(todo)} 个，同配置复用 {len(aliases) + reused} 个，"
          f"已有结果跳过 {len(report) - reused} 个"
          f"（代码 {commit['commit'][:8] or '无 git'}{'，有未提交改动' if commit['dirty'] else ''}）", flush=True)
    for s in todo:
        print(f"  - {s.course}/{s.scenario} s{s.seed}  {s.rel_dir}", flush=True)
    for s, p in aliases:
        print(f"  = {s.course}/{s.scenario} s{s.seed} 复用 {p.course}/{p.scenario}", flush=True)
    if dry_run:
        return report

    stop = threading.Event()
    lock = threading.Lock()
    done_count = [0]

    def on_sigint(signum, frame):  # noqa: ARG001
        print("\n收到中断，正在停止所有运行……", flush=True)
        stop.set()

    def run_phase(phase: list, slots: list[Slot]) -> None:
        q: queue.Queue = queue.Queue()
        for s in phase:
            q.put(s)

        def worker_loop(slot: Slot):
            while not stop.is_set():
                try:
                    spec = q.get_nowait()
                except queue.Empty:
                    return
                t0 = time.time()
                run = prepare_run(spec, commit)
                status = execute(run, slot, stop)
                retried = False
                if status == "failed" and _log_has_oom(run) and not stop.is_set():
                    print(f"  [{slot.name}] {spec.run_id} OOM，30 秒后重试一次", flush=True)
                    time.sleep(30)
                    run = prepare_run(spec, commit)
                    status = execute(run, slot, stop)
                    retried = True
                ok = check_expect(run, spec, allow_missing_base=True) if status == "succeeded" else None
                minutes = round((time.time() - t0) / 60, 1)
                meta = run.meta()
                r = row(spec, status, minutes=minutes, expect_ok=ok, end_reason=meta.get("end_reason"),
                        oom_retried=retried, slot=slot.name)
                with lock:
                    done_count[0] += 1
                    report.append(r)
                    mark = {True: "断言通过", False: "断言不通过！", None: "断言待定" if spec.expect else ""}[ok]
                    print(f"[{done_count[0]}/{len(todo)}] {slot.name} {spec.course}/{spec.scenario} s{spec.seed} "
                          f"{status} {minutes}min {mark} {'' if status == 'succeeded' else meta.get('end_reason', '')}",
                          flush=True)

        threads = [threading.Thread(target=worker_loop, args=(sl,), daemon=True) for sl in slots]
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(1)

    old = signal.signal(signal.SIGINT, on_sigint)
    try:
        # 槽位配置不同的课程分组依次跑；每卡并发多的（小模型）排在前面
        groups: dict[tuple, list] = {}
        for s in todo:
            groups.setdefault(_slot_key(s.course, profile), []).append(s)
        for (gpus, per), group in sorted(groups.items(), key=lambda kv: -kv[0][1]):
            if stop.is_set():
                break
            slots = make_slots({"slots": {"gpus_per_job": gpus, "jobs_per_device": per}}, max_slots)
            print(f"== {sorted({s.course for s in group})}：{len(group)} 个运行，槽位 {[sl.name for sl in slots]}", flush=True)
            first: dict[str, scenarios.RunSpec] = {}
            for s in group:
                if scenarios.load_course(s.course).get("prepare_first"):
                    first.setdefault(s.data_group, s)
            # 第一阶段：每份预处理数据各跑一个（这些之间可以并行）；第二阶段：其余全部
            for phase in (list(first.values()), [s for s in group if s not in first.values()]):
                if phase and not stop.is_set():
                    run_phase(phase, slots)
    finally:
        signal.signal(signal.SIGINT, old)

    # 同配置的场景：直接复制主运行的结果
    for s, p in aliases:
        src = paths.batch_dir() / p.rel_dir
        if (read_json(src / "run.json") or {}).get("status") == "succeeded":
            copy_alias(src, s)
            report.append(row(s, "succeeded", note=f"复用 {p.course}/{p.scenario}", minutes=0, expect_ok=None))
        else:
            report.append(row(s, "failed", note=f"复用的 {p.course}/{p.scenario} 没有成功"))

    # 依赖 baseline 的断言：全部跑完后再补查一次（包括之前批次跑完、当时还缺 baseline 的运行）
    by_dir = {str(s.rel_dir): s for s in specs}
    for r in report:
        spec = by_dir[r["dir"]]
        raw = paths.batch_dir() / spec.rel_dir
        meta = read_json(raw / "run.json") or {}
        if spec.expect and meta.get("status") == "succeeded" and meta.get("expect_ok") is None:
            r["expect_ok"] = check_expect(Run(raw), spec, allow_missing_base=False)
        elif r.get("expect_ok") is None and meta.get("expect_ok") is not None:
            r["expect_ok"] = meta["expect_ok"]
    bad = [r for r in report if r["status"] not in ("succeeded", "skipped") or r.get("expect_ok") is False]
    print(f"完成：成功 {sum(r['status'] == 'succeeded' for r in report)}，跳过 {sum(r['status'] == 'skipped' for r in report)}，"
          f"有问题 {len(bad)}", flush=True)
    for r in bad:
        print(f"  ! {r['course']}/{r['scenario']} s{r['seed']} {r['status']} expect_ok={r.get('expect_ok')} "
              f"{r.get('end_reason') or r.get('note') or ''}", flush=True)
    return report
