"""lessonbench 执行入口：把 vendor/（讲义代码的逐字节拷贝）复制到运行目录下的工作区 → 按验证项依次跑命令 →
收集 stdout 和新生成的报告（md / csv / txt / json）→ summary 和 trace。

讲义脚本自己负责测量（warmup、多次取中位数）；这里只负责环境、执行、取消和超时、收证据。
环境缺失（没有 nvcc、ncu 没权限、vLLM 环境没装好）时明确记下缺什么，运行仍然完成，summary.ok = false。
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
from pathlib import Path

from .items import ITEMS, MAIN_WITH, TEXT_SUFFIXES

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
MAX_TEXT = 200_000
MAX_TOTAL = 1_500_000


def _repo_root() -> Path:
    """uv 的项目目录：代码快照在运行目录下，不是仓库根，所以从 labkit 包的位置推（可用 LABKIT_REPO 覆盖）。"""
    import labkit

    return Path(os.environ.get("LABKIT_REPO") or Path(labkit.__file__).resolve().parents[1])


def _run_quiet(cmd: list[str], timeout: float = 30) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()[-600:] if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


# --8<-- [start:probe]
def probe() -> dict:
    """各验证项需要的环境：cuda（torch 看得到 GPU）、nvcc（编译 CUDA 扩展）、ncu（Nsight Compute）、vllm（vllm-env）。"""
    import torch

    from labkit.assets import resolve_env_python

    info: dict = {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "cuda": torch.cuda.is_available()}
    if info["cuda"]:
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "gpu_mem_gb": round(p.total_memory / 1e9, 1), "sm_count": p.multi_processor_count,
                     "capability": f"{p.major}.{p.minor}"})
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:  # noqa: BLE001
        CUDA_HOME = None
    nvcc = shutil.which("nvcc") or (str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME and (Path(CUDA_HOME) / "bin" / "nvcc").exists() else None)
    info["nvcc"] = nvcc
    info["nvcc_version"] = (_run_quiet([nvcc, "--version"]) or "").splitlines()[-1:] if nvcc else None
    ncu = shutil.which("ncu") or next((str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/ncu"))), None)
    info["ncu"] = ncu
    info["ncu_version"] = (_run_quiet([ncu, "--version"]) or "").splitlines()[-1:] if ncu else None
    info["vllm_python"] = resolve_env_python("vllm-env", must_exist=False)
    info["uv"] = shutil.which("uv")
    return info


def missing(needs: list[str], env: dict) -> list[str]:
    have = {"cuda": env.get("cuda"), "nvcc": env.get("nvcc"), "ncu": env.get("ncu"), "vllm": env.get("vllm_python")}
    return [n for n in needs if not have.get(n)]
# --8<-- [end:probe]


def interpreter(kind: str, env: dict) -> list[str]:
    if kind == "vllm":
        return [env["vllm_python"]]
    if env.get("uv"):
        return [env["uv"], "run", "--project", str(_repo_root()), "--no-sync", *[x for w in MAIN_WITH for x in ("--with", w)], "python"]
    return [sys.executable]


def _child_env(kind: str, work: Path, extra: dict) -> dict:
    e = dict(os.environ)
    if kind == "vllm":                                   # 不让主环境的 torch 混进 vllm-env
        for k in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"):
            e.pop(k, None)
    e.update({"PYTHONPATH": str(work), "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "VLLM_NO_USAGE_STATS": "1",
              "TOKENIZERS_PARALLELISM": "false", **extra})
    return e


def _kill(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
        except subprocess.TimeoutExpired:
            pass


# --8<-- [start:run_step]
def run_step(ctx, cmd: list[str], work: Path, env: dict) -> dict:
    """跑一条命令，逐行转发输出到运行日志；每秒检查一次取消和超时（超了杀整个进程组）。"""
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: ([lines.put(ln) for ln in proc.stdout], lines.put(None)), daemon=True).start()
    tail: list[str] = []
    try:
        while True:
            ctx.check()
            try:
                ln = lines.get(timeout=1.0)
            except queue.Empty:
                continue
            if ln is None:
                break
            ln = ln.rstrip()
            ctx.log(ln[-400:])
            tail = (tail + [ln[-300:]])[-40:]
        proc.wait(30)
    except BaseException:
        _kill(proc)
        raise
    return {"rc": proc.returncode, "seconds": round(time.time() - t0, 1), "tail": tail}
# --8<-- [end:run_step]


def _snapshot(work: Path) -> dict[str, float]:
    return {str(p.relative_to(work)): p.stat().st_mtime for p in work.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def collect(work: Path, before: dict[str, float]) -> tuple[dict[str, str], list[str]]:
    """新生成或改过的文件：文本类收全文（单个 ≤ 200 KB、合计 ≤ 1.5 MB），其余只记文件名（图片、.ncu-rep、检查点）。"""
    texts, others, total = {}, [], 0
    for rel, mt in sorted(_snapshot(work).items()):
        if before.get(rel) == mt:
            continue
        p = work / rel
        if p.suffix in TEXT_SUFFIXES and p.stat().st_size <= MAX_TEXT and total < MAX_TOTAL:
            t = p.read_text(errors="replace")
            texts[rel], total = t, total + len(t)
        else:
            others.append(f"{rel}（{p.stat().st_size / 1e6:.1f} MB）")
    return texts, others


def run(ctx, cfg: dict) -> None:
    item = ITEMS[cfg["item"]]
    ctx.stage("prepare")
    env = probe()
    ctx.info("model", {"arch": "讲义代码（vendor/ 逐字节拷贝）", "params": {"total": 0, "trainable": 0}, "train_mode": "benchmark",
                       "item": cfg["item"], "lesson": item["lesson"]})
    ctx.log(f"验证项 {cfg['item']}（{item['lesson']}）：{item['title']}")
    ctx.log("环境：" + "，".join(f"{k}={v}" for k, v in env.items()))
    lack = missing(item["needs"], env)
    work = ctx.artifact("work")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(VENDOR, work, ignore=shutil.ignore_patterns("__pycache__"))
    before = _snapshot(work)
    steps: list[dict] = []
    ctx.stage("train")
    if lack:
        ctx.warn("env_missing", f"缺少 {lack}，这一项没有执行（见 summary.env）")
    else:
        for kind, args in item["steps"]:
            args = [str(HERE / a[1:]) if a.startswith("@") else a.replace("{hw}", cfg["hw"]) for a in args]
            cmd = interpreter(kind, env) + args
            shown = f"[{kind}] python " + " ".join(args).replace(str(HERE) + "/", "")
            ctx.log(f"$ {shown}")
            r = run_step(ctx, cmd, work, _child_env(kind, work, {}))
            steps.append({"cmd": shown, **r})
            if r["rc"] != 0:
                ctx.warn("step_failed", f"{shown} 退出码 {r['rc']}")
                break
    ctx.stage("evaluate")
    texts, others = collect(work, before)
    ok = not lack and bool(steps or not item["steps"]) and all(s["rc"] == 0 for s in steps)
    ctx.trace("outputs", {"note": "这一项新生成的报告（讲义脚本写在 experiments/ 下的 md / csv），全文", "files": texts})
    ctx.trace("steps", {"note": "每条命令最后 40 行输出", "steps": steps})
    seconds = round(sum(s["seconds"] for s in steps), 1)
    ctx.summary({"item": cfg["item"], "lesson": item["lesson"], "title": item["title"], "ok": ok, "missing": lack, "env": env,
                 "steps": [{k: s[k] for k in ("cmd", "rc", "seconds")} for s in steps], "outputs": sorted(texts), "binary_outputs": others,
                 "seconds": seconds, "headline": {"ok": int(ok), "seconds": seconds},
                 "display": {"通过": int(ok), "用时 s": seconds}})
    ctx.log(f"{'通过' if ok else '未通过'}：{len(steps)} 条命令，{seconds} 秒；报告 {sorted(texts) or '无'}；缺少 {lack or '无'}")
    if steps and steps[-1]["rc"] != 0:
        raise RuntimeError(f"{steps[-1]['cmd']} 失败（退出码 {steps[-1]['rc']}）：\n" + "\n".join(steps[-1]["tail"][-15:]))
