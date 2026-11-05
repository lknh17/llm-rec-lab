"""执行一个运行目录：python -m labkit.worker <run_dir>

- 从 <run_dir>/code 快照 import 项目代码执行（不是从工作区），保证展示的代码就是执行的代码；
- 拿设备锁：同一把锁一次只跑一个作业，排队的作业在这里等。锁名默认是环境编号（本机一次一个），
  批量运行时由 labkit.batch 设成 LABKIT_DEVICE_LOCK=gpu<k>[-slot<j>]，实现按卡并行；
- 心跳线程每 3 秒写一次时间戳，服务进程据此判断进程是否还活着；
- SIGTERM -> 置取消标志，训练循环在 ctx.check() 处退出，状态记为 cancelled 并写明原因。
"""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from . import paths
from .probe import probe
from .runtime import BudgetExceeded, Cancelled, RunContext
from .store import Run, write_json


def _heartbeat(run_dir: Path, stop: threading.Event) -> None:
    while not stop.is_set():
        (run_dir / "heartbeat").write_text(str(time.time()))
        stop.wait(3)


def _record_resources(run_dir: Path, wall_s: float) -> None:
    """显存峰值（只有项目代码已经导入 torch 时才记；CUDA 记峰值，MPS 只有当前分配量）→ resources.json。"""
    out: dict = {"wall_seconds": round(wall_s, 1)}
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            dev = torch.cuda.current_device()
            out.update(device="cuda", gpu_name=torch.cuda.get_device_name(dev),
                       peak_allocated_bytes=int(torch.cuda.max_memory_allocated(dev)),
                       peak_reserved_bytes=int(torch.cuda.max_memory_reserved(dev)),
                       device_total_bytes=int(torch.cuda.get_device_properties(dev).total_memory))
        elif torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            out.update(device="mps", current_allocated_bytes=int(torch.mps.current_allocated_memory()),
                       driver_allocated_bytes=int(torch.mps.driver_allocated_memory()))
    except Exception as e:  # noqa: BLE001  记录资源失败不影响运行结果
        out["error"] = f"{type(e).__name__}: {e}"
    write_json(run_dir / "resources.json", out)


def execute(run_dir: Path) -> str:
    run = Run(run_dir)
    meta = run.meta()
    cfg = run.config()
    cancel_flag = [False]

    def on_term(signum, frame):  # noqa: ARG001
        if cancel_flag[0]:
            raise SystemExit(143)
        cancel_flag[0] = True
        print(f"[worker] 收到信号 {signum}，将在下一个检查点停止", flush=True)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    lock_dir = paths.sub("locks")
    lock_name = os.environ.get("LABKIT_DEVICE_LOCK") or meta.get("env_id", "local")
    lock_file = open(lock_dir / f"{lock_name}.lock", "w")
    print(f"[worker] 等待设备锁 {lock_name}", flush=True)
    while True:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if cancel_flag[0]:
                run.set_status("cancelled", "排队中被取消")
                return "cancelled"
            time.sleep(1)

    stop = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(run_dir, stop), daemon=True)
    hb.start()
    status, reason = "failed", ""
    try:
        run.update(pid_started=time.time())
        run.set_status("preparing", "worker 启动")
        env = probe(compute=False)
        write_json(run_dir / "env.json", env)
        print(f"[worker] 环境：{env.get('hostname')} {env.get('os')} device={env.get('device')} "
              f"torch={env['packages'].get('torch')}", flush=True)

        sys.path.insert(0, str(run_dir / "code"))
        module = importlib.import_module(meta["entry"])
        print(f"[worker] 从快照加载 {module.__file__}", flush=True)
        ctx = RunContext(run, cfg, cancel_flag)
        module.run(ctx, cfg)
        status, reason = "succeeded", "完成"
    except Cancelled as e:
        status, reason = "cancelled", str(e) or "用户取消"
    except BudgetExceeded as e:
        status, reason = "failed", f"budget: {e}"
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        status, reason = "failed", f"{type(e).__name__}: {e}"
    finally:
        stop.set()
        _record_resources(run_dir, time.time() - (run.meta().get("pid_started") or time.time()))
        (run_dir / "heartbeat").write_text(str(time.time()))
        run.set_status(status, reason)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        print(f"[worker] 结束：{status}（{reason}）", flush=True)
    return status


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    status = execute(Path(sys.argv[1]).resolve())
    print(json.dumps({"status": status}))
    sys.exit(0 if status == "succeeded" else 1)


if __name__ == "__main__":
    main()
