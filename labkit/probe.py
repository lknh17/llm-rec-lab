"""环境探测与预检。只依赖标准库和（可选的）torch，可以单独拷到远端机器执行：

    python -m labkit.probe            # 打印 JSON
    python -m labkit.probe --compute  # 额外做一次最小前向反向，证明设备真的能算

预检通过 != GPU 接入完成。验收必须看 compute 一节里实际在 GPU 上跑出的结果。
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _cmd(args: list[str], timeout: float = 10) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _pkg_versions() -> dict:
    from importlib import metadata

    names = ["torch", "transformers", "peft", "numpy", "safetensors", "accelerate", "fastapi"]
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def probe(compute: bool = False) -> dict:
    info: dict = {
        "time": time.time(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": _pkg_versions(),
        "cpu_count": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        info["chip"] = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _cmd(["sysctl", "-n", "hw.memsize"])
        info["memory_gb"] = round(int(mem) / 1e9, 1) if mem else None
    smi = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"])
    info["nvidia_smi"] = smi.splitlines() if smi else None
    from .paths import data_dir

    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(root)
    info["disk_free_gb"] = round(du.free / 1e9, 1)
    info["data_dir_writable"] = os.access(root, os.W_OK)

    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            info["gpus"] = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                info["gpus"].append({"index": i, "name": p.name, "total_gb": round(total / 1e9, 1),
                                     "free_gb": round(free / 1e9, 1), "capability": f"{p.major}.{p.minor}"})
            info["device"] = "cuda"
        elif info["mps_available"]:
            info["device"] = "mps"
        else:
            info["device"] = "cpu"
        if compute:
            info["compute"] = compute_check(info["device"])
    except ImportError:
        info["torch_error"] = "torch 未安装"
    return info


def compute_check(device_name: str) -> dict:
    """最小前向 + 反向：一个 2 层 MLP 在目标设备上算一步，并和 CPU 结果核对。"""
    import torch

    torch.manual_seed(0)
    x = torch.randn(256, 512)
    w1 = torch.randn(512, 1024, requires_grad=True)
    w2 = torch.randn(1024, 1, requires_grad=True)

    def step(dev: str) -> tuple[float, float, float]:
        a, b = w1.detach().to(dev).requires_grad_(), w2.detach().to(dev).requires_grad_()
        t0 = time.perf_counter()
        loss = (torch.relu(x.to(dev) @ a) @ b).pow(2).mean()
        loss.backward()
        g = a.grad.norm().item()
        return loss.item(), g, (time.perf_counter() - t0) * 1e3

    ref_loss, ref_grad, _ = step("cpu")
    try:
        loss, grad, ms = step(device_name)
        ok = abs(loss - ref_loss) / max(abs(ref_loss), 1e-6) < 1e-2 and abs(grad - ref_grad) / max(ref_grad, 1e-6) < 1e-2
        return {"device": device_name, "ok": ok, "loss": loss, "cpu_loss": ref_loss, "grad_norm": grad,
                "cpu_grad_norm": ref_grad, "ms": round(ms, 2)}
    except Exception as e:  # noqa: BLE001
        return {"device": device_name, "ok": False, "error": f"{type(e).__name__}: {e}"}


def assets_status() -> dict:
    """课程需要的数据和模型文件是否齐全。缺什么就明确说缺什么。"""
    from .assets import status

    return status()


if __name__ == "__main__":
    result = probe(compute="--compute" in sys.argv)
    result["assets"] = assets_status()
    print(json.dumps(result, ensure_ascii=False, indent=2))
