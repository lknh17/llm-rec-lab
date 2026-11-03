"""脱敏：把批量运行的原始目录导出成可以公开的结果，以及推送前的残留检查。

三道防线：
1. 导出白名单（export）：只复制下表里的文件；env.json 只保留白名单字段；文本统一过替换规则；
   metrics.jsonl 超过 2000 行等间隔降采样，predictions.jsonl 只保留前 200 行。模型权重、检查点一律不导出。
2. 推送前检查（check）：扫描 results/ 和 jobs/ 里每个文件：
   - 本机主机名、用户名、家目录、IPv4/IPv6、MAC 地址、邮箱、令牌（ghp_/gho_/github_pat_/hf_/sk-）；
   - LABKIT_REDACT_EXTRA 里额外登记的敏感字符串（逗号分隔，例如内部挂载前缀、集群名）；
   - 禁止的扩展名（.safetensors .bin .pt .pth .ckpt .npz .npy .pkl）和单文件 > 2 MB、单个运行 > 4 MB。
   任何一项命中，git pre-push 钩子拒绝推送。
3. GitHub Actions 在服务端再跑一遍 check（推送后才执行，只能事后发现，所以本地钩子是主防线）。
"""

from __future__ import annotations

import getpass
import json
import os
import re
import socket
from pathlib import Path

from . import paths

EXPORT_FILES = {"run.json", "config.json", "env.json", "summary.json", "trace.json", "log.txt", "metrics.jsonl",
                "predictions.jsonl"}
EXPORT_CODE = True               # code/ 快照：只含项目源码（.py/.json/.yaml/.md），公开仓库里本来就有
CODE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".md", ".txt"}
ENV_KEEP = {"device", "os", "python", "packages", "cuda", "gpus", "gpu_name", "gpu_count", "torch_cuda", "cpu_count",
            "mem_gb", "mps", "env_id", "lab_version", "compute"}
RUN_META_DROP = {"pid", "spawned_at", "pid_started", "_dir", "_origin", "cancel_reason_detail"}
MAX_FILE = 2 * 1024 * 1024
MAX_RUN = 4 * 1024 * 1024
MAX_METRIC_LINES = 2000
MAX_PRED_LINES = 200
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".npz", ".npy", ".pkl", ".pickle", ".gguf"}

TOKEN_PATTERNS = [
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), "<github-token>"),
    (re.compile(r"\bgho_[A-Za-z0-9]{20,}"), "<github-token>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "<github-token>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "<hf-token>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "<api-key>"),
]
IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
IPV6 = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{1,4}:){4,7}[0-9A-Fa-f]{1,4}(?![0-9A-Fa-f:])")
MAC = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:])")
# 量词都设上限：长串字母数字（base64、超长日志行）上无界的 [..]+@ 会退化成 O(n²) 回溯
EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,24}")
HOME = re.compile(r"(/home|/Users|/root|/data/home)/[A-Za-z0-9._-]{1,64}")
# 允许出现的"像 IP"的串：版本号之类不会是 4 段，但 0.0.0.0 / 127.0.0.1 无害
IP_ALLOW = {"0.0.0.0", "127.0.0.1", "255.255.255.255"}
EMAIL_ALLOW = re.compile(r"@(example\.com|users\.noreply\.github\.com)$")


def sensitive_strings() -> list[str]:
    """本机的主机名、用户名、家目录、仓库绝对路径，以及 LABKIT_REDACT_EXTRA 登记的字符串。"""
    out = set()
    for v in (socket.gethostname(), socket.getfqdn(), os.environ.get("HOSTNAME", "")):
        if v and v not in ("localhost", "localhost.localdomain"):
            out.add(v)
            out.add(v.split(".")[0])
    try:
        out.add(getpass.getuser())
    except Exception:  # noqa: BLE001
        pass
    out.add(str(Path.home()))
    out.add(str(paths.ROOT))
    out.add(str(paths.data_dir()))
    out.add(str(paths.assets_dir()))
    for v in os.environ.get("LABKIT_REDACT_EXTRA", "").split(","):
        if v.strip():
            out.add(v.strip())
    # 太短的串（如用户名 "a"）会误伤正文，至少 3 个字符才参与替换/检查
    return sorted((s for s in out if len(s) >= 3 and s not in ("root", "user", "ubuntu", "admin")), key=len, reverse=True)


def redact_text(text: str, extra: list[str] | None = None) -> str:
    for s in extra if extra is not None else sensitive_strings():
        if s == str(paths.ROOT):
            text = text.replace(s, "<repo>")
        elif s in (str(paths.data_dir()), str(paths.assets_dir())):
            text = text.replace(s, "<data>" if s == str(paths.data_dir()) else "<assets>")
        else:
            text = text.replace(s, "<redacted>")
    for pat, rep in TOKEN_PATTERNS:
        text = pat.sub(rep, text)
    text = HOME.sub(lambda m: m.group(1) + "/<user>", text)
    text = IPV4.sub(lambda m: m.group(0) if m.group(0) in IP_ALLOW else "<ip>", text)
    text = IPV6.sub("<ipv6>", text)
    text = MAC.sub("<mac>", text)
    text = EMAIL.sub(lambda m: m.group(0) if EMAIL_ALLOW.search(m.group(0)) else "<email>", text)
    return text


def _downsample(lines: list[str], limit: int) -> list[str]:
    if len(lines) <= limit:
        return lines
    step = len(lines) / limit
    keep = sorted({int(i * step) for i in range(limit)} | {len(lines) - 1})
    return [lines[i] for i in keep]


def _clean_env(env: dict) -> dict:
    out = {k: v for k, v in env.items() if k in ENV_KEEP}
    out["hostname"] = "<redacted>"
    return out


def export_run(src: Path, dst: Path, extra: list[str] | None = None) -> dict:
    """src：batch/ 下的原始运行目录；dst：results/ 下的目标目录。返回导出统计。"""
    extra = sensitive_strings() if extra is None else extra
    if dst.exists():
        import shutil

        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    written, skipped = [], []
    for f in sorted(src.iterdir()):
        if f.is_dir():
            if f.name == "code" and EXPORT_CODE:
                for cf in sorted(f.rglob("*")):
                    if cf.is_file() and cf.suffix in CODE_SUFFIXES and "__pycache__" not in cf.parts:
                        out = dst / cf.relative_to(src)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_text(redact_text(cf.read_text(errors="replace"), extra))
                        written.append(str(cf.relative_to(src)))
            else:
                skipped.append(f.name + "/")
            continue
        if f.name not in EXPORT_FILES:
            skipped.append(f.name)
            continue
        text = f.read_text(errors="replace")
        if f.name == "env.json":
            text = json.dumps(_clean_env(json.loads(text)), ensure_ascii=False, indent=2)
        elif f.name == "run.json":
            meta = {k: v for k, v in json.loads(text).items() if k not in RUN_META_DROP}
            meta["exported"] = True
            text = json.dumps(meta, ensure_ascii=False, indent=2)
        elif f.name == "metrics.jsonl":
            text = "\n".join(_downsample(text.splitlines(), MAX_METRIC_LINES)) + "\n"
        elif f.name == "predictions.jsonl":
            text = "\n".join(text.splitlines()[:MAX_PRED_LINES]) + "\n"
        elif f.name == "log.txt" and len(text) > MAX_FILE // 2:
            half = MAX_FILE // 4
            text = text[:half] + "\n……（日志过长，中间已省略）……\n" + text[-half:]
        (dst / f.name).write_text(redact_text(text, extra))
        written.append(f.name)
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    return {"written": written, "skipped": skipped, "bytes": size}


def export_all(profile: str | None = None, courses: list[str] | None = None) -> list[dict]:
    """batch/ 下所有成功或失败的运行（排队中、运行中的不导出）→ results/。"""
    root = paths.batch_dir()
    extra = sensitive_strings()
    out = []
    for run_json in sorted(root.glob("*/*/*/*/run.json")):
        src = run_json.parent
        rel = src.relative_to(root)
        if profile and rel.parts[0] != profile:
            continue
        if courses and rel.parts[1] not in courses:
            continue
        meta = json.loads(run_json.read_text())
        if meta.get("status") not in ("succeeded", "failed", "cancelled", "interrupted"):
            continue
        st = export_run(src, paths.RESULTS_DIR / rel, extra)
        out.append({"dir": str(rel), "status": meta.get("status"), **st})
    return out


# ------------------------------------------------------------------ 检查
def _scan_text(text: str, extra: list[str]) -> list[str]:
    hits = []
    for s in extra:
        if s in text:
            hits.append(f"本机敏感字符串 {s[:3]}…（{len(s)} 字符）")
    for pat, name in TOKEN_PATTERNS:
        if pat.search(text):
            hits.append(f"疑似令牌 {name}")
    for m in IPV4.findall(text):
        if m not in IP_ALLOW:
            hits.append(f"IPv4 {m}")
            break
    if IPV6.search(text):
        hits.append("IPv6 地址")
    if MAC.search(text):
        hits.append("MAC 地址")
    for m in EMAIL.findall(text):
        if not EMAIL_ALLOW.search(m):
            hits.append(f"邮箱 {m.split('@')[0][:2]}…@{m.split('@')[1]}")
            break
    for m in HOME.findall(text):
        hits.append(f"家目录路径 {m}/…")
        break
    return hits


def check(targets: list[Path] | None = None, extra: list[str] | None = None) -> list[dict]:
    """返回问题列表 [{path, problem}]；空列表表示通过。"""
    extra = sensitive_strings() if extra is None else extra
    targets = targets or [paths.RESULTS_DIR, paths.JOBS_DIR]
    problems = []
    run_sizes: dict[Path, int] = {}
    for t in targets:
        files = [t] if t.is_file() else [p for p in t.rglob("*") if p.is_file()]
        for f in files:
            rel = f.relative_to(paths.ROOT) if f.is_relative_to(paths.ROOT) else f
            if f.suffix.lower() in FORBIDDEN_SUFFIXES:
                problems.append({"path": str(rel), "problem": f"禁止的文件类型 {f.suffix}（模型权重/数组不能推到公开仓库）"})
                continue
            size = f.stat().st_size
            run_root = next((p for p in [f.parent, *f.parents] if (p / "run.json").exists()), None)
            if run_root is not None:
                run_sizes[run_root] = run_sizes.get(run_root, 0) + size
            if size > MAX_FILE:
                # 超限文件本身就不允许推送，不再逐字扫描（几十 MB 的文本扫一遍也要时间）
                problems.append({"path": str(rel), "problem": f"文件 {size / 1e6:.1f} MB 超过 {MAX_FILE / 1e6:.0f} MB 上限"})
                continue
            try:
                text = f.read_text()
            except UnicodeDecodeError:
                problems.append({"path": str(rel), "problem": "二进制文件（结果目录只允许文本）"})
                continue
            for h in _scan_text(text, extra):
                problems.append({"path": str(rel), "problem": h})
    for d, size in run_sizes.items():
        if size > MAX_RUN:
            problems.append({"path": str(d), "problem": f"单个运行 {size / 1e6:.1f} MB 超过 {MAX_RUN / 1e6:.0f} MB"})
    return problems
