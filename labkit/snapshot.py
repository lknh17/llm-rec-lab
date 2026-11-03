"""源码快照与代码差异。

每次运行把项目代码复制到 <run_dir>/code/<project>/，worker 从快照 import 执行，
网页展示的关键代码也从快照读——保证"看到的就是跑的"。
"""

from __future__ import annotations

import difflib
import hashlib
import re
import shutil
from pathlib import Path

from . import paths

SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
MAX_FILE_BYTES = 2_000_000

REGION_START = re.compile(r"#\s*--8<--\s*\[start:([\w-]+)\]")
REGION_END = re.compile(r"#\s*--8<--\s*\[end:([\w-]+)\]")


def _files(project_dir: Path) -> list[Path]:
    out = []
    for p in sorted(project_dir.rglob("*")):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts) or p.suffix in SKIP_SUFFIXES:
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            continue
        out.append(p)
    return out


def tree_manifest(project_dir: Path) -> dict:
    files = {}
    for p in _files(project_dir):
        files[str(p.relative_to(project_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    digest = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(files.items())).encode()).hexdigest()[:12]
    return {"files": files, "code_hash": digest}


def project_dir(project: str) -> Path:
    return paths.PROJECTS_DIR / project


def snapshot(project: str, run_dir: Path) -> dict:
    src = project_dir(project)
    dest = run_dir / "code" / project
    if dest.exists():
        shutil.rmtree(dest)
    for p in _files(src):
        target = dest / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    manifest = tree_manifest(src)
    manifest["project"] = project
    return manifest


def read_snapshot_file(run_dir: Path, rel: str) -> str | None:
    path = (run_dir / "code" / rel).resolve()
    if not str(path).startswith(str((run_dir / "code").resolve())) or not path.exists():
        return None
    return path.read_text(errors="replace")


def extract_region(text: str, region: str) -> tuple[str, int] | None:
    """取出 `# --8<-- [start:region]` 和 `[end:region]` 之间的代码，返回 (代码, 起始行号)。"""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = REGION_START.search(line)
        if m and m.group(1) == region:
            start = i + 1
            continue
        m = REGION_END.search(line)
        if m and m.group(1) == region and start is not None:
            body = lines[start:i]
            # 去掉公共缩进，方便阅读
            indents = [len(l) - len(l.lstrip()) for l in body if l.strip()]
            cut = min(indents) if indents else 0
            return "\n".join(l[cut:] for l in body), start + 1
    return None


def list_regions(text: str) -> list[str]:
    return [m.group(1) for line in text.splitlines() if (m := REGION_START.search(line))]


def diff_snapshots(dir_a: Path, dir_b: Path) -> list[dict]:
    """两个运行的代码差异（unified diff）。"""
    code_a, code_b = dir_a / "code", dir_b / "code"
    files_a = {str(p.relative_to(code_a)) for p in code_a.rglob("*") if p.is_file()} if code_a.exists() else set()
    files_b = {str(p.relative_to(code_b)) for p in code_b.rglob("*") if p.is_file()} if code_b.exists() else set()
    out = []
    for rel in sorted(files_a | files_b):
        if rel.endswith(".pyc") or "__pycache__" in rel or rel == "MANIFEST.json":
            continue
        a = (code_a / rel).read_text(errors="replace").splitlines() if rel in files_a else []
        b = (code_b / rel).read_text(errors="replace").splitlines() if rel in files_b else []
        if a == b:
            continue
        status = "added" if rel not in files_a else "removed" if rel not in files_b else "modified"
        text = "\n".join(difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}", lineterm="", n=3))
        out.append({"file": rel, "status": status, "diff": text})
    return out
