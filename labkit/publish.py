"""把 results/ 和 jobs/ 推回 GitHub。推送前一定先跑脱敏检查。

    python -m labkit publish -m "results: L1 h20x8"

步骤：redact.check() → git add results jobs → commit → pull --rebase（Mac 端可能改了代码）→ push。
推送冲突只会出现在 jobs/ 同一个文件被两边同时改的情况；遇到时停下来报告，不做强推。
"""

from __future__ import annotations

import subprocess

from . import paths, redact


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{r.stderr.strip() or r.stdout.strip()}")
    return r


def publish(message: str, paths_to_add: list[str] | None = None, push: bool = True) -> dict:
    add = paths_to_add or ["results", "jobs"]
    problems = redact.check([paths.ROOT / p for p in add])
    if problems:
        for p in problems[:30]:
            print(f"  ✗ {p['path']}: {p['problem']}")
        raise SystemExit(f"脱敏检查没通过（{len(problems)} 处），不推送。修复后重跑 python -m labkit redact --check")
    _git("add", "--", *add)
    staged = _git("diff", "--cached", "--name-only").stdout.split()
    outside = [f for f in staged if not any(f == a or f.startswith(a.rstrip("/") + "/") for a in add)]
    if outside:
        raise SystemExit(f"暂存区里有 {add} 以外的文件，H20 端不应改代码：{outside[:10]}。先 git restore --staged 它们")
    if not staged:
        print("没有需要提交的变化")
        return {"committed": False}
    _git("commit", "-m", message)
    if not push:
        return {"committed": True, "pushed": False, "files": len(staged)}
    r = _git("pull", "--rebase", "--autostash", check=False)
    if r.returncode != 0:
        raise SystemExit(f"pull --rebase 冲突，停止推送，请人工处理：\n{r.stderr or r.stdout}")
    _git("push")
    print(f"已推送 {len(staged)} 个文件")
    return {"committed": True, "pushed": True, "files": len(staged)}
