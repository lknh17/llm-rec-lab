"""把 results/、jobs/、status/ 和课程 README（结果表、证据图）推回 GitHub。推送前一定先跑脱敏检查。

    python -m labkit publish -m "results: L1 h20x8"

步骤：redact.check() → git add results jobs → commit → pull --rebase（Mac 端可能改了代码）→ push。
推送冲突只会出现在 jobs/ 同一个文件被两边同时改的情况；遇到时停下来报告，不做强推。
"""

from __future__ import annotations

import subprocess
import time

from . import paths, redact


GIT_TIMEOUT_S = 900
# 网络停滞（每秒不到 1 KB、持续 120 秒）时让 git 自己放弃，不要无限挂住（H20 到 GitHub 的连接偶尔会卡死）
GIT_NET_OPTS = ("-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=120")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # 关掉目录重命名检测：作业单在 pending/ running/ done/ 之间移动时，它会把对方新加的作业单也挪走造成假冲突
    try:
        r = subprocess.run(["git", "-c", "merge.directoryRenames=false", *GIT_NET_OPTS, *args], cwd=paths.ROOT,
                           capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"git {' '.join(args)} 超过 {GIT_TIMEOUT_S} 秒没结束（网络卡住？），下一轮补推") from e
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{r.stderr.strip() or r.stdout.strip()}")
    return r


def publish(message: str, paths_to_add: list[str] | None = None, push: bool = True) -> dict:
    if paths_to_add is None:
        from . import docs

        paths_to_add = ["results", "jobs", "status"] + docs.outputs()
    add = [p for p in paths_to_add if (paths.ROOT / p).exists()]
    if not add:
        print("没有需要提交的路径")
        return {"committed": False}
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
    # 远端常有并发推送（Mac 端、pages 工作流），push 被拒时重新 pull --rebase 再推，最多 3 次
    for attempt in range(3):
        r = _git("pull", "--rebase", "--autostash", check=False)
        if r.returncode != 0:
            raise SystemExit(f"pull --rebase 冲突，停止推送，请人工处理：\n{r.stderr or r.stdout}")
        r = _git("push", check=False)
        if r.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError(f"git push 失败：{r.stderr.strip() or r.stdout.strip()}")
        time.sleep(5 * (attempt + 1))
    print(f"已推送 {len(staged)} 个文件")
    return {"committed": True, "pushed": True, "files": len(staged)}
