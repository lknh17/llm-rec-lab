"""H20 常驻 agent：盯着 GitHub 上的作业单，自动领取、执行、推回脱敏结果和心跳。

    python -m labkit agent --profile h20x8               # 常驻（systemd 见 deploy/labkit-agent.service）
    python -m labkit agent --profile h20x8 --once        # 只处理一轮（会真的领取并执行作业！）
    python -m labkit agent --profile h20x8 --check       # 只读检查：pull、列出待领取作业、校验、预览要跑的运行；不领取、不执行、不推送

每一轮：
1. git pull --rebase：拿到 Mac 端新下发的作业单和代码；
2. 抢占：最早的 pending 作业（档位匹配）→ running，写上 claimed_by，**只提交这张作业单**并推送。
   推送被拒（别人先推了）就 pull --rebase；同一张作业单冲突说明被别的 agent 抢走了，放弃这次抢占（rebase --abort
   并丢掉自己的抢占提交），没有冲突就重推。抢占成功才开始跑，保证一张作业单只跑一次；
3. 校验：作业单字段在白名单内，课程、档位、场景编号、种子都在场景清单里，requested_commit 是当前 HEAD 的祖先。
   不通过就移到 done（status: rejected）并写明原因，不执行任何东西；
4. 执行：jobs.run_job（批量运行 → 导出脱敏结果 → 写回报），期间心跳线程每 15 分钟推一次进度；
5. 发布：labkit publish（脱敏检查 → 只提交 results/ jobs/ status/ → 推送）；
6. 心跳：status/<档位>.json，只含白名单字段（状态、作业编号、进度、时间、GPU 型号与数量、agent 版本），
   状态变化时立即推送；状态不变时，运行中每 15 分钟、空闲每 3 小时推一次（beat_every_s 写明本次间隔，
   读心跳的一方按"超过两个间隔没更新"判离线），避免空闲时刷提交历史（每次推送还会触发 pages 工作流再提交一次）；
7. 日志：LAB_DATA_DIR/agent/agent.log，5 MB × 5 份轮转；
8. 代码更新：每次 pull 后比较 labkit/ 的 git 树哈希，和启动时不同就在两张作业之间重启自己（os.execv），
   保证 Mac 端修好的 labkit 代码不用人工重启就能生效。项目代码 projects/ 每个运行都从快照重新加载，不受影响。

agent 不改代码、不执行作业单里的任意命令：作业单只能指定场景编号和种子，参数全部来自仓库里的场景清单。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import logging.handlers
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import paths

AGENT_VERSION = 4
HEARTBEAT_EVERY = 15 * 60  # 运行中 / 推送中：进度要能看到
HEARTBEAT_IDLE_EVERY = 3 * 3600  # 空闲 / 出错：只证明 agent 还活着
BUSY_STATES = {"running", "publishing"}
JOB_KEYS = {"job_id", "created_at", "profile", "courses", "only", "seeds", "requested_commit", "note", "how_to_run",
            # agent / run_job 自己写的字段（作业单被重新领取时会带着）
            "claimed_by", "claimed_at", "started_at", "ran_commit", "host_summary", "status", "finished_at", "hours",
            "runs", "n_exported", "problems", "cancelled_at", "cancel_reason", "rejected_reason"}
STATUS_KEYS = {"agent_version", "agent_id", "profile", "state", "job_id", "progress", "updated_at", "last_pull_at",
               "last_error", "started_at", "gpu", "gpu_count", "gpu_mem_gb", "queue", "beat_every_s"}
_git_lock = threading.Lock()
log = logging.getLogger("labkit.agent")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# 作业单在 pending/ running/ done/ 之间移动，git 的"目录重命名检测"会把对方新加进 pending/ 的作业单也挪走，
# 造成假冲突，所以关掉。
GIT_OPTS = ("-c", "merge.directoryRenames=false", "-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=120")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        r = subprocess.run(["git", *GIT_OPTS, *args], cwd=paths.ROOT, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"git {' '.join(args)} 超过 600 秒没结束（网络卡住？）") from e
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{(r.stderr or r.stdout).strip()[:500]}")
    return r


def agent_id() -> str:
    """随机生成并保存在 LAB_DATA_DIR 里的编号（不用主机名，主机名不能进公开仓库）。"""
    p = paths.sub("agent") / "agent_id"
    if not p.exists():
        p.write_text(f"agent-{secrets.token_hex(3)}")
    return p.read_text().strip()


def setup_logging() -> None:
    if log.handlers:
        return
    h = logging.handlers.RotatingFileHandler(paths.sub("agent") / "agent.log", maxBytes=5 * 1024 * 1024, backupCount=5)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
    s = logging.StreamHandler()
    s.setFormatter(logging.Formatter("[agent %(asctime)s] %(message)s", "%H:%M:%S"))
    log.addHandler(s)
    log.setLevel(logging.INFO)


# ------------------------------------------------------------------ 抢占
def pull() -> bool:
    with _git_lock:
        try:
            r = git("pull", "--rebase", "--autostash", check=False)
        except RuntimeError as e:
            log.warning("pull 失败：%s", e)
            return False
    if r.returncode != 0:
        log.warning("pull 失败：%s", (r.stderr or r.stdout).strip()[:300])
        return False
    return True


def unpublished() -> list[str]:
    """results/ jobs/ 下还没提交推送的改动（上一张作业发布失败留下的）。"""
    r = git("status", "--porcelain", "--untracked-files=all", "--", "results", "jobs", check=False)
    return [line[3:] for line in r.stdout.splitlines() if line.strip()]


def flush_unpublished(hb: "Heartbeat") -> bool:
    """有没推出去的结果时先补推；补推失败就不领新作业、不续跑（否则会在脏工作区上反复重跑同一张作业）。"""
    left = unpublished()
    if not left:
        return True
    from .publish import publish

    log.warning("本地有 %d 个文件没推送（例如 %s），先补推", len(left), left[0])
    with _git_lock:
        try:
            publish("results: 补推上次没推出去的结果")
            return True
        except (SystemExit, RuntimeError) as e:
            err = e
    log.error("补推失败，暂停领取和续跑，需要人工处理：%s", err)
    hb.update(state="error", last_error=f"本地有没推送的结果且补推失败，暂停领取：{err}"[:300])  # 锁外调用：update 自己会拿锁
    return False


def pending_jobs(profile: str) -> list[dict]:
    from . import jobs

    return sorted((j for j in jobs.list_jobs() if j["_state"] == "pending" and j.get("profile") == profile),
                  key=lambda j: (str(j.get("created_at") or ""), j["job_id"]))


def claim(job_id: str, me: str, push: bool = True) -> bool:
    """pending → running 并推送。返回 True 表示抢到了；False 表示被别人抢走（本地已恢复到远端状态）。"""
    from . import jobs

    with _git_lock:
        src = paths.JOBS_DIR / "pending" / f"{job_id}.yaml"
        if not src.exists():
            return False
        dst = jobs.move(job_id, "running", claimed_by=me, claimed_at=_now())
        rel_src, rel_dst = str(src.relative_to(paths.ROOT)), str(dst.relative_to(paths.ROOT))
        git("add", "-A", "--", rel_src, rel_dst)
        git("commit", "-m", f"jobs: {me} 领取 {job_id}", "--", rel_src, rel_dst)
        if not push:
            return True
        for _ in range(3):
            if git("push", check=False).returncode == 0:
                return True
            r = git("pull", "--rebase", "--autostash", check=False)
            if r.returncode != 0:
                git("rebase", "--abort", check=False)
                # 只有远端已经不在 pending/ 才算被别人抢走；否则是本地问题（例如工作区不干净），不能误判后去续跑别的作业
                git("fetch", check=False)
                taken = git("cat-file", "-e", f"@{{u}}:jobs/pending/{job_id}.yaml", check=False).returncode != 0
                git("reset", "--hard", "HEAD~1")
                git("pull", "--rebase", "--autostash", check=False)
                if taken:
                    log.info("作业 %s 被别的 agent 领取，放弃", job_id)
                else:
                    log.warning("领取 %s 时 pull --rebase 失败（远端仍是 pending，不是被抢）：%s", job_id,
                                (r.stderr or r.stdout).strip()[:300])
                return False
            state = jobs.find(job_id)
            if not state or state[0] != "running" or (jobs.load(job_id).get("claimed_by") != me):
                return False
        log.warning("推送领取 %s 连续失败，放弃", job_id)
        git("reset", "--hard", "@{u}", check=False)
        return False


# ------------------------------------------------------------------ 校验
def validate_job(job: dict) -> list[str]:
    """作业单只能指定课程、档位、场景编号、种子；参数一律来自场景清单。"""
    from . import scenarios

    errs = []
    extra = set(job) - JOB_KEYS - {"_state"}
    if extra:
        errs.append(f"作业单有不认识的字段 {sorted(extra)}（只接受 {sorted(JOB_KEYS)}）")
    known = scenarios.course_files()
    courses = job.get("courses") or []
    if not courses or not isinstance(courses, list):
        errs.append("courses 为空")
    for c in courses:
        if c not in known:
            errs.append(f"未知课程 {c}")
            continue
        course = scenarios.load_course(c)
        if job.get("profile") not in course["profiles"]:
            errs.append(f"{c} 没有档位 {job.get('profile')}")
    if not all(isinstance(s, int) for s in job.get("seeds") or []):
        errs.append("seeds 只能是整数")
    only = job.get("only") or []
    if only:
        ids = set()
        for c in courses:
            if c in known:
                ids |= {s["id"] for s in scenarios.load_course(c)["scenarios"]}
        bad = [s for s in only if s not in ids]
        if bad:
            errs.append(f"场景编号不在场景清单里：{bad}")
    rc = job.get("requested_commit") or ""
    if rc:
        if git("cat-file", "-e", f"{rc}^{{commit}}", check=False).returncode != 0:
            errs.append(f"requested_commit {rc[:10]} 在本地仓库里不存在（先 pull）")
        elif git("merge-base", "--is-ancestor", rc, "HEAD", check=False).returncode != 0:
            errs.append(f"requested_commit {rc[:10]} 不是当前 HEAD 的祖先")
    return errs


# ------------------------------------------------------------------ 心跳
class Heartbeat:
    def __init__(self, profile: str, me: str, push: bool = True):
        self.profile, self.me, self.push = profile, me, push
        self.state: dict = {}
        self.last_pushed_state: str | None = None
        self.last_push = 0.0
        self.host = {}

    def path(self) -> Path:
        d = paths.ROOT / "status"
        d.mkdir(exist_ok=True)
        return d / f"{self.profile}.json"

    def update(self, force: bool = False, **fields) -> bool:
        """写心跳文件；状态变化，或超过本状态的心跳间隔（运行中 15 分钟、空闲 3 小时）才提交推送。返回是否推送。"""
        from . import jobs

        if not self.host:
            hs = jobs.host_summary()
            self.host = {"gpu": hs.get("gpu"), "gpu_count": hs.get("gpu_count"), "gpu_mem_gb": hs.get("gpu_mem_gb")}
        if isinstance(fields.get("last_error"), str):  # 异常信息常带本机路径，心跳文件也会推到公开仓库
            from . import redact

            fields["last_error"] = redact.redact_text(fields["last_error"])
        self.state.update(fields)
        every = HEARTBEAT_EVERY if self.state.get("state") in BUSY_STATES else HEARTBEAT_IDLE_EVERY
        data = {"agent_version": AGENT_VERSION, "agent_id": self.me, "profile": self.profile, **self.host, **self.state,
                "beat_every_s": every, "updated_at": time.time()}
        data = {k: v for k, v in data.items() if k in STATUS_KEYS}
        key = json.dumps({k: data.get(k) for k in ("state", "job_id", "last_error")}, sort_keys=True)  # 进度变化不单独推
        due = force or key != self.last_pushed_state or time.time() - self.last_push >= every
        if not due:
            return False
        self.path().write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        if not self.push:
            self.last_pushed_state, self.last_push = key, time.time()
            return False
        from .publish import publish

        with _git_lock:
            try:
                publish(f"status: {self.profile} {data.get('state')}", paths_to_add=["status"], push=True)
            except (SystemExit, RuntimeError) as e:
                log.warning("心跳推送失败：%s", e)
                return False
        self.last_pushed_state, self.last_push = key, time.time()
        return True


def _progress(job: dict) -> str:
    """batch/ 下这张作业单涉及的运行有几个已经结束（粗略进度，只用于心跳显示）。"""
    from . import scenarios

    total = done = 0
    for c in job.get("courses") or []:
        try:
            specs = scenarios.expand(c, job["profile"], job.get("only") or None, job.get("seeds") or None)
        except (KeyError, ValueError):
            continue
        for s in specs:
            total += 1
            meta = paths.batch_dir() / s.rel_dir / "run.json"
            try:
                if json.loads(meta.read_text()).get("status") in ("succeeded", "failed", "cancelled", "interrupted"):
                    done += 1
            except (OSError, ValueError):
                pass
    return f"{done}/{total}"


# ------------------------------------------------------------------ 代码更新后自动重启
def code_version() -> str | None:
    """当前检出的 labkit/ 的 git 树哈希。只看 labkit/：它是 agent 进程里已经 import 的代码。"""
    r = git("rev-parse", "HEAD:labkit", check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def restart_if_code_changed(started_with: str | None) -> None:
    now = code_version()
    if started_with and now and now != started_with:
        log.info("labkit 代码已更新（%s → %s），重启 agent 以加载新代码", started_with[:8], now[:8])
        for h in log.handlers:
            h.flush()
        os.execv(sys.executable, [sys.executable, "-m", "labkit", *sys.argv[1:]])


# ------------------------------------------------------------------ 只读检查
def check(profile: str = "h20x8") -> int:
    """只读：pull 后列出可领取的作业、逐张校验、预览会跑哪些运行。不领取、不执行、不写作业单、不推送。
    返回不通过的作业数（命令行据此给退出码）。"""
    from . import batch, scenarios, snapshot

    print(f"agent 版本 {AGENT_VERSION}，labkit 树 {(code_version() or '?')[:8]}，档位 {profile}")
    print("git pull：" + ("成功" if pull() else "失败（看 git 输出）"))
    todo = pending_jobs(profile)
    if not todo:
        print("没有待领取的作业单（jobs/pending/ 里没有这个档位的作业）")
        return 0
    bad = 0
    for job in todo:
        errs = validate_job(job)
        print(f"\n== {job['job_id']}：{'校验通过' if not errs else '校验不通过'}")
        for e in errs:
            print(f"   ✗ {e}")
        if errs:
            bad += 1
            continue
        specs = []
        for c in job["courses"]:
            specs += scenarios.expand(c, profile, job.get("only") or None, job.get("seeds") or None)
        code = {pr: snapshot.tree_manifest(snapshot.project_dir(pr))["code_hash"] for pr in {s.project for s in specs}}
        done = sum(1 for s in specs if batch.existing_status(s, code[s.project]))
        by: dict[str, int] = {}
        for s in specs:
            by[s.scenario] = by.get(s.scenario, 0) + 1
        print(f"   共 {len(specs)} 个运行，已有同代码成功结果、会跳过 {done} 个")
        print("   " + "  ".join(f"{k}×{v}" for k, v in by.items()))
    print("\n只读检查结束：没有领取作业、没有执行、没有推送。")
    return bad


# ------------------------------------------------------------------ 主循环
def process_one(profile: str, me: str, hb: Heartbeat, max_slots: int | None = None, push: bool = True) -> str | None:
    """处理一张作业单。返回作业编号（没有可领取的作业时返回 None）。"""
    from . import jobs
    from .publish import publish

    resumable = [j for j in jobs.list_jobs() if j["_state"] == "running" and j.get("profile") == profile
                 and j.get("claimed_by") == me]  # 自己领过、没跑完（agent 重启）：直接续跑，不重新抢
    for j in resumable + pending_jobs(profile):
        job_id = j["job_id"]
        if j["_state"] == "pending" and not claim(job_id, me, push=push):
            continue
        if j["_state"] == "running":
            log.info("续跑作业 %s", job_id)
        job = jobs.load(job_id)
        errs = validate_job(job)
        if errs:
            log.warning("作业 %s 校验不通过：%s", job_id, errs)
            jobs.move(job_id, "done", status="rejected", finished_at=_now(), rejected_reason=errs, problems=errs)
            if push:
                with _git_lock:
                    publish(f"jobs: 拒绝 {job_id}", paths_to_add=["jobs"])
            hb.update(state="idle", job_id=None, progress=None, last_error=f"拒绝 {job_id}：{errs[0]}"[:300])
            return job_id
        log.info("开始作业 %s", job_id)
        hb.update(state="running", job_id=job_id, progress=_progress(job), started_at=time.time(), last_error=None)
        stop = threading.Event()

        def beat() -> None:
            while not stop.wait(60):
                hb.update(progress=_progress(job))

        t = threading.Thread(target=beat, daemon=True)
        t.start()
        try:
            res = jobs.run_job(job_id, max_slots, publish_each=False)
            err = None
        except Exception as e:  # noqa: BLE001  一张作业失败不影响 agent 继续
            log.exception("作业 %s 异常", job_id)
            res, err = {"status": "failed"}, f"{type(e).__name__}: {e}"
            if (jobs.find(job_id) or ("",))[0] == "running":
                jobs.move(job_id, "done", status="failed", finished_at=_now(), problems=[err[:500]])
        finally:
            stop.set()
            t.join(timeout=5)
        hb.update(state="publishing", progress=_progress(job))
        if push:
            with _git_lock:
                try:
                    publish(f"results: {job_id} {res.get('status')}")  # results jobs status + 课程 README/证据图
                except (SystemExit, RuntimeError) as e:
                    err = f"发布失败：{e}"
                    log.error(err)
        log.info("作业 %s 结束：%s", job_id, res.get("status"))
        hb.update(state="idle", job_id=None, progress=None, last_error=err[:300] if err else None)
        return job_id
    return None


def run(profile: str = "h20x8", once: bool = False, interval: int = 60, max_slots: int | None = None,
        push: bool = True) -> None:
    setup_logging()
    me = agent_id()
    hb = Heartbeat(profile, me, push=push)
    started_with = code_version()
    log.info("agent %s 启动（版本 %s，档位 %s，labkit 树 %s）", me, AGENT_VERSION, profile, (started_with or "?")[:8])
    while True:
        ok = pull()
        hb.state["last_pull_at"] = time.time()
        if ok:
            restart_if_code_changed(started_with)  # 只在两张作业之间发生，不会打断正在跑的作业
        if not ok:
            hb.update(state="error", last_error="git pull 失败，下一轮重试")
        elif push and not flush_unpublished(hb):
            pass
        else:
            hb.update(state="idle", queue=len(pending_jobs(profile)))
            try:
                did = process_one(profile, me, hb, max_slots=max_slots, push=push)
            except Exception as e:  # noqa: BLE001
                log.exception("本轮异常")
                hb.update(state="error", last_error=f"{type(e).__name__}: {e}"[:300])
                did = None
            if did:
                continue  # 刚跑完一张，马上看下一张
        if once:
            return
        time.sleep(interval)


def load_status(profile: str) -> dict:
    p = paths.ROOT / "status" / f"{profile}.json"
    try:
        return json.loads(p.read_text()) or {}
    except (OSError, ValueError):
        return {}
