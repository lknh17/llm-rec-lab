"""H20 常驻 agent：抢占锁（两个 clone 抢同一张作业单，只有一个成功）、作业单校验、作废、心跳白名单；
以及证据摘要 evidence.json。git 操作全部在临时 bare 仓库里进行，不碰真实远端。"""

import json
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from labkit import agent, evidence, jobs, paths

ROOT = Path(__file__).resolve().parent.parent


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _use_clone(monkeypatch, clone: Path) -> None:
    monkeypatch.setattr(paths, "ROOT", clone)
    monkeypatch.setattr(paths, "JOBS_DIR", clone / "jobs")
    monkeypatch.setattr(paths, "RESULTS_DIR", clone / "results")


@pytest.fixture
def two_clones(tmp_path):
    bare = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(bare), str(seed))
    for c in (seed,):
        _git(c, "config", "user.name", "t")
        _git(c, "config", "user.email", "t@example.com")
    (seed / "jobs" / "pending").mkdir(parents=True)
    job = {"job_id": "j1", "created_at": "2026-01-01T00:00:00Z", "profile": "h20x8", "courses": ["L1"],
           "only": ["baseline"], "seeds": [0], "requested_commit": "", "note": ""}
    (seed / "jobs" / "pending" / "j1.yaml").write_text(yaml.safe_dump(job, allow_unicode=True))
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "init")
    _git(seed, "push", "-u", "origin", "main")
    clones = []
    for name in ("a", "b"):
        c = tmp_path / name
        _git(tmp_path, "clone", str(bare), str(c))
        _git(c, "config", "user.name", name)
        _git(c, "config", "user.email", f"{name}@example.com")
        clones.append(c)
    return bare, clones


def test_claim_is_exclusive(two_clones, monkeypatch):
    _bare, (a, b) = two_clones
    _use_clone(monkeypatch, a)
    assert agent.claim("j1", "agent-a") is True
    _use_clone(monkeypatch, b)  # b 还没 pull，本地仍看到 pending/j1
    assert (b / "jobs" / "pending" / "j1.yaml").exists()
    assert agent.claim("j1", "agent-b") is False
    # b 放弃后同步到了远端状态：作业在 running，归 agent-a
    data = yaml.safe_load((b / "jobs" / "running" / "j1.yaml").read_text())
    assert data["claimed_by"] == "agent-a"
    assert not (b / "jobs" / "pending" / "j1.yaml").exists()
    assert _git(b, "status", "--porcelain") == ""


def test_claim_retries_when_remote_moved_unrelated(two_clones, monkeypatch):
    """远端有无关的新提交（Mac 推了另一张作业单）时，rebase 没有冲突，重推成功。"""
    _bare, (a, b) = two_clones
    (b / "jobs" / "pending" / "j2.yaml").write_text("job_id: j2\nprofile: h20x8\ncourses: [L1]\n")
    _git(b, "add", "-A")
    _git(b, "commit", "-m", "j2")
    _git(b, "push")
    _use_clone(monkeypatch, a)
    assert agent.claim("j1", "agent-a") is True
    assert _git(a, "log", "-1", "--format=%s").startswith("jobs: agent-a 领取 j1")
    assert _git(a, "status", "-sb").startswith("## main...origin/main") and "ahead" not in _git(a, "status", "-sb")


def test_validate_job_rejects_unknown_fields_and_scenarios(monkeypatch):
    monkeypatch.setattr(agent, "git", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    ok = {"job_id": "x", "profile": "h20x8", "courses": ["L1"], "only": ["baseline"], "seeds": [0]}
    assert agent.validate_job(ok) == []
    bad = dict(ok, only=["no_such"], command="rm -rf /", seeds=["0"])
    errs = agent.validate_job(bad)
    assert any("不认识的字段" in e for e in errs)
    assert any("no_such" in e for e in errs)
    assert any("seeds" in e for e in errs)
    assert any("档位" in e for e in agent.validate_job(dict(ok, profile="nope")))


def test_cancel_job(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "JOBS_DIR", tmp_path / "jobs")
    jobs.new_job(["L1"], "h20x8", job_id="c1")
    p = jobs.cancel("c1", "旧作业单作废")
    assert p.parent.name == "cancelled"
    d = jobs.load("c1")
    assert d["_state"] == "cancelled" and d["cancel_reason"] == "旧作业单作废"
    with pytest.raises(ValueError):
        jobs.move("c1", "running")
    with pytest.raises(ValueError):
        jobs.run_job("c1")


def test_heartbeat_whitelist_and_throttle(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "host_summary", lambda: {"gpu": "NVIDIA H20", "gpu_count": 8, "gpu_mem_gb": 95.6,
                                                       "driver": "x", "hostname": "secret-host"})
    hb = agent.Heartbeat("h20x8", "agent-x", push=False)
    hb.update(state="idle", secret="no")
    data = json.loads((tmp_path / "status" / "h20x8.json").read_text())
    assert set(data) <= agent.STATUS_KEYS
    assert "secret-host" not in json.dumps(data) and data["gpu_count"] == 8
    first = data["updated_at"]
    hb.update(state="idle")  # 状态没变、没到 15 分钟：不重写
    assert json.loads((tmp_path / "status" / "h20x8.json").read_text())["updated_at"] == first
    hb.update(state="running", job_id="j1")
    data = json.loads((tmp_path / "status" / "h20x8.json").read_text())
    assert data["state"] == "running" and data["beat_every_s"] == agent.HEARTBEAT_EVERY
    # 空闲：间隔是 3 小时，16 分钟后不推；运行中 16 分钟后要推
    clock = [time.time()]
    monkeypatch.setattr(agent.time, "time", lambda: clock[0])
    hb.update(state="idle", job_id=None)
    assert json.loads((tmp_path / "status" / "h20x8.json").read_text())["beat_every_s"] == agent.HEARTBEAT_IDLE_EVERY
    clock[0] += 16 * 60
    assert hb.update(state="idle") is False and json.loads((tmp_path / "status" / "h20x8.json").read_text())["updated_at"] != clock[0]
    hb.update(state="running", job_id="j2")
    clock[0] += 16 * 60
    hb.update(state="running", progress="3/34")
    assert json.loads((tmp_path / "status" / "h20x8.json").read_text())["progress"] == "3/34"


def test_evidence_from_existing_result():
    runs = sorted((ROOT / "results" / "mac-m4pro" / "L1" / "baseline").glob("*/run.json"))
    if not runs:
        pytest.skip("没有 Mac 档 L1 结果")
    ev = evidence.build(runs[0].parent)
    assert ev["params"]["trainable"] > 0 and ev["params"]["total"] > ev["params"]["trainable"]
    assert 0 < len(ev["curves"]["train_loss"]) <= evidence.MAX_POINTS
    assert ev["throughput"]["tokens_per_s_median"] > 0
    assert ev["memory"]["breakdown"]["weights_gb"] > 0
    assert ev["memory"]["peak_allocated_gb"] is None  # Mac 档没有 CUDA 峰值
    assert len(json.dumps(ev).encode()) <= evidence.MAX_BYTES


def test_evidence_memory_breakdown_with_peak():
    meta = {"model": {"dtype": "torch.bfloat16", "train_mode": "lora",
                      "params": {"total": 7_000_000_000, "trainable": 40_000_000}}}
    mem = evidence.memory(meta, {"device": "cuda", "peak_allocated_bytes": 30 * 1024 ** 3})
    b = mem["breakdown"]
    assert abs(b["weights_gb"] - (7e9 - 4e7) * 2 / 1024 ** 3) < 0.01
    assert b["optimizer_gb"] == pytest.approx(4e7 * 8 / 1024 ** 3, abs=1e-3)
    assert b["activations_other_gb"] == pytest.approx(30 - sum(v for k, v in b.items() if k != "activations_other_gb"),
                                                      abs=0.02)


def test_figures_deterministic_and_valid_svg():
    import xml.etree.ElementTree as ET

    from labkit import docs, figures, scenarios

    course = scenarios.load_course("L1")
    got = docs.collect("L1", "mac-m4pro")
    if not got:
        pytest.skip("没有 Mac 档 L1 结果")
    a = figures.course_figures(course, got, "mac-m4pro")
    b = figures.course_figures(course, got, "mac-m4pro")
    assert a == b and {"metrics", "loss"} <= set(a)
    for svg in a.values():
        ET.fromstring(svg)  # 合法 XML，GitHub 能直接显示
        assert "<script" not in svg
    assert figures._nice(0.99) == 1 and figures._nice(3.7) == 5 and figures._nice(0.21) == 0.25


def test_docs_outputs_and_smoke_profile_skipped():
    from labkit import docs, scenarios

    outs = docs.outputs()
    assert "courses/L1_lora_extraction/README.md" in outs
    table = docs.results_table(scenarios.load_course("L1"))
    assert "mac-smoke" not in table
    specs = scenarios.expand("L1", "mac-smoke")
    assert specs and all(s.expect == [] for s in specs), "冒烟档不检查教学断言"


def test_check_is_read_only(two_clones, monkeypatch, capsys):
    """agent --check：列出并校验待领取作业、预览运行数，但不领取、不改文件、不提交。"""
    _bare, (a, _b) = two_clones
    _use_clone(monkeypatch, a)
    head = _git(a, "rev-parse", "HEAD")
    assert agent.check("h20x8") == 0
    out = capsys.readouterr().out
    assert "j1：校验通过" in out and "共 1 个运行" in out and "没有领取作业" in out
    assert (a / "jobs" / "pending" / "j1.yaml").exists()
    assert _git(a, "rev-parse", "HEAD") == head and _git(a, "status", "--porcelain") == ""


def test_restart_when_labkit_code_changes(monkeypatch):
    """pull 到新的 labkit 代码后，agent 在两张作业之间 execv 重启自己；代码没变就不重启。"""
    calls = []
    monkeypatch.setattr(agent.os, "execv", lambda exe, argv: calls.append(argv))
    monkeypatch.setattr(agent, "code_version", lambda: "tree-new")
    agent.restart_if_code_changed("tree-new")
    assert calls == []
    agent.restart_if_code_changed("tree-old")
    assert calls and calls[0][1:3] == ["-m", "labkit"]
