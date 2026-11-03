"""作业单状态流转；批量调度（CPU 小规模合成数据）的续跑、断言和同配置复用。"""

import json

import pytest
import yaml

from labkit import batch, jobs, paths, scenarios


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "JOBS_DIR", tmp_path / "jobs")
    return tmp_path / "jobs"


def test_job_lifecycle(jobs_dir):
    p = jobs.new_job(["R1"], "h20x8", only=["baseline"], seeds=[0], job_id="t1")
    assert p.parent.name == "pending"
    with pytest.raises(FileExistsError):
        jobs.new_job(["R1"], "h20x8", job_id="t1")
    jobs.move("t1", "running", started_at="now")
    assert jobs.load("t1")["_state"] == "running"
    with pytest.raises(ValueError):
        jobs.move("t1", "pending")
    jobs.move("t1", "done", status="done", runs=[])
    d = yaml.safe_load((jobs_dir / "done" / "t1.yaml").read_text())
    assert d["status"] == "done" and d["started_at"] == "now" and d["only"] == ["baseline"]
    assert not (jobs_dir / "running" / "t1.yaml").exists()


# ---------------------------------------------------------------- 端到端：小规模合成数据
SMOKE = {
    "course": "ZT",
    "title": "测试课程",
    "project": "rec",
    "entry": "rec.experiment",
    "prepare_first": True,
    "profiles": {"cpu": {"base": {"dataset": "synth", "synth_users": 300, "synth_items": 200, "synth_rows": 20000,
                                  "epochs": 1, "batch_size": 512, "device": "cpu", "hidden": "16"},
                         "seeds": [0], "slots": {"gpus_per_job": 0, "jobs_per_device": 2}}},
    "scenarios": [
        {"id": "baseline", "role": "baseline", "params": {},
         "expect": ["summary.valid.auc < meta.checks.data.oracle_auc_valid", "meta.data.rows > 1000"]},
        {"id": "leak", "role": "fault", "params": {"feature_groups": ["ids", "user_stats"], "stats_mode": "full_data"},
         "expect": ["meta.checks.data.user_stats_rows_at_or_after_sample_share > 0.5"]},
        {"id": "impossible", "role": "variant", "params": {"lr": 0.002}, "expect": ["summary.valid.auc > 0.999"]},
    ],
}


@pytest.fixture
def smoke_course(tmp_path, monkeypatch):
    courses = tmp_path / "courses"
    (courses / "ZT").mkdir(parents=True)
    (courses / "ZT" / "scenarios.yaml").write_text(yaml.safe_dump(SMOKE, allow_unicode=True))
    monkeypatch.setattr(paths, "COURSES_DIR", courses)
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    return courses


def test_batch_end_to_end(smoke_course):
    report = batch.run_batch(["ZT"], "cpu")
    by = {r["scenario"]: r for r in report}
    assert by["baseline"]["status"] == "succeeded" and by["baseline"]["expect_ok"] is True
    assert by["leak"]["expect_ok"] is True
    assert by["impossible"]["status"] == "succeeded" and by["impossible"]["expect_ok"] is False
    spec = scenarios.expand("ZT", "cpu", ["baseline"])[0]
    meta = json.loads((paths.batch_dir() / spec.rel_dir / "run.json").read_text())
    assert meta["scenario"] == "baseline" and meta["code_hash"] and meta["config_hash"] == spec.config_hash
    assert meta["expect_results"][0]["ok"] is True

    # 续跑：第二次全部跳过
    again = batch.run_batch(["ZT"], "cpu")
    assert all(r["status"] == "skipped" for r in again)
