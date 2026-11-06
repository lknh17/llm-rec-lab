"""作业单状态流转；批量调度（测试用玩具项目，CPU 上秒级）的续跑、断言和同配置复用。"""

import json
from pathlib import Path

import pytest
import yaml

from labkit import batch, jobs, paths, scenarios


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "JOBS_DIR", tmp_path / "jobs")
    return tmp_path / "jobs"


def test_job_lifecycle(jobs_dir):
    p = jobs.new_job(["L1"], "h20x8", only=["baseline"], seeds=[0], job_id="t1")
    assert p.parent.name == "pending"
    with pytest.raises(FileExistsError):
        jobs.new_job(["L1"], "h20x8", job_id="t1")
    jobs.move("t1", "running", started_at="now")
    assert jobs.load("t1")["_state"] == "running"
    with pytest.raises(ValueError):
        jobs.move("t1", "pending")
    jobs.move("t1", "done", status="done", runs=[])
    d = yaml.safe_load((jobs_dir / "done" / "t1.yaml").read_text())
    assert d["status"] == "done" and d["started_at"] == "now" and d["only"] == ["baseline"]
    assert not (jobs_dir / "running" / "t1.yaml").exists()


# ---------------------------------------------------------------- 端到端：测试用玩具项目（tests/fixtures/projects/toy）
FIXTURE_PROJECTS = Path(__file__).parent / "fixtures" / "projects"
SMOKE = {
    "course": "ZT",
    "title": "测试课程",
    "project": "toy",
    "entry": "toy.experiment",
    "prepare_first": True,
    "profiles": {"cpu": {"base": {"device": "cpu"}, "seeds": [0],
                         "slots": {"gpus_per_job": 0, "jobs_per_device": 2}}},
    "scenarios": [
        {"id": "baseline", "role": "baseline", "params": {},
         "expect": ["summary.valid.acc > 0.7", "meta.data.rows > 1000"]},
        {"id": "leak", "role": "fault", "params": {"leak": True},
         "expect": ["meta.checks.data.label_in_features == True", "summary.valid.acc > base.summary.valid.acc"]},
        {"id": "impossible", "role": "variant", "params": {"lr": 0.2}, "expect": ["summary.valid.acc > 0.999"]},
    ],
}


@pytest.fixture
def smoke_course(tmp_path, monkeypatch):
    courses = tmp_path / "courses"
    (courses / "ZT").mkdir(parents=True)
    (courses / "ZT" / "scenarios.yaml").write_text(yaml.safe_dump(SMOKE, allow_unicode=True))
    monkeypatch.setattr(paths, "COURSES_DIR", courses)
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(paths, "PROJECTS_DIR", FIXTURE_PROJECTS)
    monkeypatch.syspath_prepend(str(FIXTURE_PROJECTS))
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


def test_run_job_end_to_end_writes_valid_yaml(smoke_course, jobs_dir):
    """作业单完整走一遍 run_job：领取 → 批量运行 → 导出 → 回报。
    回归：host_summary 里的 torch.__version__ 是 TorchVersion（str 子类），曾让 yaml.safe_dump 报 RepresenterError，
    H20 上 34 个运行一个都没跑。CI 和 Mac 都装了 torch，这条测试会走到同一段代码。"""
    jobs.new_job(["ZT"], "cpu", job_id="e2e")
    res = jobs.run_job("e2e", publish_each=False)
    assert res["status"] == "partial" and res["runs"] == 3 and res["problems"] == 1  # impossible 的断言故意不通过
    d = yaml.safe_load((jobs_dir / "done" / "e2e.yaml").read_text())
    assert d["status"] == "partial" and len(d["runs"]) == 3
    hs = d["host_summary"]
    assert type(hs.get("torch", "")) is str and "hostname" not in hs
    assert {r["scenario"]: r["expect_ok"] for r in d["runs"]} == {"baseline": True, "leak": True, "impossible": False}


def test_job_yaml_accepts_str_subclasses_and_numpy(jobs_dir):
    """写作业单前统一转成纯 Python 类型：str 子类、numpy 标量、Path、tuple 都不能让 safe_dump 失败。"""
    import numpy as np

    class FakeVersion(str):  # 和 torch.torch_version.TorchVersion 一样是 str 的子类
        pass

    jobs.new_job(["L1"], "h20x8", job_id="t2")
    jobs.move("t2", "running", host_summary={"torch": FakeVersion("2.15.0+cu132"), "gpu_count": np.int64(8),
                                            "gpu_mem_gb": np.float32(95.6)},
              runs=[{"dir": Path("h20x8/L1/x"), "seeds": (0, 1)}])
    d = yaml.safe_load((jobs_dir / "running" / "t2.yaml").read_text())
    assert d["host_summary"] == {"torch": "2.15.0+cu132", "gpu_count": 8, "gpu_mem_gb": pytest.approx(95.6, rel=1e-6)}
    assert d["runs"] == [{"dir": "h20x8/L1/x", "seeds": [0, 1]}]
