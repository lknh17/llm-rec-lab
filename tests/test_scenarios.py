"""场景清单：展开、哈希稳定、断言白名单与求值。"""

import pytest

from labkit import scenarios
from labkit.params import config_hash

COURSES = list(scenarios.course_files())


@pytest.mark.parametrize("course", COURSES)
def test_course_loads_and_expands(course):
    c = scenarios.load_course(course)
    assert set(c["profiles"]) >= {"h20x8", "mac-smoke"}
    for prof in c["profiles"]:
        specs = scenarios.expand(course, prof)
        assert specs, f"{course}/{prof} 没有运行"
        ids = [s.run_id for s in specs]
        assert len(ids) == len(set(ids)), "同一档位里运行编号重复"
        for s in specs:
            assert s.config_hash == config_hash(s.cfg)
            assert s.rel_dir.parts[:3] == (prof, course, s.scenario)


def test_published_courses_have_lessons():
    """每门公开课都有场景清单和公开讲义（lesson.yaml），L05（课程编号 L1）一定在。"""
    files = scenarios.course_files()
    assert "L1" in files
    for course, path in files.items():
        assert (path.parent / "lesson.yaml").exists(), f"{course} 缺 lesson.yaml"


def test_expand_is_stable():
    a = scenarios.expand("L1", "h20x8")
    b = scenarios.expand("L1", "h20x8")
    assert [s.run_id for s in a] == [s.run_id for s in b]


def test_profile_base_and_scenario_params_merge():
    spec = next(s for s in scenarios.expand("L1", "h20x8") if s.scenario == "fault_truncation")
    assert spec.cfg["model"] == "Qwen/Qwen2.5-7B-Instruct"   # 档位 base
    assert spec.cfg["max_len"] == 256                          # 场景参数
    assert spec.seed in (0, 1, 2)


def test_profile_only_scenarios_are_filtered():
    mac = {s.scenario for s in scenarios.expand("L1", "mac-m4pro")}
    h20 = {s.scenario for s in scenarios.expand("L1", "h20x8")}
    assert "lora_r64" not in mac and "lora_r64" in h20


def test_h20_job_size():
    """H20 档：base_model 1 个种子，其余 11 个场景各 3 个种子，共 34 个运行。"""
    specs = scenarios.expand("L1", "h20x8")
    assert len(specs) == 34
    assert {s.seed for s in specs if s.scenario == "baseline"} == {0, 1, 2}


def test_data_group_ignores_train_params():
    specs = {s.scenario: s for s in scenarios.expand("L1", "h20x8") if s.seed == 0}
    assert specs["baseline"].data_group == specs["lr_1e-3"].data_group == specs["lora_r64"].data_group
    assert specs["baseline"].data_group != specs["fault_truncation"].data_group


@pytest.mark.parametrize("bad", [
    "__import__('os').system('x')",
    "summary.__class__",
    "summary._d",
    "summary.__class__.__mro__",
    "open('/etc/passwd')",
    "[x for x in summary]",
    "lambda: 1",
    "os.getcwd()",
])
def test_expect_rejects_code(bad):
    with pytest.raises((ValueError, SyntaxError)):
        scenarios.check_expr(bad)


def test_expect_evaluate():
    summary = {"valid": {"auc": 0.79}, "dev": {"format_ok": 0.0}}
    meta = {"checks": {"data": {"train_has_future": True}}}
    base = {"summary": {"valid": {"auc": 0.75}}, "meta": {}}
    assert scenarios.evaluate("summary.valid.auc > base.summary.valid.auc + 0.02", summary, meta, base)["ok"] is True
    assert scenarios.evaluate("meta.checks.data.train_has_future == True", summary, meta)["ok"] is True
    assert scenarios.evaluate("summary.dev.format_ok == 0", summary, meta)["ok"] is True
    missing = scenarios.evaluate("summary.test.auc > 0", summary, meta)
    assert missing["ok"] is None and "summary.test" in missing["error"]
    no_base = scenarios.evaluate("summary.valid.auc > base.summary.valid.auc", summary, meta, None)
    assert no_base["ok"] is None
