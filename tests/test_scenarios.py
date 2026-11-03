"""场景清单：展开、哈希稳定、断言白名单与求值。"""

import pytest

from labkit import scenarios
from labkit.params import config_hash

COURSES = ["R1", "R2", "L1", "L2"]


@pytest.mark.parametrize("course", COURSES)
def test_course_loads_and_expands(course):
    c = scenarios.load_course(course)
    assert set(c["profiles"]) >= {"mac-m4pro", "h20x8"}
    for prof in c["profiles"]:
        specs = scenarios.expand(course, prof)
        assert specs, f"{course}/{prof} 没有运行"
        ids = [s.run_id for s in specs]
        assert len(ids) == len(set(ids)), "同一档位里运行编号重复"
        for s in specs:
            assert s.config_hash == config_hash(s.cfg)
            assert s.rel_dir.parts[:3] == (prof, course, s.scenario)


def test_expand_is_stable():
    a = scenarios.expand("R1", "mac-m4pro")
    b = scenarios.expand("R1", "mac-m4pro")
    assert [s.run_id for s in a] == [s.run_id for s in b]


def test_profile_base_and_scenario_params_merge():
    spec = next(s for s in scenarios.expand("L1", "h20x8") if s.scenario == "fault_truncation")
    assert spec.cfg["model"] == "Qwen/Qwen2.5-7B-Instruct"   # 档位 base
    assert spec.cfg["max_len"] == 256                          # 场景参数
    assert spec.seed in (0, 1, 2)


def test_profile_only_scenarios_are_filtered():
    mac = {s.scenario for s in scenarios.expand("R1", "mac-m4pro")}
    h20 = {s.scenario for s in scenarios.expand("R1", "h20x8")}
    assert "ml20m_baseline" not in mac and "ml20m_baseline" in h20


def test_same_config_same_hash_across_courses():
    """L1 和 L2 的基线是同一个配置：批量运行据此只跑一次。"""
    l1 = next(s for s in scenarios.expand("L1", "mac-m4pro") if s.scenario == "baseline" and s.seed == 0)
    l2 = next(s for s in scenarios.expand("L2", "mac-m4pro") if s.scenario == "baseline" and s.seed == 0)
    assert l1.config_hash == l2.config_hash


def test_data_group_ignores_train_params():
    specs = {s.scenario: s for s in scenarios.expand("R1", "mac-m4pro") if s.seed == 0}
    assert specs["baseline"].data_group == specs["lr3e-3"].data_group
    assert specs["baseline"].data_group != specs["fault_random_split"].data_group


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
