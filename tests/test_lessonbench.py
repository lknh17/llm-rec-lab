"""lessonbench（V01）：每个验证项的入口都在 vendor/ 里、环境缺失的判定、执行与报告收集（用当前解释器跑一个假入口）、场景清单。"""

import json
import sys

import pytest

from labkit import scenarios
from lessonbench import experiment as X
from lessonbench.items import ITEMS
from lessonbench.params import PARAMS


def test_every_item_entry_is_vendored():
    for name, item in ITEMS.items():
        for _kind, args in item["steps"]:
            if args[0] == "-m":
                mod = args[1]
                if mod == "pytest":
                    assert (X.VENDOR / args[-1]).is_dir(), name
                    continue
                path = X.VENDOR.joinpath(*mod.split("."))
                assert path.with_suffix(".py").exists(), f"{name}: vendor 里没有 {mod}（先跑私有仓库 tools/export_lessonbench.py）"
            elif args[0].startswith("@"):
                assert (X.HERE / args[0][1:]).exists(), name
    assert (X.VENDOR / "stages/s03_kernels/cuda/matmul_kernels.cu").exists()


def test_missing():
    env = {"cuda": True, "nvcc": "/usr/local/cuda/bin/nvcc", "ncu": None, "vllm_python": None}
    assert X.missing(ITEMS["cuda_matmul"]["needs"], env) == []
    assert X.missing(ITEMS["ncu"]["needs"], env) == ["ncu"]
    assert X.missing(ITEMS["s7_vllm"]["needs"], env) == ["vllm"]
    assert X.missing(ITEMS["ncu_parse"]["needs"], {}) == []


class Ctx:
    def __init__(self, run_dir):
        self.run_dir, self.logs, self.tr, self.s, self.w = run_dir, [], {}, {}, []
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    def stage(self, s): pass
    def log(self, m): self.logs.append(m)
    def check(self): pass
    def info(self, k, d): pass
    def warn(self, c, m): self.w.append(c)
    def artifact(self, n): return self.run_dir / "artifacts" / n
    def trace(self, k, p): self.tr[k] = json.loads(json.dumps(p))
    def summary(self, d): self.s.update(d)


def _fake(monkeypatch, tmp_path, script: str, needs=()):
    vendor = tmp_path / "vendor"
    (vendor / "pkg").mkdir(parents=True)
    (vendor / "pkg" / "job.py").write_text(script)
    (vendor / "old.md").write_text("讲义里本来就有的报告")
    monkeypatch.setattr(X, "VENDOR", vendor)
    monkeypatch.setitem(ITEMS, "fake", {"title": "假", "lesson": "S0", "steps": [("main", ["-m", "pkg.job", "--hw", "{hw}"])],
                                        "needs": list(needs)})
    monkeypatch.setattr(X, "probe", lambda: {"cuda": False, "uv": None})        # uv=None：用当前解释器
    return Ctx(tmp_path / "run")


def test_run_collects_new_reports(monkeypatch, tmp_path):
    ctx = _fake(monkeypatch, tmp_path, "import sys, pathlib\nprint('hw', sys.argv[-1])\n"
                "pathlib.Path('stages/x/experiments').mkdir(parents=True)\n"
                "pathlib.Path('stages/x/experiments/r.md').write_text('# 报告')\npathlib.Path('fig.png').write_bytes(b'0' * 10)\n")
    X.run(ctx, {"item": "fake", "hw": "H20"})
    s = ctx.s
    assert s["ok"] and s["outputs"] == ["stages/x/experiments/r.md"] and s["binary_outputs"][0].startswith("fig.png")
    assert ctx.tr["outputs"]["files"]["stages/x/experiments/r.md"] == "# 报告" and "hw H20" in ctx.logs
    assert s["steps"][0]["cmd"] == "[main] python -m pkg.job --hw H20" and s["steps"][0]["rc"] == 0


def test_failed_step_and_missing_env(monkeypatch, tmp_path):
    ctx = _fake(monkeypatch, tmp_path, "raise SystemExit('nvcc fatal: 找不到 cuda_runtime.h')\n")
    with pytest.raises(RuntimeError, match="cuda_runtime"):
        X.run(ctx, {"item": "fake", "hw": "H20"})
    assert ctx.s["ok"] is False and "step_failed" in ctx.w
    ctx2 = _fake(monkeypatch, tmp_path / "b", "print(1)\n", needs=["nvcc"])
    X.run(ctx2, {"item": "fake", "hw": "H20"})
    assert ctx2.s["ok"] is False and ctx2.s["missing"] == ["nvcc"] and ctx2.s["steps"] == [] and "env_missing" in ctx2.w


def test_interpreter_choice(monkeypatch):
    env = {"uv": "/bin/uv", "vllm_python": "/venvs/vllm/bin/python"}
    assert X.interpreter("vllm", env) == ["/venvs/vllm/bin/python"]
    main = X.interpreter("main", env)
    assert main[:2] == ["/bin/uv", "run"] and "--no-sync" in main and main.count("--with") == 3 and main[-1] == "python"
    assert X.interpreter("main", {"uv": None}) == [sys.executable]


def test_scenarios_v01():
    h = scenarios.expand("V01", "h20x8")
    assert {s.scenario for s in h} == set(ITEMS)
    assert all(s.cfg["hw"] == "H20" for s in h)
    assert {s.scenario for s in scenarios.expand("V01", "mac-smoke")} == {"env_probe", "ncu_parse"}
    PARAMS.resolve({"item": "ncu"})
    for s in scenarios.load_course("V01")["scenarios"]:
        for e in s.get("expect", []):
            scenarios.check_expr(e)
