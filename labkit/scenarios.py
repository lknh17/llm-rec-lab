"""场景清单：一门课要预先跑的全部运行，按档位（硬件规模）展开成具体配置。

courses/<课程目录>/scenarios.yaml 的结构：

    course: L1
    project: adextract
    entry: adextract.experiment
    profiles:                      # 档位：同一批场景在不同硬件规模下的基础参数
      mac-m4pro: {base: {...}, seeds: [0], assets: [...], slots: {gpus_per_job: 0, jobs_per_device: 1}}
      h20x8:     {base: {...}, seeds: [0, 1, 2], assets: [...], slots: {gpus_per_job: 1, jobs_per_device: 1}}
      mac-smoke: {base: {...}, seeds: [0], skip_expect: true}   # 冒烟：极小规模，只验证流程，不检查教学断言
    scenarios:
      - id: baseline
        role: baseline             # reference | baseline | variant | fault | fix | final
        title: 一句话说明
        params: {}                 # 相对档位 base 的改动
        profile_params: {h20x8: {...}}   # 只在某个档位额外改的参数（可选）
        seeds: [0]                 # 覆盖档位的种子列表（可选）
        seeds_by_profile: {mac-m4pro: [0, 1, 2]}   # 按档位覆盖种子（优先级最高，可选）
        profiles: [h20x8]          # 只在这些档位跑（可选，默认全部）
        expect:                    # 运行结束后自动检查的断言（可选）
          - "summary.dev.format_ok == 0"
          - "summary.valid.auc > base.summary.valid.auc + 0.02"   # base = 同档位同种子的 baseline 场景

运行编号是稳定的：<档位>.<课程>.<场景>.s<种子>-<配置哈希前 6 位>。配置不变，重跑就落在同一个位置；
配置变了（档位 base 或场景参数被改），编号跟着变，旧结果不会被误用。
"""

from __future__ import annotations

import ast
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths
from .params import ParamSpace, config_hash

ROLES = ("reference", "baseline", "variant", "fault", "fix", "final")


# ------------------------------------------------------------------ 课程与参数
def course_files() -> dict[str, Path]:
    out = {}
    for p in sorted(paths.COURSES_DIR.glob("*/scenarios.yaml")):
        data = yaml.safe_load(p.read_text())
        out[data["course"]] = p
    return out


def load_course(course_id: str) -> dict:
    files = course_files()
    if course_id not in files:
        raise KeyError(f"未知课程 {course_id}，可选 {list(files)}")
    data = yaml.safe_load(files[course_id].read_text())
    data["_dir"] = str(files[course_id].parent)
    validate(data)
    return data


def param_space(project: str) -> ParamSpace:
    """项目的参数定义。项目目录作为顶层包导入（例如 adextract），和 worker 从快照导入的方式一致。"""
    if str(paths.PROJECTS_DIR) not in sys.path:
        sys.path.insert(0, str(paths.PROJECTS_DIR))
    return importlib.import_module(f"{project}.params").PARAMS


def validate(course: dict) -> None:
    ids = [s["id"] for s in course["scenarios"]]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"{course['course']} 场景编号重复：{dup}")
    for s in course["scenarios"]:
        if s.get("role") not in ROLES:
            raise ValueError(f"{course['course']}/{s['id']} 的 role={s.get('role')!r} 不在 {ROLES}")
        for prof in s.get("profiles", []):
            if prof not in course["profiles"]:
                raise ValueError(f"{course['course']}/{s['id']} 引用了未定义的档位 {prof}")
        for expr in s.get("expect", []):
            check_expr(expr)
    if not any(s["role"] == "baseline" for s in course["scenarios"]):
        raise ValueError(f"{course['course']} 没有 role=baseline 的场景")


# ------------------------------------------------------------------ 展开
@dataclass
class RunSpec:
    course: str
    project: str
    entry: str
    profile: str
    scenario: str
    role: str
    title: str
    seed: int
    cfg: dict
    config_hash: str
    expect: list = field(default_factory=list)
    data_group: str = ""

    @property
    def run_id(self) -> str:
        return f"{self.profile}.{self.course}.{self.scenario}.s{self.seed}-{self.config_hash[:6]}"

    @property
    def rel_dir(self) -> Path:
        """results/ 和 batch/ 下的相对路径。"""
        return Path(self.profile) / self.course / self.scenario / f"s{self.seed}-{self.config_hash[:6]}"


def expand(course_id: str, profile: str, only: list[str] | None = None, seeds: list[int] | None = None) -> list[RunSpec]:
    course = load_course(course_id)
    if profile not in course["profiles"]:
        raise KeyError(f"{course_id} 没有档位 {profile}，可选 {list(course['profiles'])}")
    prof = course["profiles"][profile]
    space = param_space(course["project"])
    data_params = [p.name for p in space.params if p.affects == "data"]
    out = []
    for s in course["scenarios"]:
        if only and s["id"] not in only:
            continue
        if s.get("profiles") and profile not in s["profiles"]:
            continue
        for seed in (s.get("seeds_by_profile") or {}).get(profile, s.get("seeds", prof.get("seeds", [0]))):
            if seeds is not None and seed not in seeds:
                continue
            merged = dict(prof.get("base", {}))
            merged.update(s.get("params", {}))
            merged.update((s.get("profile_params") or {}).get(profile, {}))
            merged["seed"] = seed
            cfg = space.resolve(merged)
            group = config_hash({k: cfg[k] for k in data_params} | {"_project": course["project"]})
            out.append(RunSpec(course=course_id, project=course["project"], entry=course["entry"], profile=profile,
                               scenario=s["id"], role=s["role"], title=s.get("title", ""), seed=seed, cfg=cfg,
                               config_hash=config_hash(cfg), data_group=group,
                               # 冒烟档（skip_expect: true）只验证流程能跑通，规模太小，教学断言不适用
                               expect=[] if prof.get("skip_expect") else list(s.get("expect", []))))
    return out


def profile_info(course_id: str, profile: str) -> dict:
    return load_course(course_id)["profiles"][profile]


# ------------------------------------------------------------------ 期望断言
_ALLOWED = (ast.Expression, ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.BinOp,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Name, ast.Load, ast.Attribute, ast.Constant, ast.Call,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.List, ast.Tuple)
_FUNCS = {"abs": abs, "len": len, "min": min, "max": max}
_ROOTS = {"summary", "meta", "base"}


def check_expr(expr: str) -> ast.Expression:
    """只允许比较、算术、属性访问和 abs/len/min/max：断言是数据，不是任意代码。"""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"断言里不允许 {type(node).__name__}：{expr}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ValueError(f"断言里只能调用 {list(_FUNCS)}：{expr}")
        if isinstance(node, ast.Name) and node.id not in _ROOTS | set(_FUNCS) | {"True", "False", "None"}:
            raise ValueError(f"断言只能引用 {sorted(_ROOTS)}：{expr}")
        # summary.__class__ 这类写法会拿到 Python 对象本身，而不是结果字段；断言只能读数据
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"断言里不允许访问下划线开头的属性 {node.attr}：{expr}")
    return tree


class _Missing(Exception):
    pass


class _View:
    """把嵌套 dict 包成可以用 a.b.c 访问的对象；缺字段时抛 _Missing，断言记为"取不到值"。"""

    def __init__(self, d: Any, path: str):
        self._d, self._path = d, path

    def __getattr__(self, k: str) -> Any:
        if not isinstance(self._d, dict) or k not in self._d:
            raise _Missing(f"{self._path}.{k}")
        v = self._d[k]
        return _View(v, f"{self._path}.{k}") if isinstance(v, dict) else v


def uses_base(expr: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "base" for n in ast.walk(ast.parse(expr, mode="eval")))


def evaluate(expr: str, summary: dict, meta: dict, base: dict | None = None) -> dict:
    """返回 {expr, ok, error?}。ok=None 表示取不到值（字段缺失或缺少 base 运行）。"""
    tree = check_expr(expr)
    env = {"summary": _View(summary, "summary"), "meta": _View(meta, "meta"), **_FUNCS}
    if base is not None:
        env["base"] = _View(base, "base")
    try:
        ok = eval(compile(tree, "<expect>", "eval"), {"__builtins__": {}}, env)  # noqa: S307  已白名单校验
        return {"expr": expr, "ok": bool(ok)}
    except _Missing as e:
        return {"expr": expr, "ok": None, "error": f"取不到 {e}"}
    except NameError:
        return {"expr": expr, "ok": None, "error": "缺少 base（同档位同种子的 baseline 运行）"}
    except Exception as e:  # noqa: BLE001
        return {"expr": expr, "ok": None, "error": f"{type(e).__name__}: {e}"}
