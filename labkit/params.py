"""参数定义：网页表单、执行配置和代码说明共用同一份定义。

每个项目在自己的 params.py 里声明 ParamSpace。网页按它渲染表单，提交时用 resolve() 校验，
训练代码读的 cfg 也是 resolve() 的输出。改一个参数一定会改变 config_hash，从而产生新运行。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# 参数影响到哪一层。前置层变了，后面的缓存和结论都要失效
AFFECTS_ORDER = ["data", "model", "train", "eval", "infer", "runtime"]


@dataclass(frozen=True)
class Param:
    name: str
    type: str  # int | float | str | bool | choice | multi
    default: Any
    help: str
    affects: str = "train"
    choices: tuple = ()
    choice_notes: dict = field(default_factory=dict)  # 某个取值的说明，教学故障在这里标注
    min: float | None = None
    max: float | None = None
    cost_note: str = ""  # 改它的代价（时间、显存、计算量）
    advanced: bool = False

    def coerce(self, value: Any) -> Any:
        t = self.type
        if t == "int":
            value = int(value)
        elif t == "float":
            value = float(value)
        elif t == "bool":
            if isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            value = bool(value)
        elif t == "str":
            value = str(value)
        elif t == "choice":
            value = str(value)
            if value not in self.choices:
                raise ValueError(f"参数 {self.name}={value!r} 不在可选值 {list(self.choices)} 中")
        elif t == "multi":
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            bad = [v for v in value if v not in self.choices]
            if bad:
                raise ValueError(f"参数 {self.name} 含未知取值 {bad}，可选 {list(self.choices)}")
            # 按定义顺序排序：同一组取值不管怎么勾选，配置都一样
            value = [c for c in self.choices if c in value]
        else:
            raise ValueError(f"未知参数类型 {t}")
        if self.min is not None and value < self.min:
            raise ValueError(f"参数 {self.name}={value} 小于下限 {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"参数 {self.name}={value} 大于上限 {self.max}")
        return value


class ParamSpace:
    def __init__(self, params: list[Param]):
        self.params = params
        self.by_name = {p.name: p for p in params}

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def resolve(self, overrides: dict | None = None) -> dict:
        """默认值 + 覆盖值 -> 完整配置。未知参数直接报错，避免"以为改了其实没生效"。"""
        overrides = overrides or {}
        unknown = set(overrides) - set(self.by_name)
        if unknown:
            raise ValueError(f"未知参数 {sorted(unknown)}")
        cfg = {}
        for p in self.params:
            cfg[p.name] = p.coerce(overrides.get(p.name, p.default))
        return cfg

    def to_json(self) -> list[dict]:
        out = []
        for p in self.params:
            d = asdict(p)
            d["choices"] = list(p.choices)
            out.append(d)
        return out


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(canonical(cfg).encode()).hexdigest()[:12]


def diff_configs(a: dict, b: dict, space: ParamSpace | None = None) -> list[dict]:
    rows = []
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            p = space.by_name.get(key) if space else None
            rows.append({
                "name": key,
                "a": a.get(key),
                "b": b.get(key),
                "affects": p.affects if p else "?",
                "help": p.help if p else "",
            })
    return rows
