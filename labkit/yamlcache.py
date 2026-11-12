"""只读 YAML 的缓存：按 (路径, 修改时间, 大小) 缓存解析结果，有 libyaml 时用 C 解析器。

场景清单和作业单在一次网页请求里会被读几十到上千次（每门课展开场景、查作业状态），纯 Python 解析器一次请求
就要十几秒。文件一变（git pull、作业单移动、改清单），修改时间或大小就变，缓存自然失效。返回深拷贝，调用方随便改。
写文件的地方（jobs.move 等）不走这里，照旧读最新内容。
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

import yaml

_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_cache: dict[str, tuple[tuple[int, int], Any]] = {}
_lock = threading.Lock()


def load(path: Path | str) -> Any:
    p = Path(path)
    st = p.stat()
    sig = (st.st_mtime_ns, st.st_size)
    key = str(p)
    with _lock:
        hit = _cache.get(key)
    if hit and hit[0] == sig:
        return copy.deepcopy(hit[1])
    data = yaml.load(p.read_text(), Loader=_Loader)  # noqa: S506  只用 SafeLoader 系列
    with _lock:
        _cache[key] = (sig, data)
    return copy.deepcopy(data)


def clear() -> None:
    with _lock:
        _cache.clear()
