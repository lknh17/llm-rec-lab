"""阶段产物缓存：只有版本与条件完全匹配时才复用。

缓存键由"影响这一阶段的全部条件"组成（数据版本、切分规则、预处理版本、模型版本……）。
条件变了键就变，旧产物自然不再被命中——这就是"参数变化让后续旧结果失效"的落地方式。
每次命中或重建都记进运行记录，网页上能看到本次运行复用了谁的产物。
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

from . import paths
from .params import canonical, config_hash
from .store import read_json, write_json


def stage_cache(ctx, stage: str, key: dict, build: Callable[[Path], None]) -> Path:
    """返回该阶段产物目录。命中则直接用，否则调用 build(tmp_dir) 生成后原子替换。"""
    digest = config_hash(key)
    root = paths.sub("cache") / stage
    final = root / digest
    meta_path = final / "_cache_meta.json"
    if meta_path.exists():
        meta = read_json(meta_path, {})
        ctx.note_cache(stage, digest, key, hit=True, source_run=meta.get("created_by_run"))
        return final
    tmp = root / f".{digest}.tmp-{ctx.run_id}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    build(tmp)
    write_json(tmp / "_cache_meta.json", {
        "stage": stage, "key": key, "key_canonical": canonical(key),
        "created_by_run": ctx.run_id, "created_at": time.time(),
    })
    if final.exists():  # 并发情况下别人先建好了，用别人的
        shutil.rmtree(tmp)
    else:
        try:
            tmp.rename(final)
        except OSError:
            # 检查和改名之间别的进程抢先建好了（rename 到非空目录会报 ENOTEMPTY / EEXIST）：用别人的
            if not (final / "_cache_meta.json").exists():
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    ctx.note_cache(stage, digest, key, hit=False, source_run=ctx.run_id)
    return final
