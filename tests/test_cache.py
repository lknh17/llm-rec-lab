"""阶段缓存：命中、重建，以及并发时"检查之后、改名之前被别人抢先建好"的竞态（H20 上 L01 fault_nodedup s1 因此失败）。"""

import json
from pathlib import Path

from labkit import cache


class Ctx:
    run_id = "r1"

    def __init__(self):
        self.notes = []

    def note_cache(self, stage, digest, key, hit, source_run):
        self.notes.append((stage, hit, source_run))


def test_build_then_hit():
    ctx = Ctx()
    d1 = cache.stage_cache(ctx, "t", {"a": 1}, lambda d: (d / "x.txt").write_text("1"))
    d2 = cache.stage_cache(ctx, "t", {"a": 1}, lambda d: (_ for _ in ()).throw(AssertionError("不该重建")))
    assert d1 == d2 and (d1 / "x.txt").read_text() == "1"
    assert [n[1] for n in ctx.notes] == [False, True]


def test_concurrent_winner_between_check_and_rename(monkeypatch):
    """别的进程在 final.exists() 检查之后、本进程 rename 之前把同一份缓存建好了：用别人的，不报错。"""
    ctx = Ctx()
    real_rename = Path.rename

    def racing_rename(self, target):
        target = Path(target)
        if not target.exists():                       # 模拟另一个进程抢先完成
            target.mkdir(parents=True)
            (target / "x.txt").write_text("winner")
            (target / "_cache_meta.json").write_text(json.dumps({"created_by_run": "r0"}))
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", racing_rename)
    d = cache.stage_cache(ctx, "t", {"a": 2}, lambda d: (d / "x.txt").write_text("loser"))
    assert (d / "x.txt").read_text() == "winner"
    assert not any(p.name.startswith(".") for p in d.parent.iterdir()), "临时目录要清理掉"
