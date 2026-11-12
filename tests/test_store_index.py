"""运行目录索引（labkit.store）：查不到的运行不能每次都触发全量扫描，刚写出的运行仍要很快可见。"""

import json
import time

from labkit import store


def _make_run(root, rel: str, run_id: str) -> None:
    d = root / rel
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"run_id": run_id, "status": "succeeded"}))


def test_misses_do_not_rescan_each_time(tmp_path, monkeypatch):
    root = tmp_path / "results"
    for i in range(30):
        _make_run(root, f"h20x8/X/s{i}/s0-abc{i:03d}", f"h20x8.X.s{i}.s0-abc{i:03d}")
    monkeypatch.setattr(store, "ROOT_PROVIDERS", [lambda: [(root, "pregenerated")]])
    monkeypatch.setitem(store._INDEX, "t", 0.0)
    scans = []
    real = store._run_dirs
    monkeypatch.setattr(store, "_run_dirs", lambda r: (scans.append(r), real(r))[1])
    assert store.get_run("h20x8.X.s3.s0-abc003") is not None
    first = len(scans)
    for i in range(50):                                   # 一门还没结果的课：几十个运行都查不到
        assert store.get_run(f"h20x8.Y.missing.s{i}-zzz") is None
    assert len(scans) == first, "同一批未命中不该再全量扫描"


def test_new_run_visible_after_short_delay(tmp_path, monkeypatch):
    root = tmp_path / "results"
    _make_run(root, "h20x8/X/a/s0-aaa", "h20x8.X.a.s0-aaa")
    monkeypatch.setattr(store, "ROOT_PROVIDERS", [lambda: [(root, "pregenerated")]])
    monkeypatch.setitem(store._INDEX, "t", 0.0)
    assert store.get_run("h20x8.X.b.s0-bbb") is None
    _make_run(root, "h20x8/X/b/s0-bbb", "h20x8.X.b.s0-bbb")  # 刚写出的运行
    time.sleep(store.MISS_RESCAN_S + 0.05)
    assert store.get_run("h20x8.X.b.s0-bbb") is not None
    store._index(max_age=0)                                 # lab sync 之后主动重建：立刻可见
    _make_run(root, "h20x8/X/c/s0-ccc", "h20x8.X.c.s0-ccc")
    store._index(max_age=0)
    assert store.get_run("h20x8.X.c.s0-ccc") is not None
