"""只读 YAML 缓存（labkit.yamlcache）：文件改了要读到新内容；调用方修改返回值不能污染缓存。"""

import os

from labkit import yamlcache


def test_reload_on_change_and_isolated_copies(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("course: X\nlist: [1, 2]\n")
    a = yamlcache.load(p)
    a["list"].append(99)
    assert yamlcache.load(p) == {"course": "X", "list": [1, 2]}, "返回的是副本"
    p.write_text("course: Y\nlist: [3]\n")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # 同一毫秒内连写两次时，修改时间也要变
    assert yamlcache.load(p)["course"] == "Y"
    yamlcache.clear()
    assert yamlcache.load(p)["list"] == [3]
