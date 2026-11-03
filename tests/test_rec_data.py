"""推荐项目的数据口径：时间切分没有未来，时点正确的统计特征看不到自己和未来。"""

import numpy as np
import pandas as pd

from rec.data import assign_split, cnt_bucket, history_stats


def toy():
    # 用户 1：t=1,2,2,3；用户 2：t=5
    return pd.DataFrame({"user_id": [1, 1, 1, 1, 2], "timestamp": [1, 2, 2, 3, 5], "label": [1, 0, 1, 1, 0]})


def test_time_split_has_no_future():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"timestamp": rng.integers(0, 10_000, 5000)})
    s = assign_split(df, "time", 0)
    ts = df["timestamp"].to_numpy()
    assert ts[s == 0].max() < ts[s == 1].min()
    assert ts[s == 1].max() < ts[s == 2].min()


def test_random_split_mixes_time():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"timestamp": rng.integers(0, 10_000, 5000)})
    s = assign_split(df, "random", 0)
    ts = df["timestamp"].to_numpy()
    assert ts[s == 0].max() > ts[s == 1].min()  # 这正是随机切分的问题


def test_point_in_time_stats():
    cnt, rate = history_stats(toy(), "user_id", prior=0.5, mode="point_in_time")
    # 同一秒（t=2）的两条互相看不见；t=3 看得见前 3 条
    assert cnt.tolist() == [0, 1, 1, 3, 0]
    assert rate[0] == np.float32(0.5)  # 没有历史 -> 先验


def test_full_data_stats_leak_own_label():
    cnt, _ = history_stats(toy(), "user_id", prior=0.5, mode="full_data")
    assert cnt.tolist() == [4, 4, 4, 4, 1]  # 包含自己和未来


def test_cnt_bucket():
    assert cnt_bucket(np.array([0, 1, 2, 3, 4, 171])).tolist()[:5] == [0, 1, 2, 2, 3]
