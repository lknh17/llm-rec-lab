"""recbase（R0a / R0b）：CSV 解析、AUC / GAUC 对拍、三种切分、训练集词表、四种统计口径、分组、三种结构、端到端。
用合成的小 MovieLens（格式和 HF 镜像相同：timestamp 带引号和换行），不依赖真实数据。"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from labkit import scenarios
from recbase import data as D
from recbase import experiment as E
from recbase import features as F
from recbase import metrics as M
from recbase.model import RecModel, param_counts
from recbase.params import PARAMS

GENRES = ["Action", "Comedy", "Drama", "Children's", "Sci-Fi"]


def write_synth(d: Path, n_users=60, n_movies=40, n_rows=2400, seed=0) -> list[str]:
    rng = np.random.default_rng(seed)
    ub, mb = rng.normal(0, 1, n_users), rng.normal(0, 1, n_movies)
    u = rng.integers(1, n_users + 1, n_rows)
    m = rng.integers(1, n_movies + 1, n_rows)
    ts = np.sort(rng.integers(956_700_000, 1_046_000_000, n_rows))
    z = ub[u - 1] + mb[m - 1] + rng.normal(0, 1, n_rows)
    rating = np.clip(np.round(3.5 + z), 1, 5).astype(int)
    lines = ["userId,movieId,rating,timestamp"] + [f'{a},{b},{c},"{t}\n"' for a, b, c, t in zip(u, m, rating, ts)]
    (d / "ratings.csv").write_text("\n".join(lines) + "\n")
    users = ["userId,gender,age,occupation,zip_code"] + [
        f"{i},{'F' if i % 2 else 'M'},{[1, 18, 25, 35, 45, 50, 56][i % 7]},{i % 21},{'55117-1234' if i % 5 == 0 else f'{(i * 7) % 100000:05d}'}"
        for i in range(1, n_users + 1)]
    (d / "users.csv").write_text("\n".join(users) + "\n")
    movies = ["movieId,title,genres"] + [
        f'{i},"Movie {i}, The ({1930 + i * 2})",{"|".join(GENRES[(i + k) % 5] for k in range(1 + i % 3))}'
        for i in range(1, n_movies + 1)]
    (d / "movies.csv").write_text("\n".join(movies) + "\n")
    return [str(d / f) for f in ("ratings.csv", "users.csv", "movies.csv")]


@pytest.fixture
def synth(tmp_path, monkeypatch):
    files = write_synth(tmp_path)
    monkeypatch.setattr(D.assets, "resolve_dataset_files", lambda name, must_exist=True: files)
    return files


# ---------------------------------------------------------------- 解析
def test_load_raw_parses_quoted_timestamp_zip_and_genres(synth):
    raw = D.load_raw(synth)
    r = raw["ratings"]
    assert r["ts"].dtype == np.int64 and r["ts"].min() > 9e8
    assert np.all(np.diff(r["ts"]) >= 0)                                       # 按时间排好
    assert raw["users"]["zip1"][4] == 5 + 1                                    # 用户 5 的 "55117-1234" → 首位 5 → 编码 6
    assert raw["movies"]["year"][0] == 1932
    g = raw["movies"]["genres"]
    assert g.shape[1] == D.MAX_GENRES and (g[0] > 0).sum() == 2               # 电影 1：1 + 1 % 3 = 2 个题材


# ---------------------------------------------------------------- 指标
def brute_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(pos, neg)])


def test_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 2, 200).astype(float)
        s = np.round(rng.random(200) + 0.3 * y, 1)                             # 大量并列
        assert M.auc(y, s) == pytest.approx(brute_auc(y, s), abs=1e-12)
    assert np.isnan(M.auc(np.ones(5), np.arange(5.0)))


def test_gauc_weights_rows_and_skips_single_class_users():
    rng = np.random.default_rng(1)
    user = np.repeat([1, 2, 3], [30, 50, 10])
    y = rng.integers(0, 2, 90).astype(float)
    y[user == 3] = 1                                                           # 用户 3 只有正例：排除
    s = rng.random(90)
    want = sum((user == u).sum() * brute_auc(y[user == u], s[user == u]) for u in (1, 2)) / 80
    g, cov = M.gauc(y, s, user)
    assert g == pytest.approx(want, abs=1e-12) and cov == pytest.approx(80 / 90)
    # 对同一用户的所有行给同一个分数的特征：GAUC 恰好 0.5
    assert M.gauc(y, user.astype(float), user)[0] == pytest.approx(0.5)


def test_ne_and_bootstrap():
    y = np.array([0, 1, 1, 0, 1, 1, 0, 1], float)
    assert M.ne(y, np.full(8, y.mean())) == pytest.approx(1.0)
    user = np.repeat(np.arange(40), 25)
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 1000).astype(float)
    sd = M.boot_std(yy, rng.random(1000) + 0.2 * yy, user, 30)
    assert 0.001 < sd < 0.05


# ---------------------------------------------------------------- 切分与编码
def test_splits_share_the_future_holdout(synth):
    r = D.load_raw(synth)["ratings"]
    out = {m: D.split(r["ts"], r["user"], m, data_seed=0, k=3) for m in ("time", "random", "user_last")}
    hashes = {D.test_hash(s) for s, _ in out.values()}
    assert len(hashes) == 1                                                    # 三种切分的留出集完全相同
    s, cut = out["time"]
    assert r["ts"][s == 0].max() < r["ts"][s == 1].min() and r["ts"][s == 1].max() < r["ts"][s == 2].min()
    # user_last：每个用户 pool 里最后 k 条（行数 ≥ 2k 的用户）
    s = out["user_last"][0]
    for u in np.unique(r["user"]):
        rows = np.flatnonzero((r["user"] == u) & (s != 2))
        if len(rows) >= 6:
            assert set(np.flatnonzero((r["user"] == u) & (s == 1))) == set(rows[-3:])
    # random 只随 data_seed 变
    a = D.split(r["ts"], r["user"], "random", data_seed=0)[0]
    b = D.split(r["ts"], r["user"], "random", data_seed=1)[0]
    assert (a == D.split(r["ts"], r["user"], "random", data_seed=0)[0]).all() and (a != b).any()


def test_vocab_from_train_only_oov_is_zero():
    known, n = D.vocab(np.array([5, 3, 5, 9]))
    assert n == 4
    assert D.encode(np.array([3, 4, 9, 100, 5]), known).tolist() == [1, 0, 3, 0, 2]


# ---------------------------------------------------------------- 统计口径
def test_stats_modes():
    key = np.array([1, 1, 1, 1, 2, 2])
    ts = np.array([10, 20, 20, 86400 + 5, 10, 30])
    y = np.array([1, 0, 1, 1, 0, 1])
    c, p = F.stats(key, ts, y, "point_in_time")
    assert c.tolist() == [0, 1, 1, 3, 0, 1] and p.tolist() == [0, 1, 1, 2, 0, 0]  # 同一秒互相看不见
    c, p = F.stats(key, ts, y, "include_self")
    assert c.tolist() == [1, 2, 2, 4, 1, 2] and p.tolist() == [1, 1, 2, 3, 0, 1]
    c, _ = F.stats(key, ts, y, "daily")
    assert c.tolist() == [0, 0, 0, 3, 0, 0]                                  # 同一天的都看不见
    c, p = F.stats(key, ts, y, "full")
    assert c.tolist() == [4, 4, 4, 4, 2, 2] and p.tolist() == [3, 3, 3, 3, 1, 1]
    assert F.rows_upto_now(key, ts).tolist() == [1, 3, 3, 4, 1, 2]
    assert F.serve_mode("full") == F.serve_mode("include_self") == "point_in_time" and F.serve_mode("daily") == "daily"
    assert F.smooth_rate(np.array([3]), np.array([3]), 0.58, 20.0)[0] == pytest.approx((3 + 11.6) / 23)
    assert F.cnt_bucket(np.array([0, 1, 2, 3, 4, 7, 8, 10**9])).tolist() == [0, 1, 2, 2, 3, 3, 4, F.CNT_BUCKETS - 1]


def test_stats_random_against_loop():
    rng = np.random.default_rng(0)
    key, ts, y = rng.integers(0, 5, 300), rng.integers(0, 50, 300), rng.integers(0, 2, 300)
    c, p = F.stats(key, ts, y, "point_in_time")
    for i in range(300):
        m = (key == key[i]) & (ts < ts[i])
        assert c[i] == m.sum() and p[i] == y[m].sum()


# ---------------------------------------------------------------- 分组与模型
def test_groups_keys_list_and_dict_agree():
    rng = np.random.default_rng(0)
    n = 500
    masks = M.group_masks(rng.integers(0, 3, n), rng.integers(0, 600, n), rng.integers(0, 900, n), rng.random(n) < 0.3)
    y, s, u = rng.integers(0, 2, n).astype(float), rng.random(n), rng.integers(0, 30, n)
    by_key, rows = M.groups(y, s, u, masks)
    assert [r["key"] for r in rows] == [k for k, _, _ in M.GROUPS] == list(by_key)
    assert all("." not in k for k in by_key)
    for r in rows:
        assert by_key[r["key"]]["auc"] == r["auc"] and {"group", "value", "n", "label_rate", "auc"} <= set(r)


VOCAB = {"user_id": 50, "movie_id": 30, "gender": 3, "age": 8, "occupation": 22, "zip1": 11, "year_bucket": 18,
         "genre": 19, "user_cnt_bucket": F.CNT_BUCKETS, "item_cnt_bucket": F.CNT_BUCKETS}


def fake_batch(B=7):
    g = torch.Generator().manual_seed(0)
    b = {k: torch.randint(0, v, (B,), generator=g) for k, v in VOCAB.items() if k != "genre"}
    b["genres"] = torch.randint(0, 19, (B, 6), generator=g)
    for k in ("user_rate", "user_lcnt", "item_rate", "item_lcnt"):
        b[k] = torch.rand(B, generator=g)
    b["label"] = torch.randint(0, 2, (B,), generator=g).float()
    return b


@pytest.mark.parametrize("arch", ["mlp", "deepfm", "dcn"])
@pytest.mark.parametrize("groups", [["ids"], ["ids", "item_attr", "user_stats"],
                                    ["ids", "user_profile", "item_attr", "user_stats", "item_stats"], ["user_stats", "item_stats"]])
def test_model_shapes_and_counts(arch, groups):
    cfg = PARAMS.resolve({"arch": arch, "feature_groups": groups, "embedding_dim": 8, "hidden": "16,8"})
    m = RecModel(VOCAB, cfg)
    out = m(fake_batch())
    assert out.shape == (7,)
    pc = param_counts(m)
    n_dense = 2 * sum(g in groups for g in ("user_stats", "item_stats"))
    assert pc["mlp_input_dim"] == 8 * m.n_fields + n_dense
    assert pc["embedding"] + pc["mlp"] == pc["total"] and 0 < pc["embedding_share"] < 1
    out.sum().backward()


def test_ids_param_count_formula():
    m = RecModel({**VOCAB, "user_id": 6041, "movie_id": 3707}, PARAMS.resolve({}))
    pc = param_counts(m)
    assert pc["embedding"] == (6041 + 3707) * 32
    assert pc["mlp"] == 64 * 256 + 256 + 256 * 128 + 128 + 128 + 1


# ---------------------------------------------------------------- 端到端
class Ctx:
    def __init__(self, d: Path):
        self.run_dir, self.run_id = d, "t"
        self.s, self.tr, self.m, self.warns = {}, {}, [], []
        (d / "artifacts").mkdir(parents=True, exist_ok=True)

    def stage(self, n): pass
    def log(self, msg): pass
    def check(self): pass
    def info(self, *a): pass
    def note_cache(self, *a, **k): pass
    def register_artifact(self, *a, **k): pass
    def warn(self, c, m): self.warns.append(c)
    def metric(self, name, value, step=None, split="train", **kw): self.m.append({"name": name, "value": value, "step": step, **kw})
    def trace(self, k, p): self.tr[k] = json.loads(json.dumps(p, default=float))
    def summary(self, d): self.s.update(d)
    def artifact(self, n): return self.run_dir / "artifacts" / n


SUMMARY_PATHS = [
    "valid.auc", "valid.gauc", "valid.logloss", "valid.ne", "valid.auc_boot_std", "valid.gauc_row_cov", "test.auc", "test.gauc",
    "valid_serving.auc", "gap.valid_minus_test_auc", "gap.valid_minus_test_gauc", "gap.valid_minus_serving_auc",
    "reference.valid.item_mean.auc", "reference.valid.user_mean.gauc", "reference.test.item_mean.auc",
    "groups.valid.u_unseen.n", "groups.valid.u_seen.n", "groups.valid.u_new.n", "groups.valid.i_tail.n", "groups.valid.fut_user.n",
    "data.cold.valid.user_rows", "data.leak.train_has_future", "data.leak.valid_rows_with_later_train_same_user",
    "data.leak.valid_rows_with_later_train_same_item", "data.test_hash", "data.label_rate.valid",
    "stats_check.user_rows_using_future_share", "stats_check.single_auc.user_rate.valid_serving",
    "params.embedding_share", "best_epoch.epoch", "overfit.valid_auc_drop_from_best", "overfit.train_valid_logloss_gap",
]


def dig(d, path):
    for p in path.split("."):
        d = d[p]
    return d


@pytest.mark.parametrize("over", [{}, {"split": "random", "epochs": 2},
                                  {"feature_groups": ["ids", "user_profile", "item_attr", "user_stats", "item_stats"],
                                   "stats_mode": "full", "arch": "dcn"}])
def test_end_to_end(synth, tmp_path, over):
    cfg = PARAMS.resolve({"device": "cpu", "embedding_dim": 4, "hidden": "8", "batch_size": 64, "bootstrap": 5, **over})
    ctx = Ctx(tmp_path / "run")
    E.run(ctx, cfg)
    for p in SUMMARY_PATHS:
        dig(ctx.s, p)                                                          # 讲义要用的路径都在
    assert len(ctx.s["epochs"]) == cfg["epochs"]
    assert {"raw_csv", "sample", "batch", "forward", "update", "reload"} <= set(ctx.tr)
    assert ctx.tr["reload"]["ok"]
    assert set(ctx.tr["sample"]["stats_by_mode"]) == set(F.MODES)
    assert "ml1m_shape" in ctx.warns                                           # 合成数据的行数和官方不同
    curves = {m["name"] for m in ctx.m if m["step"] is None}
    assert {"valid_auc", "valid_auc_delta_pp", "test_auc"} <= curves
    if over.get("stats_mode") == "full":
        assert ctx.s["stats_check"]["user_rows_using_future_share"] > 0.5
        assert "验证 − 同行线上口径 AUC" in ctx.s["display"]
    if over.get("split") == "random":
        assert ctx.s["data"]["leak"]["train_has_future"]


def test_baseline_has_no_future_in_train(synth, tmp_path):
    ctx = Ctx(tmp_path / "run")
    E.run(ctx, PARAMS.resolve({"device": "cpu", "embedding_dim": 4, "hidden": "8", "batch_size": 64, "bootstrap": 0}))
    assert ctx.s["data"]["leak"]["train_has_future"] is False
    assert ctx.s["stats_check"]["user_rows_using_future_share"] == 0


# ---------------------------------------------------------------- 场景清单
def test_r0a_r0b_scenarios():
    for c, n in (("R0a", 8), ("R0b", 11)):
        specs = scenarios.expand(c, "h20x8")
        assert len({s.scenario for s in specs}) == n
        for s in scenarios.load_course(c)["scenarios"]:
            for e in s.get("expect", []):
                scenarios.check_expr(e)
    a = next(s for s in scenarios.expand("R0a", "h20x8", seeds=[0]) if s.scenario == "baseline")
    b = next(s for s in scenarios.expand("R0b", "h20x8", seeds=[0]) if s.scenario == "baseline")
    assert a.config_hash == b.config_hash, "R0a / R0b 的基线必须同配置，H20 上只跑一次"
    assert len([s for s in scenarios.expand("R0b", "h20x8") if s.scenario == "f_all"]) == 5
