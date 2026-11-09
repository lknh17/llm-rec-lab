"""recseq（R01 / R02）：合成长序列、截断、四个模型的因果 / padding 正确性、实现之间的等价、指标与成本公式、端到端。"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from labkit import scenarios
from recseq import cost as C
from recseq import data as D
from recseq import experiment as E
from recseq import synth as S
from recseq.models import DIN, build, build_mask, pos_bucket, time_bucket
from recseq.params import PARAMS
from recseq.train import recall_ndcg

V = 50


def cfg_for(model, attn_impl=None, attn_chunk=None, **over):
    c = PARAMS.resolve({"model": model, "d_model": 16, "n_head": 2, "dropout": 0.0, **over})
    impl = attn_impl or {"sasrec": "sdpa", "hstu": "manual"}.get(model, "none")
    return {**c, "max_len_eff": 24, "attn_impl_eff": impl, "attn_chunk": attn_chunk or c["attn_chunk"]}


def batch(B=3, L=12, lengths=(12, 7, 3), seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, V, (B, L), generator=g)
    for i, n in enumerate(lengths):
        x[i, : L - n] = 0
    ts = torch.cumsum(torch.randint(1, 5000, (B, L), generator=g), 1)
    return {"x": x, "ts": ts, "ts_next": torch.cat([ts[:, 1:], ts[:, -1:] + 30], 1)}


# ---------------------------------------------------------------- 合成数据
SYN = {**PARAMS.resolve({"dataset": "synth"}), "len_scale": 0.05}


def test_synth_deterministic_and_bursts_copy():
    a, b = S.generate(SYN, 5, seed=3), S.generate(SYN, 5, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in ("items", "ts", "burst"))
    for u in range(5):
        o, N = a["off"][u], a["N"][u]
        items, burst, ts = a["items"][o:o + N], a["burst"][o:o + N], a["ts"][o:o + N]
        assert np.all(np.diff(ts) > 0)
        starts = np.flatnonzero(burst == 1)
        assert len(starts) >= 2
        b_ = SYN["synth_burst"]
        segs = [items[s + 1:s + b_ + 1] for s in starts]
        assert len({int(items[s]) for s in starts}) == 1                    # 同一个回访入口 τ
        clusters = {int((x - S.FIRST_BG) // SYN["synth_items_per_cluster"]) for x in np.concatenate(segs)}
        assert len(clusters) == 1                                           # 回访里的物品全来自同一个回访簇


def test_cut_eval_types_and_long_dist():
    g = S.generate(SYN, 60, seed=1)
    c = S.cut_eval(g, seed=2)
    for u in range(60):
        o = g["off"][u]
        items = g["items"][o:o + g["N"][u]]
        cut = c["cut"][u]
        assert 0 < cut < g["N"][u]
        if c["kind"][u] == 0:
            assert g["burst"][o + cut] == 2                                 # 回访入口之后的第一个物品
            if c["dist"][u] > 0:
                assert g["burst"][o + cut - c["dist"][u]] == 2              # 上一段回访的第一个物品


def test_long_cover_monotone_and_full_at_max_len():
    """周期还受用户长度限制（P ≤ N − 2(b+1)），所以覆盖率不等于 ln(L/P_min)/ln(P_max/P_min)；只验证单调、满长度时为 1。"""
    cfg = {**SYN, "len_scale": 0.1}
    g = S.generate(cfg, 400, seed=0)
    c = S.cut_eval(g, seed=0)
    sc = S.scaled(cfg)
    covers = [S.long_cover(c["dist"], c["kind"], L) for L in (sc["p_min"], 100, 400, sc["len_max"])]
    assert covers == sorted(covers) and covers[-1] == 1.0 and covers[0] < 0.2


def test_pad_waste_example():
    w = S.pad_waste(np.array([10000, 1000, 1000, 100]), 10000)
    assert w["pad_token_waste"] == pytest.approx(1 - 12100 / 40000, abs=1e-4)
    assert 1 / (1 - w["pad_attn_waste"]) == pytest.approx(4e8 / (1e8 + 2e6 + 1e4), rel=1e-3)


# ---------------------------------------------------------------- 截断与窗口
def test_truncate_modes():
    assert D.truncate(10, 4, "recent") == (6, 10)
    assert D.truncate(10, 4, "earliest") == (0, 4)
    assert D.truncate(3, 4, "earliest") == (0, 3)


def test_train_batch_left_padded_and_shifted():
    g = S.generate(SYN, 4, seed=0)
    d = D.from_synth(g, g, g, S.cut_eval(g, 0), S.cut_eval(g, 1), S.n_items(SYN))
    b = D.train_batch(d, np.random.default_rng(0), 8, 40, "recent")
    for x, y in zip(b["x"], b["y"]):
        n = int((x != 0).sum())
        assert (x[: 40 - n] == 0).all() and (x[40 - n:] != 0).all()
        assert np.array_equal(x[40 - n + 1:], y[40 - n:-1])                # y 是 x 右移一位


# ---------------------------------------------------------------- 模型
@pytest.mark.parametrize("model,impl", [("mean", "none"), ("din", "none"), ("sasrec", "sdpa"), ("sasrec", "manual"), ("hstu", "manual"),
                                        ("hstu", "chunked")])
def test_causal_and_pad_invariance(model, impl):
    torch.manual_seed(0)
    c = cfg_for(model, attn_impl=impl, attn_chunk=5)
    m = build(V, c).eval()
    b = batch()
    with torch.no_grad():
        full = m(b)
        b2 = {**b, "x": b["x"].clone()}
        b2["x"][:, -1] = (b2["x"][:, -1] % (V - 1)) + 1
        real = (b["x"] != 0)[:, :-1, None]                                  # padding 位置的输出不参与监督和评估
        assert ((m(b2)[:, :-1] - full[:, :-1]).abs() * real).max() < 1e-5  # 改最后一个输入，前面的真实位置不变
        for i, n in enumerate((12, 7, 3)):
            single = {k: v[i:i + 1, 12 - n:] for k, v in b.items()}
            assert (m(single)[0, -1] - full[i, -1]).abs().max() < 1e-4       # 补不补齐结果一样


def test_causal_only_mask_is_detected():
    torch.manual_seed(0)
    m = build(V, cfg_for("hstu", mask="causal_only")).eval()
    b = batch()
    with torch.no_grad():
        full = m(b)
        single = {k: v[2:3, 9:] for k, v in b.items()}
        assert (m(single)[0, -1] - full[2, -1]).abs().max() > 1e-3
        m(b)
        a = m.layers[0].last_attn
        assert float((a[2] * (b["x"][2] == 0)).sum() / a[2].sum()) > 0.5   # 短用户的注意力大部分落在 pad 上


def test_sasrec_manual_equals_sdpa_and_hstu_chunked_equals_manual():
    torch.manual_seed(0)
    a = build(V, cfg_for("sasrec", attn_impl="sdpa"))
    b_ = build(V, cfg_for("sasrec", attn_impl="manual"))
    b_.load_state_dict(a.state_dict())
    x = batch()
    assert (a.eval()(x) - b_.eval()(x)).abs().max() < 1e-4
    torch.manual_seed(1)
    h1 = build(V, cfg_for("hstu", attn_impl="manual"))
    h2 = build(V, cfg_for("hstu", attn_impl="chunked", attn_chunk=5))
    h2.load_state_dict(h1.state_dict())
    o1, o2 = h1(x), h2(x)
    assert (o1 - o2).abs().max() < 1e-5
    o1.sum().backward(); o2.sum().backward()
    for p1, p2 in zip(h1.parameters(), h2.parameters()):
        assert (p1.grad - p2.grad).abs().max() < 1e-4


def test_din_prefix_sum_equals_per_position_softmax():
    """前缀和写法 = 每个位置 t、每个候选 c 单独对 j ≤ t 做 softmax 再加权。"""
    torch.manual_seed(0)
    m = DIN(V, cfg_for("din")).eval()
    b = batch()
    with torch.no_grad():
        got = m(b)                                                          # [B, L, V]
        h = m.emb(b["x"])
        for i in range(b["x"].shape[0]):
            for t in (5, 11):
                ok = (b["x"][i] != 0) & (torch.arange(12) <= t)
                if not ok.any():
                    continue
                for c in (1, 7, 33):
                    e = m.emb.weight[c]
                    a = torch.softmax(((m.W(h[i]) @ e) / math.sqrt(m.d))[ok], 0)
                    assert got[i, t, c] == pytest.approx(float((a * (h[i][ok] @ e)).sum()), abs=1e-5)


def test_buckets():
    assert time_bucket(torch.tensor([0, 1, 3, 86400, 10 ** 12]), 32).tolist() == [0, 1, 2, 16, 31]
    pb = pos_bucket(torch.arange(0, 20000, 7), 64, 10240)
    assert (pb[1:] >= pb[:-1]).all() and pb.max() == 63 and pos_bucket(torch.tensor([5]), 64, 10240).item() == 5
    m = build_mask(torch.tensor([[False, True, True]]), 0, 3, "causal+pad")[0, 0]
    assert m.int().tolist() == [[0, 0, 0], [0, 1, 0], [0, 1, 1]]


# ---------------------------------------------------------------- 指标与成本
def test_recall_ndcg():
    r = recall_ndcg(np.array([0, 3, 9, 10, 50]))
    assert r["recall10"] == pytest.approx(0.6)
    assert r["ndcg10"] == pytest.approx((1 + 1 / math.log2(5) + 1 / math.log2(11)) / 5, abs=1e-5)


def test_cost_formulas():
    c = PARAMS.resolve({})
    s = C.estimate({**c, "model": "sasrec"}, 3707, 200)
    assert s["attn_share"] == pytest.approx(200 / (200 + 768), abs=1e-3)
    din = C.estimate({**c, "model": "din"}, 3707, 200)
    assert din["infer_macs_per_user"] / s["infer_macs_per_user"] == pytest.approx(3.8, abs=0.1)
    h = C.estimate({**c, "model": "hstu"}, 4105, 10240)
    assert h["attn_share"] > 0.96
    assert C.est_attn_gb({**c, "model": "sasrec"}, 10240) == pytest.approx(2 * 2 * 10240 ** 2 * 6 / 1e9, rel=1e-3)


def test_probe_fields_cpu():
    c = {**cfg_for("hstu"), "max_len_eff": 32}
    p = C.probe(c, V, 32, "manual", torch.device("cpu"))
    assert set(p) == {"ms", "peak_gb", "est_gb", "oom"} and p["ms"] > 0 and not p["oom"]
    j = C.jagged_probe(c, V, [4, 8, 32], torch.device("cpu"))
    assert j["speedup"] is not None


# ---------------------------------------------------------------- 端到端
class Ctx:
    def __init__(self, d: Path):
        self.run_dir, self.run_id, self.s, self.tr, self.m = d, "t", {}, {}, []
        (d / "artifacts").mkdir(parents=True, exist_ok=True)

    def stage(self, n): pass
    def log(self, m): pass
    def check(self): pass
    def info(self, *a): pass
    def warn(self, *a): pass
    def note_cache(self, *a, **k): pass
    def register_artifact(self, *a, **k): pass
    def metric(self, name, value, step=None, **kw): self.m.append(name)
    def trace(self, k, p): self.tr[k] = json.loads(json.dumps(p, default=str))
    def summary(self, d): self.s.update(d)
    def artifact(self, n): return self.run_dir / "artifacts" / n


def dig(d, p):
    for k in p.split("."):
        d = d[k]
    return d


R02_PATHS = ["test.recall10", "test.ndcg10", "test.sampled100.recall10", "test.pop.recall10", "test.long.recall10",
             "test.recent.recall10", "test.newsess.recall10", "test.long_by_dist.lt512.recall10", "test.long_by_dist.unseen.n", "test.mix.r05.recall10",
             "test.by_len.short.recall10", "data.long_cover.L512", "data.pad_attn_waste", "cost.flops.attn_share",
             "cost.L512.manual.ms", "cost.L10240.chunked.ms", "cost.jagged.speedup", "checks.pad_invariance_max_dlogit",
             "checks.pad_attn_mass_short", "train.tokens_per_s", "valid.recall10"]


@pytest.mark.parametrize("model", ["hstu", "sasrec"])
def test_end_to_end_synth(tmp_path, model):
    over = {"dataset": "synth", "model": model, "max_len": 10240, "batch_size": 0, "tokens_per_step": 2048, "steps": 4,
            "ckpt_every": 2, "valid_frac": 1.0, "len_scale": 0.01, "synth_users": 16, "synth_eval_users": 16, "d_model": 16,
            "probe_lens": "512,2048,8192,10240", "device": "cpu"}
    ctx = Ctx(tmp_path / "run")
    E.run(ctx, PARAMS.resolve(over))
    for p in R02_PATHS:
        if model == "sasrec":
            p = p.replace("chunked", "sdpa")
            if p.startswith("checks.pad_attn"):
                continue
        dig(ctx.s, p)
    assert {"batch", "synth_user", "mask", "rab", "probe"} <= set(ctx.tr)
    assert ctx.s["checks"]["causal_max_dlogit"] < 1e-4


def test_scenarios_r01_r02():
    for c, n in (("R01", 8), ("R02", 10)):
        assert len({s.scenario for s in scenarios.expand(c, "h20x8")}) == n
        for s in scenarios.load_course(c)["scenarios"]:
            for e in s.get("expect", []):
                scenarios.check_expr(e)
