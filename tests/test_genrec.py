"""genrec（R03 / R04）：k-means、残差量化、去重位、碰撞统计、trie、beam search（含故障）、encoder-decoder、端到端。"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from genrec import sid as S
from genrec.items import synth_items
from genrec.model import BOS, TensorTrie, TigerEncDec, Tokenizer, beam_search
from genrec.params import PARAMS
from genrec.train import Decoder
from labkit import scenarios


def blobs(n=600, k=6, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(k, d, generator=g) * 6
    return c[torch.randint(0, k, (n,), generator=g)] + torch.randn(n, d, generator=g)


def test_kmeans_full_utilization_and_finds_blobs():
    x = blobs()
    c, a = S.kmeans(x, 6, 20, seed=0)
    assert torch.bincount(a, minlength=6).min() > 0
    assert S.level_stats(a[:, None], 6)["L1"]["used_frac"] == 1.0


def test_residual_vs_noresid():
    x = blobs(800, 16, 16)
    codes, _, errs = S.rq_kmeans(x, 3, 16, 15, seed=0, residual=True)
    assert errs[2] < errs[0]
    bad, _, errs_bad = S.rq_kmeans(x, 3, 16, 15, seed=0, residual=False)
    assert errs_bad[2] >= errs_bad[0] - 1e-6                                # 每层都对原向量聚类：重建不会变好
    assert S.nmi(bad[:, 0], bad[:, 1]) > S.nmi(codes[:, 0], codes[:, 1])
    assert S.collision_stats(bad, 16)["collision_rate"] > S.collision_stats(codes, 16)["collision_rate"]


def test_greedy_depth_prefix_equals_shallow_run():
    x = blobs(500, 10, 8)
    deep, _, _ = S.rq_kmeans(x, 3, 6, 10, seed=1)
    shallow, _, _ = S.rq_kmeans(x, 2, 6, 10, seed=1)
    assert torch.equal(deep[:, :2], shallow)                               # 逐层贪心：4 层的前 d 层 = d 层


def test_pigeonhole_and_uniform():
    codes = torch.randint(0, 4, (1000, 2))
    st = S.collision_stats(codes, 4)
    assert st["collision_rate"] >= 1 - 16 / 1000
    assert st["uniform"] == pytest.approx(1 - (1 - 1 / 16) ** 999, abs=1e-4)


def test_add_dedup_unique_and_matches_loop():
    codes = torch.randint(0, 3, (200, 2))
    full = S.add_dedup(codes, 3)
    assert len({tuple(r) for r in full.tolist()}) == 200
    seen, want = {}, []
    order = sorted(range(200), key=lambda i: (S.pack(codes[i:i + 1], 3).item(), i))
    for i in order:
        key = tuple(codes[i].tolist())
        seen[key] = seen.get(key, -1) + 1
        want.append((i, seen[key]))
    assert all(full[i, -1].item() == v for i, v in want)


def toy_tok(n=40, k=4, levels=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    codes = S.add_dedup(torch.randint(0, k, (n, levels), generator=g), k)
    return Tokenizer(codes, k, levels), codes


def test_trie_allowed_matches_dict():
    tok, _ = toy_tok()
    trie = TensorTrie(tok.item_tokens, tok.vocab, "cpu")
    seqs = tok.item_tokens[1:].tolist()
    prefixes = {tuple(s[:d]) for s in seqs for d in range(tok.depth)}
    ids = {(): 0}
    node = 0
    for p in sorted(prefixes, key=len):
        if not p:
            continue
        ids[p] = trie.step(torch.tensor([ids[p[:-1]]]), torch.tensor([p[-1]])).item()
    for p, n in ids.items():
        want = {s[len(p)] for s in seqs if tuple(s[:len(p)]) == p}
        got = set(torch.nonzero(trie.allowed(torch.tensor([n]))[0]).flatten().tolist())
        assert got == want


def tiny_model(tok, seed=0):
    torch.manual_seed(seed)
    cfg = PARAMS.resolve({"d_model": 16, "enc_layers": 1, "dec_layers": 1, "n_head": 2, "dropout": 0.0, "max_hist": 6})
    return TigerEncDec(tok.vocab, tok.depth, cfg).eval()


def test_beam_unique_constrained_and_fault():
    tok, _ = toy_tok()
    trie = TensorTrie(tok.item_tokens, tok.vocab, "cpu")
    m = tiny_model(tok)
    hist = tok.item_tokens[torch.tensor([[0, 0, 3, 5, 7, 9], [1, 2, 3, 4, 5, 6]])]
    seqs, sc = beam_search(m, hist, trie, 5, "one_root")
    valid = {tuple(r) for r in tok.item_tokens[1:].tolist()}
    for b in range(2):
        rows = [tuple(r) for r in seqs[b].tolist()]
        assert len(set(rows)) == 5 and all(r in valid for r in rows)       # W 条两两不同、全部合法
    dup, _ = beam_search(m, hist, trie, 5, "all_zero")
    assert all(len({tuple(r) for r in dup[b].tolist()}) == 1 for b in range(2))   # 故障：W 条完全一样
    g1, _ = beam_search(m, hist, trie, 1, "one_root")
    assert torch.equal(g1[:, 0], dup[:, 0])                                 # 故障的结果 = 贪心


def test_beam_equals_exhaustive_topw():
    tok, _ = toy_tok(n=30, k=3, levels=2)
    trie = TensorTrie(tok.item_tokens, tok.vocab, "cpu")
    m = tiny_model(tok, seed=3)
    hist = tok.item_tokens[torch.tensor([[0, 1, 2, 3, 4, 5]])]
    W = 40                                                                  # W ≥ 合法前缀数时 beam = 全枚举
    seqs, sc = beam_search(m, hist, trie, W, "one_root")
    torch.set_grad_enabled(False)
    mem, pad = m.encode(hist)
    best = []
    for row in tok.item_tokens[1:].tolist():
        prefix = torch.tensor([[BOS] + row])
        lp = torch.log_softmax(m.decode(prefix[:, :-1], mem, pad)[0], -1)
        allowed_lp = 0.0
        node = torch.tensor([0])
        for t, tk in enumerate(row):
            mask = trie.allowed(node)[0]
            lp_t = lp[t].masked_fill(~mask, float("-inf"))
            allowed_lp += float(lp_t[tk])
            node = trie.step(node, torch.tensor([tk]))
        best.append((allowed_lp, tuple(row)))
    torch.set_grad_enabled(True)
    best.sort(reverse=True)
    got = [tuple(r) for r, s in zip(seqs[0].tolist(), sc[0].tolist()) if math.isfinite(s)]
    assert got[:10] == [r for _, r in best[:10]]


def test_decoder_without_dedup_fills_by_popularity():
    codes = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]])
    tok = Tokenizer(codes, 2, 2)
    pop = np.array([5, 50, 1, 20])
    dec = Decoder(tok, pop)
    seqs = tok.item_tokens[torch.tensor([[1, 3]])]                         # 第一个 beam 是 (0,1) 的桶，第二个是 (1,0)
    lists, st = dec.topk(seqs, torch.tensor([[0.0, -1.0]]))
    assert lists[0] == [2, 4, 1, 3] and st["invalid_frac"] == 0


def test_synth_items_deterministic_and_clustered():
    a, la = synth_items(3000, 16, 0, "cpu")
    b, _ = synth_items(3000, 16, 0, "cpu")
    assert torch.equal(a, b)
    codes, _, _ = S.rq_kmeans(torch.nn.functional.normalize(a, dim=1), 2, 16, 10, seed=0)
    from genrec.experiment import cross_stats
    cs = cross_stats(codes, None, la, 16, 0)
    assert cs["true_top_purity"] > 0.5


# ---------------------------------------------------------------- 端到端（合成数据，不需要 ML-1M）
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


@pytest.mark.parametrize("over", [{}, {"quantizer": "rqvae", "sid_levels": 3, "rqvae_steps": 20}, {"residual": False}])
def test_end_to_end_sid_synth(tmp_path, over):
    from genrec import experiment as E

    cfg = PARAMS.resolve({"items": "synth", "synth_n": 3000, "synth_dim": 16, "sid_levels": 4, "sid_k": 16, "kmeans_iters": 3,
                          "device": "cpu", **over})
    ctx = Ctx(tmp_path / "run")
    E.run(ctx, cfg)
    for p in ("sid.collision_rate", "sid.uniform", "sid.by_depth.d1.collision_rate", "sid.levels.L1.ppl_frac", "sid.nmi.L1_L2",
              "sid.prefix_sim.d1", "sid.cross.true_top_purity", "sid.at_cap.c16m"):
        dig(ctx.s, p)


def test_scenarios_r03_r04():
    for c, n in (("R03", 10), ("R04", 10)):
        assert len({s.scenario for s in scenarios.expand(c, "h20x8")}) == n
        for s in scenarios.load_course(c)["scenarios"]:
            for e in s.get("expect", []):
                scenarios.check_expr(e)
