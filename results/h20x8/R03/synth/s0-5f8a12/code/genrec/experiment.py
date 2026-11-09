"""genrec 执行入口。task=sid（R03）：物品向量 → 归一化 → 残差量化 → 碰撞 / 利用率 / 重建 / 跨视角统计。
task=retrieval（R04）：语义 ID + 去重位 → trie → 训练 encoder-decoder（缓存）→ beam 解码评估 → 成本基准。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from labkit.cache import stage_cache

from . import cost as C
from . import items as I
from . import sid as S
from .model import TensorTrie, TigerEncDec, Tokenizer, beam_search
from .params import PARAMS
from .rqvae import train_rqvae
from .train import Decoder, autocast, evaluate, train

TIGER_VERSION = 1
CAP = 16_777_216


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_x(ctx, cfg: dict, device) -> tuple[torch.Tensor, dict | None, dict]:
    """物品向量 [N, D]（第 i 行 = 物品 i+1）、ML-1M 元信息（合成时为 None）、来源说明。"""
    if cfg["items"] == "synth":
        x, labels = I.synth_items(cfg["synth_n"], cfg["synth_dim"], cfg["data_seed"], device)
        return x, None, {"source": "合成三层高斯树", "labels": labels}
    meta = I.ml1m_meta()
    if cfg["item_repr"] == "sasrec":
        emb, info = I.cf_embeddings(ctx, cfg, device)
        src = {"source": "SASRec 物品 embedding（ML-1M，只用 seq[:-2] 训练）", "cf_valid_recall10": info["best_valid_recall10"]}
    elif cfg["item_repr"] == "qwen":
        emb = I.content_embeddings(ctx, meta["titles"], meta["genres"], device)
        src = {"source": "Qwen2.5-0.5B 读片名 + 类型，末层 mean pooling"}
    else:
        emb = np.random.default_rng(cfg["seed"] + 5).standard_normal((len(meta["titles"]), cfg["cf_d_model"])).astype(np.float32)
        src = {"source": "随机高斯向量"}
    return torch.from_numpy(np.asarray(emb, np.float32)).to(device), meta, src


def quantize(ctx, cfg: dict, x: torch.Tensor, q_seed: int) -> tuple[torch.Tensor, list, list | None, dict]:
    L, K = cfg["sid_levels"], cfg["sid_k"]
    if cfg["quantizer"] == "rqvae":
        codes, info = train_rqvae(ctx, x, L, K, cfg["rqvae_steps"], q_seed)
        return codes, [], None, info
    codes, books, errs = S.rq_kmeans(x, L, K, cfg["kmeans_iters"], q_seed, cfg["residual"], cfg["fit_sample"],
                                     on_iter=lambda it, e: ctx.metric("kmeans_err", e, step=it + 1))
    return codes, books, errs, {}


def cross_stats(codes: torch.Tensor, meta: dict | None, labels: dict | None, k: int, seed: int) -> dict:
    """跨视角检验：不在量化用的那个空间里量相似度，避免循环论证。"""
    c1 = codes[:, 0].cpu().numpy()
    if meta is None:                                                         # 合成：和真实簇对照的纯度
        def purity(pred, true):
            key = pred.astype(np.int64) * (true.max() + 1) + true
            _, cnt = np.unique(key, return_counts=True)
            pairs = np.unique(key)
            best = {}
            for p, n in zip(pairs // (true.max() + 1), cnt):
                best[p] = max(best.get(p, 0), n)
            return round(sum(best.values()) / len(pred), 4)
        top, sub = labels["top"].cpu().numpy(), labels["sub"].cpu().numpy()
        c12 = S.pack(codes[:, :2], k).cpu().numpy()
        return {"true_top_purity": purity(c1, top), "true_sub_purity": purity(c12, sub)}
    rng = np.random.default_rng(seed)
    a = np.concatenate([s[:-1] for s in meta["train_seqs"]]) - 1
    b = np.concatenate([s[1:] for s in meta["train_seqs"]]) - 1
    G = meta["genre_ids"]
    sets = [set(r[r > 0].tolist()) for r in G]
    jac = lambda i, j: len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))  # noqa: E731
    order = np.argsort(c1, kind="stable")
    same = np.flatnonzero(c1[order][1:] == c1[order][:-1])
    pick = rng.choice(same, size=min(20000, len(same)), replace=False) if len(same) else []
    rnd = rng.integers(0, len(c1), size=(20000, 2))
    y = np.log1p(meta["pop"].astype(np.float64))
    gm = np.bincount(c1, weights=y, minlength=k) / np.maximum(np.bincount(c1, minlength=k), 1)
    eta2 = float(((gm[c1] - y.mean()) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    return {"cooc_same_c1": round(float((c1[a] == c1[b]).mean()), 4),
            "genre_jacc_c1": round(float(np.mean([jac(order[i], order[i + 1]) for i in pick])), 4) if len(pick) else None,
            "genre_jacc_rand": round(float(np.mean([jac(i, j) for i, j in rnd])), 4), "pop_eta2_c1": round(eta2, 4)}


def sid_summary(codes, x, errs, K, meta, labels, seed) -> dict:
    L = codes.shape[1]
    out = {**S.collision_stats(codes, K), "levels": S.level_stats(codes, K)}
    out["by_depth"] = {f"d{d}": S.collision_stats(codes[:, :d], K) for d in range(1, L + 1)}
    dcap = math.ceil(math.log(CAP) / math.log(K))
    out["at_cap"] = {"c16m": {"depth": dcap, "collision_rate": out["by_depth"][f"d{dcap}"]["collision_rate"] if dcap <= L else None}}
    out["recon"] = {f"L{j + 1}": round(e, 4) for j, e in enumerate(errs)} if errs else {}
    out["nmi"] = {f"L{j + 1}_L{j + 2}": S.nmi(codes[:, j], codes[:, j + 1]) for j in range(L - 1)}
    out["prefix_sim"] = S.prefix_sim(x, codes, K, seed=seed)
    out["cross"] = cross_stats(codes, meta, labels, K, seed)
    return out


def run(ctx, cfg: dict) -> None:
    ctx.stage("prepare")
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    if cfg["task"] == "retrieval" and cfg["method"] == "sasrec":
        return run_sasrec(ctx, cfg, device)
    import time
    t0 = time.time()
    x, meta, src = load_x(ctx, cfg, device)
    labels = src.pop("labels", None)
    if cfg["normalize"] == "l2":
        x = F.normalize(x, dim=1)
    emb_s = time.time() - t0
    q_seed = cfg["seed"] if cfg["task"] == "sid" else 0                     # R04 的语义 ID 固定，种子只影响模型训练
    t1 = time.time()
    codes, books, errs, rq_info = quantize(ctx, cfg, x, q_seed)
    fit_s = time.time() - t1
    K, L = cfg["sid_k"], cfg["sid_levels"]
    sid = sid_summary(codes, x, errs, K, meta, labels, q_seed)
    sid["timing"] = {"emb_s": round(emb_s, 2), "quantize_s": round(fit_s, 2)}
    ctx.info("data", {"kind": "合成物品" if meta is None else "MovieLens-1M 电影", "items": int(len(x)), "dim": int(x.shape[1]), **src,
                      "normalize": cfg["normalize"], "quantizer": cfg["quantizer"], "levels": L, "k": K})
    ctx.log(f"语义 ID：{len(x):,} 个物品、{L} 层 × {K} 码；碰撞率 {sid['collision_rate']:.1%}（均匀基准 {sid['uniform']:.1%}），"
            f"码组合 {sid['combos']:,}；每层重建误差 {sid['recon']}")
    if meta is not None:
        trace_quantize(ctx, codes, books, x, meta, K)
    if cfg["task"] == "sid":
        ctx.summary({"sid": sid, "rqvae": rq_info, "headline": {"collision_rate": sid["collision_rate"]}, "display": sid_display(sid, L, meta),
                     "display_pct": ["碰撞率", "均匀基准碰撞率", "L1 困惑度 / K", f"L{L} 困惑度 / K", "最低码字利用率", "同 c1 相邻行为比例"]})
        np.save(ctx.artifact("codes.npy"), codes.cpu().numpy())
        ctx.register_artifact("codes.npy", "data", "每个物品的语义 ID [N, L]")
        return
    run_retrieval(ctx, cfg, device, codes, sid, meta)


def sid_display(sid: dict, L: int, meta) -> dict:
    lv = sid["levels"]
    d = {"碰撞率": sid["collision_rate"], "均匀基准碰撞率": sid["uniform"], "码组合数": sid["combos"],
         "L1 困惑度 / K": lv["L1"]["ppl_frac"], f"L{L} 困惑度 / K": lv[f"L{L}"]["ppl_frac"],
         "最低码字利用率": min(v["used_frac"] for v in lv.values()), "最大桶": sid["max_bucket"]}
    if sid["recon"]:
        d["末层重建误差"] = sid["recon"][f"L{L}"]
    if sid["nmi"]:
        d["NMI(L1, L2)"] = sid["nmi"]["L1_L2"]
    d["同前缀余弦（d1）"] = sid["prefix_sim"]["d1"]
    if meta is not None:
        d["同 c1 相邻行为比例"] = sid["cross"]["cooc_same_c1"]
    return d


def trace_quantize(ctx, codes, books, x, meta, K) -> None:
    titles = meta["titles"]
    i = next((j for j, t in enumerate(titles) if t.startswith("Toy Story (1995)")), 0)
    c = codes[i].tolist()
    norms, r = [], x[i].clone()
    for lv, book in enumerate(books):
        r = r - book[c[lv]]
        norms.append(round(float(r.norm()), 4))
    same1 = [titles[j] for j in torch.nonzero(codes[:, 0] == c[0]).flatten().tolist() if j != i][:5]
    bucket = [titles[j] for j in torch.nonzero((codes == codes[i]).all(1)).flatten().tolist() if j != i][:5]
    ctx.trace("quantize", {"note": "《Toy Story》：每层选的码、减掉之后剩下的残差范数；同一个第 1 层码的电影；码完全相同（碰撞）的电影",
                           "title": titles[i], "x_shape": list(x.shape), "x_norm": round(float(x[i].norm()), 4), "codes": c,
                           "residual_norm_after_level": norms, "same_c1": same1, "same_bucket": bucket,
                           "codebook_shape": [len(books), K, int(x.shape[1])] if books else None})


# ------------------------------------------------------------------ R04
def tiger_key(cfg: dict) -> dict:
    keep = {p.name for p in PARAMS.params if p.affects in ("data", "model", "train")}
    return {"v": TIGER_VERSION, **{k: cfg[k] for k in sorted(keep)}}


def run_retrieval(ctx, cfg, device, codes, sid, meta) -> None:
    from recseq import data as RD

    K, L = cfg["sid_k"], cfg["sid_levels"]
    full = S.add_dedup(codes, K) if cfg["collision"] == "suffix" else codes
    tok = Tokenizer(full, K, L)
    trie = TensorTrie(tok.item_tokens, tok.vocab, device)
    dec = Decoder(tok, meta["pop"])
    d, _ = RD.load_ml1m()
    users, cuts = d.evals["test"]
    if cfg["eval_users"]:
        users, cuts = users[: cfg["eval_users"]], cuts[: cfg["eval_users"]]
    model = TigerEncDec(tok.vocab, tok.depth, cfg).to(device)

    def build_model(dd: Path) -> None:
        info = train(ctx, cfg, model, d, tok, trie, dec, device)
        torch.save(model.state_dict(), dd / "model.pt")
        (dd / "info.json").write_text(json.dumps(info))

    ctx.stage("train")
    mdir = stage_cache(ctx, "genrec_tiger", tiger_key(cfg), build_model)
    model.load_state_dict(torch.load(mdir / "model.pt", map_location=device))
    tr = json.loads((mdir / "info.json").read_text())
    ctx.info("model", {"arch": f"encoder-decoder（{cfg['enc_layers']} + {cfg['dec_layers']} 层，d = {cfg['d_model']}）逐个生成 {tok.depth} 个 token",
                       "params": {"total": tr["params"], "trainable": tr["params"]}, "device": str(device)})
    ctx.info("train", {"seconds": tr["seconds"], "steps": tr["steps"]})

    ctx.stage("evaluate")
    test, lists, targets = evaluate(model, d, tok, trie, dec, users, cuts, cfg, device, cfg["beam"], cfg["constrained"], cfg["beam_init"])
    beam = {}
    for W in [int(w) for w in str(cfg["beam_sweep"]).split(",") if w.strip()]:
        r, _, _ = evaluate(model, d, tok, trie, dec, users, cuts, cfg, device, W, cfg["constrained"], cfg["beam_init"])
        beam[f"w{W}"] = {k: r[k] for k in ("recall10", "invalid_frac", "unique_per_user", "ms_per_user")}
        ctx.metric("beam_recall10", r["recall10"], step=W, split="test")
        ctx.metric("beam_ms", r["ms_per_user"], step=W, split="test")
    notrie, _, _ = evaluate(model, d, tok, trie, dec, users, cuts, cfg, device, 10, False, cfg["beam_init"])
    top_pop = (np.argsort(-meta["pop"])[:10] + 1).tolist()
    pop = {"recall10": round(float(np.mean([t in top_pop for t in targets])), 5)}
    key = S.pack(codes, K)
    _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    collided = (cnt[inv] > 1).cpu().numpy()
    sid_r = {"collision_rate": sid["collision_rate"], "vocab": tok.vocab, "tokens_per_item": tok.depth,
             "target_collided_frac": round(float(collided[targets - 1].mean()), 4)}
    bench = C.full_scoring_ms(cfg["cost_bench_n"], cfg["d_model"], device)
    ms10 = beam.get("w10", {}).get("ms_per_user") or (test["ms_per_user"] if cfg["beam"] == 10 else None)
    cost = {"full_ms_1e7": bench["ms_per_user"], "item_table_gb_1e7": bench["table_gb"], "beam_ms_w10": ms10,
            "full_over_beam_1e7": round(bench["ms_per_user"] / ms10, 3) if bench["ms_per_user"] and ms10 else None,
            "bench_n": cfg["cost_bench_n"]}
    trace_decode(ctx, cfg, model, d, tok, trie, dec, users, cuts, meta, device)
    disp = {"Recall@10": test["recall10"], "NDCG@10": test["ndcg10"], "非法元组比例": test["invalid_frac"],
            "每用户不重复物品数": test["unique_per_user"], "解码 ms/用户": test["ms_per_user"], "碰撞率": sid["collision_rate"],
            "训练秒数": tr["seconds"], "参数量": tr["params"]}
    ctx.summary({"test": test, "beam": beam, "notrie": {"w10": notrie}, "pop": pop, "sid": sid_r,
                 "trie": {"nodes": trie.nodes, "mb": trie.mb, "branching": trie.branching}, "train": tr, "cost": cost,
                 "headline": {"recall10": test["recall10"], "ndcg10": test["ndcg10"]}, "display": disp,
                 "display_pct": ["Recall@10", "NDCG@10", "非法元组比例", "碰撞率"]})
    ctx.log(f"测试 Recall@10 {test['recall10']}（beam {cfg['beam']}，trie {cfg['constrained']}，初始化 {cfg['beam_init']}）；"
            f"不重复物品 {test['unique_per_user']:.1f}；热度参照 {pop['recall10']}")
    torch.save(model.state_dict(), ctx.artifact("model.pt"))
    again = TigerEncDec(tok.vocab, tok.depth, cfg).to(device)
    again.load_state_dict(torch.load(ctx.artifact("model.pt"), map_location=device))
    r1, l1, _ = evaluate(again, d, tok, trie, dec, users[:16], cuts[:16], cfg, device, cfg["beam"], cfg["constrained"], cfg["beam_init"])
    ctx.trace("reload", {"same_top10": l1 == lists[:16]})
    ctx.register_artifact("model.pt", "checkpoint", "encoder-decoder 权重")


def trace_decode(ctx, cfg, model, d, tok, trie, dec, users, cuts, meta, device) -> None:
    from .train import hist_batch

    titles = meta["titles"]
    u, c = users[:1], cuts[:1]
    h = torch.from_numpy(hist_batch(d, u, c, cfg["max_hist"])).to(device)
    item_tok = tok.item_tokens.to(device)
    model.eval()
    with autocast(device):
        seqs, sc = beam_search(model, item_tok[h], trie if cfg["constrained"] else None, cfg["beam"], cfg["beam_init"])
    lists, _ = dec.topk(seqs, sc)
    target = int(d.seq(int(u[0]))[0][int(c[0])])
    hist_items = [int(i) for i in h[0].tolist() if i][-5:]
    ctx.trace("decode", {"note": "一个测试用户：最近 5 部电影和它们的语义 ID token；beam 解码出的 token 元组、分数和对应电影",
                         "recent": [{"title": titles[i - 1], "tokens": tok.item_tokens[i].tolist()} for i in hist_items],
                         "encoder_input": list(item_tok[h].shape), "root_allowed": int(trie.allowed(torch.zeros(1, dtype=torch.long, device=device)).sum()),
                         "beams": [{"tokens": seqs[0, w].tolist(), "score": round(float(sc[0, w]), 3),
                                    "title": titles[dec.map[tuple(seqs[0, w].tolist())][0] - 1] if tuple(seqs[0, w].tolist()) in dec.map else None}
                                   for w in range(seqs.shape[1])],
                         "target": titles[target - 1], "hit": target in lists[0]})
    model.train()


def run_sasrec(ctx, cfg, device) -> None:
    """参照：SASRec 对全部物品打分（和 R01 基线同一套代码）。"""
    from recseq.train import recall_ndcg, scores

    ctx.stage("train")
    model, d, info = I.train_sasrec(ctx, cfg, device)
    ctx.stage("evaluate")
    users, cuts = d.evals["test"]
    if cfg["eval_users"]:
        users, cuts = users[: cfg["eval_users"]], cuts[: cfg["eval_users"]]
    c = I.cf_config(cfg)
    import time
    t0 = time.time()
    rank, _, _ = scores(model, d, users, cuts, c, device, 50)
    ms = (time.time() - t0) * 1000 / len(users)
    test = recall_ndcg(rank)
    test.update({"ms_per_user": round(ms, 4), "invalid_frac": 0.0, "unique_per_user": 10.0})
    ctx.summary({"test": test, "train": {"seconds": info["seconds"], "steps": info["steps"]},
                 "headline": {"recall10": test["recall10"]}, "display": {"Recall@10": test["recall10"], "NDCG@10": test["ndcg10"],
                                                                         "打分 ms/用户": test["ms_per_user"]},
                 "display_pct": ["Recall@10", "NDCG@10"]})
