"""recseq 执行入口：数据（ML-1M leave-one-out 或合成长序列）→ 成本探针 → 训练（取验证最好的权重）→ 全量排序评估 → 分组、诊断、检查。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from labkit.cache import stage_cache

from . import cost as C
from . import data as D
from . import synth as S
from .models import build, build_mask, param_count, pos_bucket, resolve_impl, time_bucket
from .train import recall_ndcg, sampled_rank, scores, train

SYNTH_VERSION = 3
SYNTH_KEYS = ("synth_users", "synth_eval_users", "synth_clusters", "synth_items_per_cluster", "synth_len_min", "synth_len_max",
              "synth_period_min", "synth_period_max", "synth_burst", "synth_noise", "synth_stay", "synth_alpha",
              "synth_session_mean", "data_seed", "len_scale")
DIST_EDGES = (("lt512", 0, 512), ("d512_2k", 512, 2048), ("d2k_8k", 2048, 8192), ("gt8k", 8192, 10 ** 9))
COVER_LENS = (512, 2048, 8192, 10240)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sc(cfg: dict, n: int) -> int:
    """按 len_scale 缩放一个长度（合成数据才缩放）。"""
    return max(2, int(round(n * cfg["len_scale"]))) if cfg["dataset"] == "synth" else n


# ------------------------------------------------------------------ 数据
def load_synth(ctx, cfg: dict) -> tuple[D.SeqData, dict]:
    key = {"v": SYNTH_VERSION, **{k: cfg[k] for k in SYNTH_KEYS}}

    def build_synth(d: Path) -> None:
        n_eval = cfg["synth_eval_users"]
        parts = {"train": S.generate(cfg, cfg["synth_users"], cfg["data_seed"]),
                 "valid": S.generate(cfg, max(8, n_eval // 4), cfg["data_seed"] + 1),
                 "test": S.generate(cfg, n_eval, cfg["data_seed"] + 2)}
        for name, g in parts.items():
            np.savez(d / f"{name}.npz", **g)
            if name != "train":
                np.savez(d / f"{name}_cut.npz", **S.cut_eval(g, cfg["data_seed"] + 10 + len(name)))
        ctx.log(f"合成数据：训练 {cfg['synth_users']} 个用户、{len(parts['train']['items']):,} 个行为")

    sdir = stage_cache(ctx, "recseq_synth", key, build_synth)
    ld = lambda n: dict(np.load(sdir / f"{n}.npz"))  # noqa: E731
    g = {n: ld(n) for n in ("train", "valid", "test")}
    cuts = {n: ld(f"{n}_cut") for n in ("valid", "test")}
    d = D.from_synth(g["train"], g["valid"], g["test"], cuts["valid"], cuts["test"], S.n_items(cfg))
    Nt = g["test"]["N"]
    info = {"users": int(len(g["train"]["N"])), "eval_users": int(len(Nt)), "n_items": int(d.n_items),
            "rows": int(len(d.items)), "len_median": float(np.median(Nt)), "len_p90": float(np.percentile(Nt, 90)),
            "len_max": int(Nt.max()), "period_median": float(np.median(g["test"]["P"])),
            "long_cover": {f"L{L}": S.long_cover(cuts["test"]["dist"], cuts["test"]["kind"], sc(cfg, L)) for L in COVER_LENS},
            "type_share": {t: round(float((cuts["test"]["kind"] == i).mean()), 4) for i, t in enumerate(S.TYPES)}}
    return d, info


# ------------------------------------------------------------------ 检查
@torch.no_grad()
def checks(model, d: D.SeqData, cfg: dict, device, L: int) -> dict:
    """fp32、eval 模式：
    causal_max_dlogit        改掉最后一个输入，倒数第二个位置的 logits 变了多少（应为 0：看不到未来）
    pad_invariance_max_dlogit 同一批用户"各自不补齐"和"补齐到 L"两种评估，最后一个位置 logits 的最大差（应为 0）
    pad_attn_mass_short       HSTU：短用户最后一个位置的 |A| 落在 padding 上的比例（应为 0）"""
    model.eval()
    users, cuts = d.evals["test"]
    hist = np.minimum(cuts, L)
    short = np.flatnonzero(hist <= max(4, L // 4))[:4]
    pick = np.r_[short, np.flatnonzero(hist > L // 2)[:2]][:6] if len(short) else np.arange(min(6, len(users)))
    out = {"causal_max_dlogit": None, "pad_invariance_max_dlogit": None, "pad_attn_mass_short": None}
    b = D.to_device(D.eval_batch(d, users[pick], cuts[pick], L, cfg["truncate"], pad_to=L), device)
    full = model(b).float()
    b2 = {**b, "x": b["x"].clone()}
    b2["x"][:, -1] = torch.randint(1, d.n_items, (b["x"].shape[0],), device=device)
    out["causal_max_dlogit"] = round(float((model(b2).float()[:, -2] - full[:, -2]).abs().max()), 6)
    last = full[:, -1]
    diffs = []
    for i, (u, c) in enumerate(zip(users[pick], cuts[pick])):
        bi = D.to_device(D.eval_batch(d, np.array([u]), np.array([c]), L, cfg["truncate"]), device)
        diffs.append(float((model(bi).float()[0, -1] - last[i]).abs().max()))
    out["pad_invariance_max_dlogit"] = round(max(diffs), 6)
    if cfg["model"] == "hstu" and len(short):
        bs = D.to_device(D.eval_batch(d, users[short], cuts[short], L, cfg["truncate"], pad_to=L), device)
        model(bs)
        a = model.layers[0].last_attn.float()                               # [B, L]
        pad = (bs["x"] == 0).float()
        out["pad_attn_mass_short"] = round(float(((a * pad).sum(1) / a.sum(1).clamp(min=1e-12)).mean()), 4)
    model.train()
    return out


# ------------------------------------------------------------------ 主流程
def run(ctx, cfg: dict) -> None:
    ctx.stage("prepare")
    torch.manual_seed(cfg["seed"])
    synth = cfg["dataset"] == "synth"
    L = sc(cfg, cfg["max_len"])
    cfg = {**cfg, "max_len_eff": L, "attn_impl_eff": resolve_impl(cfg)}
    B = cfg["batch_size"] or max(1, cfg["tokens_per_step"] // L)
    if synth:
        d, info = load_synth(ctx, cfg)
        lens_for_waste = d.off[1:][d.evals["test"][0]] - d.off[:-1][d.evals["test"][0]]
    else:
        d, info = D.load_ml1m()
        lens_for_waste = d.evals["test"][1]
    info.update(S.pad_waste(np.asarray(lens_for_waste), L))
    ctx.info("data", {"kind": "合成长序列" if synth else "MovieLens-1M 评分序列（leave-one-out）", **info,
                      "max_len": L, "truncate": cfg["truncate"]})
    V = d.n_items
    device = pick_device(cfg["device"])

    # ---------------- 追踪 ----------------
    tb = D.train_batch(d, np.random.default_rng(0), 2, L, cfg["truncate"])
    ctx.trace("batch", {"note": "训练窗口：左补齐（最近的行为在最右边），y 是每个位置的下一个物品", "B": B, "L": L,
                        "x_shape": list(tb["x"].shape), "x_tail": tb["x"][0, -8:].tolist(), "y_tail": tb["y"][0, -8:].tolist(),
                        "pad_frac": round(float((tb["x"] == 0).mean()), 3)})
    users, cuts = d.evals["test"]
    if synth:
        u = int(users[0])
        o, N = d.off[u], d.length(u)
        burst, st = d.meta["burst"][o:o + N], d.meta["is_start"][o:o + N]
        tail = [{"item": int(d.items[o + i]), "kind": "回访入口 τ" if burst[i] == 1 else (f"回访第 {burst[i] - 1} 个（回访簇）" if burst[i] > 1 else "背景"),
                 "session_start": bool(st[i])} for i in range(max(0, int(cuts[0]) - 12), int(cuts[0]))]
        ctx.trace("synth_user", {"note": "一个测试用户切点前的 12 个行为和它的目标", "N": int(N), "P": int(d.meta["P"][u]),
                                 "burst_starts": np.flatnonzero(burst == 1).tolist()[-4:], "cut": int(cuts[0]),
                                 "target": int(d.items[o + cuts[0]]), "target_kind": S.TYPES[int(d.meta["test_kind"][0])],
                                 "dist": int(d.meta["test_dist"][0]), "tail": tail})
    else:
        n_hist = cuts
        u_i = int(np.flatnonzero(n_hist > 2 * L)[0]) if (n_hist > 2 * L).any() else 0
        u, c = int(users[u_i]), int(cuts[u_i])
        items, _ = d.seq(u)
        title = lambda it: d.meta["titles"][int(it) - 1]  # noqa: E731
        ra, rb = D.truncate(c, L, "recent")
        ea, eb = D.truncate(c, L, "earliest")
        ctx.trace("user", {"note": "一个历史很长的测试用户：两种截断方式留下的输入末尾，和要预测的目标",
                           "history_len": c, "L": L, "target": title(items[c]),
                           "recent_tail": [title(x) for x in items[ra:rb][-5:]], "earliest_tail": [title(x) for x in items[ea:eb][-5:]],
                           "recent_gap": c - (rb - 1), "earliest_gap": c - (eb - 1)})
    vm = torch.tensor([[False, False, True, True, True, True]])
    ctx.trace("mask", {"note": "6 个位置、左补 2 个 pad：1 = query（行）能看到 key（列）",
                       "causal+pad": build_mask(vm, 0, 6, "causal+pad")[0, 0].int().tolist(),
                       "causal_only": build_mask(vm, 0, 6, "causal_only")[0, 0].int().tolist()})
    dts = torch.tensor([30, 120, 3600, 86400, 30 * 86400, 365 * 86400])
    dists = torch.tensor([1, 5, 15, 16, 100, 1000, 5000, 10000])
    ctx.trace("rab", {"note": "相对时间 → 桶（⌊log2(1+Δt)⌋）；相对距离 → 桶（0-15 精确，之后对数）",
                      "dt_seconds": dts.tolist(), "time_bucket": time_bucket(dts, cfg["time_buckets"]).tolist(),
                      "distance": dists.tolist(), "pos_bucket": pos_bucket(dists, cfg["pos_buckets"], max(L, 10240)).tolist()})

    # ---------------- 成本 ----------------
    est = C.estimate(cfg, V, L)
    probes = {}
    if cfg["probe"]:
        impls = {"sasrec": ("manual", "sdpa"), "hstu": ("manual", "chunked")}.get(cfg["model"], ("none",))
        for nominal in [int(x) for x in str(cfg["probe_lens"]).split(",") if x.strip()]:
            Lp = sc(cfg, nominal)
            probes[f"L{nominal}"] = {impl: C.probe(cfg, V, Lp, impl, device) for impl in impls}
            ctx.check()
        jl = C.lengths_sample(np.minimum(np.asarray(lens_for_waste), L), 6, seed=1)
        probes["jagged"] = C.jagged_probe(cfg, V, jl, device)
    ctx.trace("probe", {"note": "随机输入、每条序列单独测前向 + 反向（B = 1）", **probes})
    cost = {"flops": {k: est[k] for k in ("linear_per_token", "attn_per_token", "attn_share")},
            "infer_macs_per_user": est["infer_macs_per_user"], "sdpa_backend": C.sdpa_backend(device), **probes}

    # ---------------- 训练 ----------------
    model = build(V, cfg).to(device)
    pc = param_count(model)
    ctx.info("model", {"arch": {"mean": "历史均值 pooling", "din": "DIN target attention", "sasrec": "SASRec 因果自注意力",
                                "hstu": "HSTU（逐点 SiLU 注意力 + 相对位置 / 时间偏置）"}[cfg["model"]],
                       "params": pc, "attn_impl": cfg["attn_impl_eff"], "device": str(device),
                       "dtype": "bf16 autocast" if device.type == "cuda" else "float32"})
    ctx.log(f"模型 {cfg['model']}：{pc['total']:,} 参数；L = {L}，B = {B}，注意力实现 {cfg['attn_impl_eff']}")
    ctx.stage("train")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tr = train(ctx, cfg, model, d, device, L, B)
    tr["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None
    ctx.info("train", {k: tr[k] for k in ("steps", "seconds", "samples_per_s", "tokens_per_s", "train_tokens")})

    # ---------------- 评估 ----------------
    ctx.stage("evaluate")
    rank, tgt, kept = scores(model, d, users, cuts, cfg, device, L)
    test = recall_ndcg(rank)
    test["sampled100"] = recall_ndcg(sampled_rank(model, d, users, cuts, cfg, device, L, cfg["sampled_negs"]))
    vrank, _, _ = scores(model, d, *d.evals["valid"], cfg, device, L)
    valid = recall_ndcg(vrank)
    # 参照线：按训练数据里的物品热度排序
    tr_items = np.concatenate([d.seq(int(u))[0][: d.n_train[u]] for u in d.train_users])
    pop = np.bincount(tr_items, minlength=V).astype(np.float64)
    pop[0] = -1
    prank = np.array([(pop > pop[t]).sum() for t in tgt])
    test["pop"] = recall_ndcg(prank)
    diag = {}
    if synth:
        kinds, dist = d.meta["test_kind"], d.meta["test_dist"]
        for i, t in enumerate(S.TYPES):
            test[t] = recall_ndcg(rank[kinds == i])
        test["long_by_dist"] = {name: recall_ndcg(rank[(kinds == 0) & (dist >= sc(cfg, lo)) & (dist < sc(cfg, hi))])
                                for name, lo, hi in DIST_EDGES}
        test["long_by_dist"]["unseen"] = recall_ndcg(rank[(kinds == 0) & (dist < 0)])   # 历史里从没出现过
        rl, rr, rn = (test[t]["recall10"] or 0.0 for t in S.TYPES)
        test["mix"] = {f"r{int(rho * 100):02d}": {"recall10": round(rho * rl + (1 - rho) * (2 / 3 * rr + 1 / 3 * rn), 5)}
                       for rho in (0.05, 0.2, 0.5)}
        Nu = d.off[1:][users] - d.off[:-1][users]
        test["by_len"] = {"short": recall_ndcg(rank[Nu < sc(cfg, 2048)]), "long": recall_ndcg(rank[Nu > sc(cfg, 8192)])}
    else:
        test["by_hist"] = {"le_L": recall_ndcg(rank[cuts <= L]), "gt_L": recall_ndcg(rank[cuts > L])}
        gaps, days, overlap = [], [], []
        G = d.meta["genres"]
        for (a, b), u, c in zip(kept, users, cuts):
            items, ts = d.seq(int(u))
            gaps.append(c - (b - 1))
            days.append((ts[c] - ts[b - 1]) / 86400)
            tg = set(G[items[c] - 1][G[items[c] - 1] > 0].tolist())
            hg = set(G[items[max(a, b - 5):b] - 1].ravel().tolist()) - {0}
            overlap.append(len(tg & hg) / max(1, len(tg | hg)))
        diag = {"gap_mean": round(float(np.mean(gaps)), 3), "gap_days_median": round(float(np.median(days)), 3),
                "recent_genre_overlap": round(float(np.mean(overlap)), 4)}
    chk = checks(model, d, cfg, device, L)
    disp = {"Recall@10": test["recall10"], "NDCG@10": test["ndcg10"], "采样 100 Recall@10": test["sampled100"]["recall10"],
            "热度参照 Recall@10": test["pop"]["recall10"]}
    if synth:
        disp.update({"长程题 Recall@10": test["long"]["recall10"], "近期题 Recall@10": test["recent"]["recall10"],
                     "新会话题 Recall@10": test["newsess"]["recall10"]})
    disp.update({"训练 tokens/s（千）": round(tr["tokens_per_s"] / 1e3, 1), "训练峰值 GB": tr["peak_gb"]})
    ctx.summary({
        "test": test, "valid": valid, "diag": diag, "data": info, "train": tr, "cost": cost, "checks": chk, "params": pc,
        "config_eff": {"max_len": L, "batch_size": B, "attn_impl": cfg["attn_impl_eff"]},
        "headline": {"recall10": test["recall10"], "ndcg10": test["ndcg10"], "sampled100_recall10": test["sampled100"]["recall10"]},
        "display": disp, "display_pct": [k for k in disp if "Recall" in k],
    })
    ctx.log(f"测试 Recall@10 {test['recall10']}  NDCG@10 {test['ndcg10']}  采样 100 Recall@10 {test['sampled100']['recall10']}"
            f"  热度参照 {test['pop']['recall10']}")
    path = ctx.artifact("model.pt")
    torch.save({"state_dict": model.state_dict(), "V": V}, path)
    ctx.register_artifact("model.pt", "checkpoint", "序列模型权重（验证最好的那一步）")
    (ctx.artifact("cfg.json")).write_text(json.dumps({k: cfg[k] for k in cfg if not k.endswith("_eff")}, ensure_ascii=False))
