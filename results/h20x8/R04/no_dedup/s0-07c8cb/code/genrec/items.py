"""物品向量的来源：协同（SASRec 物品 embedding）、内容（Qwen2.5-0.5B 读片名 + 类型）、随机；以及合成的千万级物品。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from labkit.cache import stage_cache

CF_VERSION = 1
CONTENT_VERSION = 1
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


class _QuietCtx:
    """把内部训练 SASRec 的指标加上 cf_ 前缀写进本运行，日志照常输出。"""

    def __init__(self, ctx):
        self.ctx = ctx

    def metric(self, name, value, step=None, **kw):
        self.ctx.metric(f"cf_{name}", value, step=step, **kw)

    def __getattr__(self, k):
        return getattr(self.ctx, k)


def cf_config(cfg: dict) -> dict:
    """协同向量用的 SASRec：和 R01 基线同一套代码、同一个切分（只用 seq[:-2] 训练），种子固定为 0（物品向量不随课程种子变）。"""
    from recseq.params import PARAMS as RP

    return {**RP.resolve({"d_model": cfg["cf_d_model"], "steps": cfg["cf_steps"], "seed": 0, "probe": False,
                          "device": cfg["device"]}), "max_len_eff": 50, "attn_impl_eff": "sdpa"}


def train_sasrec(ctx, cfg: dict, device) -> tuple[torch.nn.Module, object, dict]:
    from recseq import data as RD
    from recseq.models import build
    from recseq.train import train

    d, _ = RD.load_ml1m()
    c = cf_config(cfg)
    torch.manual_seed(0)
    model = build(d.n_items, c).to(device)
    info = train(_QuietCtx(ctx), c, model, d, device, 50, c["batch_size"])
    return model, d, info


def cf_embeddings(ctx, cfg: dict, device) -> tuple[np.ndarray, dict]:
    key = {"v": CF_VERSION, "d": cfg["cf_d_model"], "steps": cfg["cf_steps"]}

    def build_cf(d: Path) -> None:
        model, _, info = train_sasrec(ctx, cfg, device)
        np.save(d / "emb.npy", model.emb.weight.detach().float().cpu().numpy()[1:])   # 去掉 padding 行
        (d / "info.json").write_text(json.dumps({k: info[k] for k in ("best_valid_recall10", "seconds", "steps")}))

    cdir = stage_cache(ctx, "genrec_cf_emb", key, build_cf)
    return np.load(cdir / "emb.npy"), json.loads((cdir / "info.json").read_text())


# --8<-- [start:content_emb]
def content_embeddings(ctx, titles: list[str], genres: list[str], device) -> np.ndarray:
    """Qwen2.5-0.5B 读 "Title: … Genres: …"，取最后一层隐藏状态，按 attention_mask 做 mean pooling（896 维）。"""
    key = {"v": CONTENT_VERSION, "model": QWEN, "pool": "mean", "template": "Title: {t}\nGenres: {g}"}

    def build_content(d: Path) -> None:
        from transformers import AutoModel, AutoTokenizer

        from labkit.assets import resolve_model_path

        path = resolve_model_path(QWEN)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).eval()
        texts = [f"Title: {t}\nGenres: {g.replace('|', ', ')}" for t, g in zip(titles, genres)]
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), 128):
                b = tok(texts[s:s + 128], padding=True, return_tensors="pt").to(device)
                h = model(**b).last_hidden_state.float()                    # [B, T, 896]
                m = b["attention_mask"][..., None].float()
                out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        np.save(d / "emb.npy", np.concatenate(out))

    return np.load(stage_cache(ctx, "genrec_content_emb", key, build_content) / "emb.npy")
# --8<-- [end:content_emb]


# --8<-- [start:synth]
def synth_items(n: int, dim: int, seed: int, device) -> tuple[torch.Tensor, dict]:
    """三层高斯树：100 个顶簇（大小按 Zipf）× 50 个子簇 × 20 个微簇，再加噪声；各维方差不同，范数按对数正态缩放。
    在设备上按种子确定性生成（千万级不落盘）。返回 (x [N, dim], 真实簇标签)。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    T, S, M = 100, 50, 20
    scale = torch.exp(torch.randn(dim, generator=g) * 0.5)                 # 各维方差不同
    top = torch.randn(T, dim, generator=g) * 4.0 * scale
    sub = top.repeat_interleave(S, 0) + torch.randn(T * S, dim, generator=g) * 2.0 * scale
    mic = sub.repeat_interleave(M, 0) + torch.randn(T * S * M, dim, generator=g) * 1.0 * scale
    zipf = 1.0 / torch.arange(1, T + 1, dtype=torch.float64)
    top_id = torch.multinomial(zipf / zipf.sum(), n, replacement=True, generator=g)
    sub_id = top_id * S + torch.randint(0, S, (n,), generator=g)
    mic_id = sub_id * M + torch.randint(0, M, (n,), generator=g)
    noise_g = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.empty(n, dim, device=device)
    for s in range(0, n, 1_000_000):                                        # 分块生成，避免 CPU 上一次建 1e7 × dim
        e = min(n, s + 1_000_000)
        blk = mic[mic_id[s:e]] + torch.randn(e - s, dim, generator=noise_g) * 0.5 * scale
        norm = torch.exp(torch.randn(e - s, 1, generator=noise_g) * 0.3)
        x[s:e] = (blk * norm).to(device)
    return x, {"top": top_id.to(device), "sub": sub_id.to(device)}
# --8<-- [end:synth]


def ml1m_meta() -> dict:
    """ML-1M 的物品元信息（和 recseq 一样的物品编号：1..3706）、训练序列（只用 seq[:-2]）。"""
    from recseq import data as RD

    d, _ = RD.load_ml1m()
    G = d.meta["genres"]
    names = d.meta["genre_names"]
    genres = ["|".join(names[j - 1] for j in row if j > 0) for row in G]
    train_seqs = [d.seq(u)[0][: d.n_train[u]] for u in d.train_users]
    pop = np.bincount(np.concatenate(train_seqs), minlength=d.n_items)[1:]
    return {"d": d, "titles": d.meta["titles"], "genres": genres, "genre_ids": G, "pop": pop, "train_seqs": train_seqs}
