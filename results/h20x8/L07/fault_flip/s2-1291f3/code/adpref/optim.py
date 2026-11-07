"""偏好阶段的两个训练循环：DPO（离线偏好对）和 GRPO（在线采样 + 可验证奖励）。

共同的前提：SFT 的 LoRA 已经合并进基础权重（merge_and_unload），在它上面挂一个新的 LoRA 训练。
于是"参考模型 π_ref"= 关掉新 LoRA 的同一个模型（model.disable_adapter()），不需要第二份 7B 权重。
"""

from __future__ import annotations

import math
import random
import time

import torch
import torch.nn.functional as F

from adextract.data import IGNORE, collate, prompt_text
from adextract.train import forward_supervised

from .pairs import encode_response, encode_sampled, sample
from .reward import reward


# --8<-- [start:token_logps]
def token_logps(model, b: dict, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """每个位置"真实下一个 token"的对数概率。返回 (logp[N, L], mask[N, L])，mask 标出被监督（回答）的位置。

    forward_supervised 只在回答附近算 lm_head（V=151936，整段 logits 太大）；
    交叉熵 = -log p(真实 token)，reduction="none" 不需要另存一份 [N, L, V] 的 log_softmax。
    """
    logits, labels, _ = forward_supervised(model, b)          # [N, T-start, V], [N, T-start]
    lg = logits[:, :-1].float() / temperature                  # 位置 t 预测 t+1
    lab = labels[:, 1:]                                        # [N, L]
    mask = lab != IGNORE
    nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), lab.clamp(min=0).reshape(-1), reduction="none")
    return -nll.view(lab.shape) * mask, mask
# --8<-- [end:token_logps]


def _schedule(opt, total: int):
    warmup = max(1, int(0.05 * total))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))


# --8<-- [start:dpo]
def dpo_loss(pc, pr, rc, rr, beta: float):
    """pc/pr：策略模型对 chosen/rejected 的序列对数概率（回答 token 求和）；rc/rr：参考模型的，形状都是 [B]。

    隐式奖励 r = β·(log π − log π_ref)。loss = −log σ(r_chosen − r_rejected)：只要求 chosen 比 rejected
    "相对参考模型涨得更多"——两者可以一起降，只要 rejected 降得更快。这就是 DPO 训练中 chosen 对数概率
    也常常下降的原因。
    """
    r_c, r_r = beta * (pc - rc), beta * (pr - rr)
    loss = -F.logsigmoid(r_c - r_r)
    return loss.mean(), r_c.detach(), r_r.detach()


def train_dpo(ctx, cfg, model, tok, pairs: list[dict], device) -> dict:
    enc = [(encode_response(tok, p["text"], p["chosen"], cfg), encode_response(tok, p["text"], p["rejected"], cfg))
           for p in pairs]
    micro = 4                                                     # 每次前向 4 对 = 8 条序列，梯度累积到一个 batch
    # 参考模型的对数概率训练前一次算好：π_ref 冻结，同一个偏好对在每一轮的参考值都一样
    ref = []
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for s in range(0, len(enc), micro):
            chunk = enc[s: s + micro]
            b = collate([c for c, _ in chunk] + [r for _, r in chunk], tok.pad_token_id, device)   # [2m, T]
            lp, _ = token_logps(model, b)
            seq = lp.sum(-1).tolist()
            ref += list(zip(seq[: len(chunk)], seq[len(chunk):]))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["pref_lr"], weight_decay=0.0)
    bs = cfg["pref_batch_size"]
    total = max(1, math.ceil(len(enc) / bs) * cfg["pref_epochs"]) if enc else 0
    sched = _schedule(opt, max(total, 1))
    gen = torch.Generator().manual_seed(cfg["seed"])
    step, t0, seen_tok = 0, time.time(), 0
    ctx.info("pref", {"planned_steps": total, "n_pairs": len(enc)})  # 不能用 pairs：那是偏好对统计（build_pairs）
    for epoch in range(cfg["pref_epochs"] if enc else 0):
        order = torch.randperm(len(enc), generator=gen).tolist()
        for s in range(0, len(order), bs):
            idx = order[s: s + bs]
            stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "dc": 0.0, "dr": 0.0}
            for m in range(0, len(idx), micro):
                mi = idx[m: m + micro]
                b = collate([enc[i][0] for i in mi] + [enc[i][1] for i in mi], tok.pad_token_id, device)  # [2m, T]
                lp, _ = token_logps(model, b)
                seq = lp.sum(-1)                                                                          # [2m]
                pc, pr = seq[: len(mi)], seq[len(mi):]
                rc = torch.tensor([ref[i][0] for i in mi], device=device)
                rr = torch.tensor([ref[i][1] for i in mi], device=device)
                loss, r_c, r_r = dpo_loss(pc, pr, rc, rr, cfg["dpo_beta"])
                (loss * len(mi) / len(idx)).backward()
                w = len(mi) / len(idx)
                stats["loss"] += loss.item() * w
                stats["acc"] += (r_c > r_r).float().mean().item() * w
                stats["margin"] += (r_c - r_r).mean().item() * w
                stats["dc"] += (pc.detach() - rc).mean().item() * w
                stats["dr"] += (pr.detach() - rr).mean().item() * w
                seen_tok += int(b["attention_mask"].sum().item())
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ctx.metric("dpo_loss", stats["loss"], step=step, epoch=epoch + 1, grad_norm=round(float(gn), 4),
                       pref_tokens_per_s=round(seen_tok / max(time.time() - t0, 1e-6)))
            ctx.metric("dpo_reward_acc", stats["acc"], step=step)
            ctx.metric("dpo_margin", stats["margin"], step=step)
            ctx.metric("chosen_logp_delta", stats["dc"], step=step)
            ctx.metric("rejected_logp_delta", stats["dr"], step=step)
            if step % 10 == 0 or step == 1:
                ctx.log(f"DPO step {step}/{total} loss={stats['loss']:.4f} 偏好准确率={stats['acc']:.2f} "
                        f"Δlogπ(chosen)={stats['dc']:+.2f} Δlogπ(rejected)={stats['dr']:+.2f}")
            ctx.check()
    return {"method": "dpo", "steps": step, "seconds": round(time.time() - t0, 1), "tokens": seen_tok,
            "beta": cfg["dpo_beta"]}
# --8<-- [end:dpo]


# --8<-- [start:grpo]
def grpo_step_loss(lp, ref_lp, mask, adv, kl_coef: float):
    """lp/ref_lp：[N, L] 逐 token 对数概率（策略 / 参考），mask：[N, L]，adv：[N] 组内归一化优势。

    每批样本只更新一次，所以重要性比 π/π_old 恒等于 1（数值上），PPO 的裁剪不会生效——
    loss 退化成"优势加权的对数似然"：−A·Σ log π / |o|。KL 用 k3 估计：exp(ref−lp) − (ref−lp) − 1 ≥ 0。
    """
    n_tok = mask.sum(-1).clamp(min=1)                                  # [N]
    ratio = torch.exp(lp - lp.detach())                                # 值为 1，梯度 = ∇log π
    d = (ref_lp - lp) * mask
    kl = (torch.exp(d) - d - 1) * mask                                 # [N, L]
    per_tok = -(adv[:, None] * ratio * mask) + kl_coef * kl
    loss = (per_tok.sum(-1) / n_tok).mean()
    return loss, (kl.sum(-1) / n_tok).mean().detach()


def train_grpo(ctx, cfg, model, tok, rows: list[dict], device) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["pref_lr"], weight_decay=0.0)
    bs, G = cfg["pref_batch_size"], cfg["pref_group"]
    total = max(1, math.ceil(len(rows) / bs))
    sched = _schedule(opt, total)
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)
    micro = 8                                                          # 每次前向 8 条序列，梯度累积
    t0, gen_s, train_s, n_zero, n_groups = time.time(), 0.0, 0.0, 0, 0
    ctx.info("pref", {"planned_steps": total, "prompts": len(rows), "group": G})
    for step in range(1, total + 1):
        batch = [rows[i] for i in order[(step - 1) * bs: step * bs]]
        prompts = [prompt_text(tok, r["text"], cfg) for r in batch]
        groups, secs = sample(model, tok, prompts, device, G, cfg["pref_temperature"], cfg["max_new_tokens"])
        gen_s += secs
        t1 = time.time()
        seqs, advs, rws, lens = [], [], [], []
        for r, p, g in zip(batch, prompts, groups):
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            rs = torch.tensor([reward(r["text"], x["output"], r["gold"], cfg["reward"])[0] for x in g])
            std = rs.std(unbiased=False)
            n_groups += 1
            if std < 1e-6:
                n_zero += 1                                            # 组内奖励全相同：优势全为 0，没有学习信号
            a = (rs - rs.mean()) / (std + 1e-4)
            for x, ai in zip(g, a.tolist()):
                seqs.append(encode_sampled(p_ids, x["ids"]))
                advs.append(ai)
                lens.append(x["new_tokens"])
            rws += rs.tolist()
        kl_sum, loss_sum = 0.0, 0.0
        for m in range(0, len(seqs), micro):
            b = collate(seqs[m: m + micro], tok.pad_token_id, device)                  # [m, T]
            adv = torch.tensor(advs[m: m + micro], device=device)
            with torch.no_grad(), model.disable_adapter():
                ref_lp, _ = token_logps(model, b, cfg["pref_temperature"])
            lp, mask = token_logps(model, b, cfg["pref_temperature"])
            loss, kl = grpo_step_loss(lp, ref_lp, mask, adv, cfg["grpo_kl"])
            w = adv.numel() / len(seqs)
            (loss * w).backward()
            loss_sum += loss.item() * w
            kl_sum += kl.item() * w
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        train_s += time.time() - t1
        zero_frac = sum(1 for k in range(0, len(rws), G) if max(rws[k:k + G]) - min(rws[k:k + G]) < 1e-6) / max(len(batch), 1)
        ctx.metric("grpo_reward", sum(rws) / len(rws), step=step, grad_norm=round(float(gn), 4))
        ctx.metric("grpo_zero_adv_frac", zero_frac, step=step)
        ctx.metric("grpo_kl", kl_sum, step=step)
        ctx.metric("grpo_loss", loss_sum, step=step)
        ctx.metric("grpo_resp_len", sum(lens) / len(lens), step=step)
        if step % 5 == 0 or step == 1:
            ctx.log(f"GRPO step {step}/{total} 奖励均值={sum(rws) / len(rws):.3f} 无信号组={zero_frac:.2f} "
                    f"KL={kl_sum:.4f} 生成 {secs:.1f}s")
        ctx.check()
    return {"method": "grpo", "steps": total, "seconds": round(time.time() - t0, 1), "gen_seconds": round(gen_s, 1),
            "train_seconds": round(train_s, 1), "gen_share": round(gen_s / max(gen_s + train_s, 1e-6), 4),
            "zero_adv_groups": n_zero, "groups": n_groups, "zero_adv_frac": round(n_zero / max(n_groups, 1), 4),
            "reward": cfg["reward"], "kl_coef": cfg["grpo_kl"]}
# --8<-- [end:grpo]
