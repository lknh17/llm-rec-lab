<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [先估算 →](04-estimate.md)

# 第 3 步 / 共 17 步：代码导读

> 阶段 1 / 7：**理解任务**

偏好阶段的全部新代码在 `adpref/` 下（SFT、数据、评测复用 `adextract/`）。读的时候只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 规则构造的 rejected

有条件就删条件，没条件就塞一个原文没有的卖点：负例覆盖什么错误，由这几行决定。

[`adpref/pairs.py#L71-L83`](../../../projects/adpref/pairs.py#L71-L83)

```python
def rule_rejected(row: dict, rng: random.Random) -> tuple[dict, str]:
    """在标准答案上做一处典型错误。返回 (rejected, 错误类型)。"""
    bad = copy.deepcopy(row["gold"])
    conds = [p for p in bad["promotions"] if p.get("condition")]
    if conds:
        rng.choice(conds)["condition"] = None             # 漏条件："指定款第二件半价"写成"第二件半价"
        return bad, "drop_condition"
    extra = _ungrounded_point(row["text"], rng)
    if extra:
        bad["selling_points"] = bad["selling_points"] + [extra]   # 无依据：原文没有的卖点
        return bad, "add_ungrounded"
    bad["category"] = None if bad["category"] else "跑鞋"
    return bad, "wrong_category"
```

### ② 在线采样

num_return_sequences=G；生成的 token 截到第一个停止符，GRPO 直接对这串 token 求对数概率。

[`adpref/pairs.py#L27-L60`](../../../projects/adpref/pairs.py#L27-L60)

```python
@torch.no_grad()
def sample(model, tok, prompts: list[str], device, n: int, temperature: float, max_new_tokens: int,
           batch_prompts: int = 8) -> tuple[list[list[dict]], float]:
    """每个 prompt 采样 n 条（num_return_sequences=n）。返回 [prompt][n] 的 {output, ids, new_tokens}，以及生成秒数。

    ids 是生成出来的 token（截到第一个停止符，含停止符）——GRPO 直接对这些 token 算对数概率，
    不重新分词，保证"被奖励的"和"被求梯度的"是同一串 token。
    """
    model.eval()
    tok.padding_side = "left"
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    out, secs = [], 0.0
    for s in range(0, len(prompts), batch_prompts):
        chunk = prompts[s: s + batch_prompts]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)   # [b, T_p]
        t0 = time.perf_counter()
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                             top_p=1.0, top_k=0, repetition_penalty=1.0,  # 覆盖 Qwen 默认的 top_k/top_p/重复惩罚：
                             num_return_sequences=n, pad_token_id=tok.pad_token_id,  # 采样分布 = 模型分布 / 温度
                             eos_token_id=stop_ids)                                                    # [b·n, T_p + T_new]
        if device.type == "cuda":
            torch.cuda.synchronize()
        secs += time.perf_counter() - t0
        new = gen[:, enc["input_ids"].shape[1]:].tolist()                                              # [b·n, T_new]
        for i in range(len(chunk)):
            group = []
            for j in range(n):
                ids = new[i * n + j]
                k = next((t + 1 for t, x in enumerate(ids) if x in stop_ids), len(ids))
                group.append({"output": tok.decode(ids[:k], skip_special_tokens=True), "ids": ids[:k], "new_tokens": k})
            out.append(group)
    tok.padding_side = "right"
    model.train()
    return out, secs
```

### ③ 逐 token 对数概率

[N, L]；mask 只留回答位置。DPO 求和成序列对数概率，GRPO 按 token 用。

[`adpref/optim.py#L24-L35`](../../../projects/adpref/optim.py#L24-L35)

```python
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
```

### ④ DPO

参考值训练前算好；每步记录偏好准确率、隐式奖励差、chosen/rejected 相对参考模型的变化。

[`adpref/optim.py#L46-L119`](../../../projects/adpref/optim.py#L46-L119)

```python
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
```

### ⑤ 可验证奖励

两种口径的差别只在哪些维度进了奖励。

[`adpref/reward.py#L29-L43`](../../../projects/adpref/reward.py#L29-L43)

```python
def reward(text: str, output: str, gold: dict, kind: str = "field") -> tuple[float, dict]:
    sc = score_one(text, output, gold)
    if not sc["format_ok"]:
        return 0.0, sc
    if kind == "recall_only":
        pr = sc["point_tp"] / sc["n_gold_points"] if sc["n_gold_points"] else 1.0
        rr = sc["promo_tp"] / sc["n_gold_promos"] if sc["n_gold_promos"] else 1.0
        return round(0.5 * pr + 0.5 * rr, 4), sc
    point_f1 = _f1(sc["point_tp"], sc["n_pred_points"], sc["n_gold_points"])
    promo_f1 = _f1(sc["promo_tp"], _n_pred_promos(output), sc["n_gold_promos"])
    cond = sc["cond_kept"] / sc["n_gold_cond"] if sc["n_gold_cond"] else 1.0
    r = 0.2 * sc["category_ok"] + 0.25 * point_f1 + 0.25 * promo_f1 + 0.3 * cond
    r -= 0.5 * (sc["n_ungrounded"] / sc["n_pred_values"] if sc["n_pred_values"] else 0.0)
    r -= 0.1 if "多条件" in sc["errors"] else 0.0
    return round(min(max(r, 0.0), 1.0), 4), sc
```

### ⑥ GRPO

组内归一化优势；组内奖励全相同时优势为 0（无信号组）；KL 用 k3 估计。

[`adpref/optim.py#L124-L200`](../../../projects/adpref/optim.py#L124-L200)

```python
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
```

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [先估算 →](04-estimate.md)
