<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一步训练](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)

# 第 4 步 / 共 16 步：代码导读

> 阶段 1 / 7：**理解任务**

只看三点：解决什么问题、关键的是哪几行、张量形状。

### ① 梯度累积与 GradScaler

外层每步取 batch_size 条，内层拆成 grad_accum 份；loss ÷ accum 再 backward。fp16 + scaler 时 backward 的是 scale × loss，更新前 unscale_、裁剪，scaler.step 遇到 inf 会跳过。

[`tinylm/train.py#L163-L295`](../../../projects/tinylm/train.py#L163-L295)

```python
def train(ctx, cfg: dict, model, train_stream: np.ndarray, train_bytes: int, valid: dict, eos_id: int, device,
          budget_steps: int | None = None) -> dict:
    """token 流切成 [seq_len + 1] 的块，打乱块顺序，每步取 batch_size 块（梯度累积时拆成 grad_accum 份）。"""
    T, B = cfg["seq_len"], cfg["batch_size"]
    accum = int(cfg.get("grad_accum", 1))
    if B % accum:
        raise ValueError(f"batch_size={B} 不能被 grad_accum={accum} 整除")
    mb = B // accum
    n_blocks = (len(train_stream) - 1) // T
    plan = plan_steps(n_blocks, cfg, budget_steps)
    total, starts = plan["total"], plan["order"] * T
    off = cfg.get("target_offset", 1)
    bytes_per_token = train_bytes / max(len(train_stream), 1)
    opt = make_optimizer(model, cfg)
    use_scaler = half_dtype(device, cfg) == torch.float16 and cfg.get("grad_scaler", False)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler) if use_scaler else None
    aux_coef = cfg.get("moe_aux_coef", 0.01)
    eval_at = sorted({max(1, round(total * (i + 1) / cfg["eval_points"])) for i in range(cfg["eval_points"])} | {total})
    log_every = max(1, total // 200)
    probe_at = {0, total // 10, total // 2}                                 # 在这几步统计 logits 梯度的下溢比例
    warm_end = max(1, int(total * cfg["warmup_frac"]))
    ctx.log(f"训练：{total} 步 × {B} × {T} = {total * B * T:,} token（训练流共 {len(train_stream):,} token，用前 "
            f"{plan['n_used'] * T:,} 个 × {plan['epochs']} 遍；每 token {bytes_per_token:.2f} 字节）"
            f"{f'；梯度累积 {accum} × {mb}' if accum > 1 else ''}；在第 {eval_at} 步评估")
    x0, y0 = make_batch(train_stream, starts[:B], T, off)
    batch_check = {"y_eq_x_frac": round(float((x0 == y0).mean()), 4), "x_head": x0[0, :8].tolist(),
                   "y_head": y0[0, :8].tolist()}
    model.train()
    t0 = time.time()
    t_last, tok_since = t0, 0
    curve, tps_log, losses, gnorms, dlogit_zero, moe_tail = [], [], [], [], [], []
    gmax = torch.zeros((), device=device)
    nan_step, loss_warm, skipped, step = None, None, 0, -1
    loss = torch.zeros(())
    for step in range(total):
        lr = lr_at(step, total, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = starts[step * B:(step + 1) * B]
        loss_sum = 0.0
        for k in range(accum):
            x, y = make_batch(train_stream, idx[k * mb:(k + 1) * mb], T, off)
            x = torch.from_numpy(x).to(device, non_blocking=True)                   # [mb, T]
            y = torch.from_numpy(y).to(device, non_blocking=True)                   # [mb, T]
            with autocast(device, cfg):
                logits = model(x)                                                   # [mb, T, V]
            if step in probe_at and k == 0 and logits.requires_grad:
                logits.register_hook(lambda g: dlogit_zero.append(float((g == 0).float().mean())))
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
            aux = model.aux_loss() if hasattr(model, "aux_loss") else None
            total_loss = (loss + aux_coef * aux if aux is not None else loss) / accum
            (scaler.scale(total_loss) if scaler else total_loss).backward()
            loss_sum = loss_sum + loss.detach() / accum
        loss = loss_sum
        if scaler:
            scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"] if cfg["grad_clip"] > 0 else 1e9)
        gmax = torch.maximum(gmax, torch.nan_to_num(gn.detach().float(), nan=0.0, posinf=1e9))
        if scaler:
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        tok_since += B * T
        logged = (step + 1) % log_every == 0 or step + 1 == total or step + 1 == warm_end
        if logged:
            lv = float(loss)
            if not math.isfinite(lv):
                nan_step = step
                ctx.warn("nan_loss", f"第 {step} 步 loss = {lv}，停止训练")
                break
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            tps = tok_since / max(now - t_last, 1e-9)
            tps_log.append(tps)
            losses.append(lv)
            gnorms.append(float(gn))
            if step + 1 == warm_end:
                loss_warm = lv
            extra = {}
            ms = model.moe_stats() if hasattr(model, "moe_stats") else None
            if ms:
                extra["moe_max_load"] = round(ms["max_over_mean"], 4)
                if step >= 0.95 * total:
                    moe_tail.append(ms)
            if scaler:
                extra["scaler_scale"] = scaler.get_scale()
            ctx.metric("train_loss", lv, step=step + 1, lr=lr, grad_norm=float(gn), tokens_per_s=round(tps),
                       bytes_per_s=round(tps * bytes_per_token), **extra)
            ctx.metric("grad_norm", round(float(gn), 4), step=step + 1)
            for k, v in extra.items():
                ctx.metric(k, v, step=step + 1)
            ctx.check()
        if step + 1 in eval_at:
            ev = evaluate(model, valid["stream"], valid["bytes"], eos_id, cfg, device)
            ctx.metric("valid_loss", ev["loss"], step=step + 1, split="valid")
            ctx.metric("valid_bpb", ev["bpb"], step=step + 1, split="valid")
            curve.append({"step": step + 1, **ev, "train_tokens": (step + 1) * B * T})   # ev 里的 tokens 是验证集 token 数
            ctx.log(f"step {step + 1}/{total} train_loss {float(loss):.4f} valid loss {ev['loss']:.4f} bpb {ev['bpb']:.4f}")
        if logged or step + 1 in eval_at:  # 评估耗时不算进吞吐
            t_last, tok_since = time.time(), 0
    seconds = time.time() - t0
    done = step + 1 if nan_step is None else nan_step
    tps_med = float(np.median(tps_log)) if tps_log else done * B * T / max(seconds, 1e-9)
    tail = losses[-max(1, len(losses) // 20):]
    spikes = sum(1 for i in range(10, len(losses)) if losses[i] > 1.2 * float(np.median(losses[i - 10:i])))
    stats = {"loss_at_warmup_end": round(loss_warm, 5) if loss_warm is not None else None,
             "train_loss_tail": round(float(np.mean(tail)), 5) if tail else None, "loss_spikes": spikes,
             "grad_norm_max": round(float(gmax), 4), "grad_norm_median": round(float(np.median(gnorms)), 5) if gnorms else None,
             "nan_step": nan_step, "steps_done": done, "micro_batch": mb, "grad_accum": accum,
             "epochs": plan["epochs"], "unique_tokens": plan["n_used"] * T, "budget_capped": plan["budget_capped"],
             "batch_check": batch_check}
    precision = {"dtype": str(half_dtype(device, cfg) or "float32").replace("torch.", ""),
                 "dlogits_zero_frac": round(float(np.mean(dlogit_zero)), 5) if dlogit_zero else None,
                 "grad_scaler": bool(scaler), "scaler_scale_final": scaler.get_scale() if scaler else None,
                 "skipped_steps": skipped if scaler else None}
    moe = None
    if moe_tail:
        load = np.mean([np.mean(m["load"], axis=0) for m in moe_tail], axis=0)        # [E]，最后 5% 步、各层平均
        E = len(load)
        moe = {"load_frac": [round(float(v), 4) for v in load],
               "load_max_over_mean": round(float(np.mean([m["max_over_mean"] for m in moe_tail])), 4),
               "dead_experts": int(np.mean([sum(v < 0.1 / E for v in np.mean(m["load"], axis=0)) for m in moe_tail]) + 0.5),
               "router_entropy_norm": round(float(np.mean([m["entropy"] for m in moe_tail])), 4)}
    return {"steps": total, "seconds": round(seconds, 1), "train_tokens": done * B * T,
            "train_bytes_seen": round(done * B * T * bytes_per_token), "bytes_per_token": round(bytes_per_token, 4),
            "tokens_per_s": round(tps_med), "bytes_per_s": round(tps_med * bytes_per_token),
            "final_train_loss": round(float(loss), 5), "valid_curve": curve, "stats": stats, "precision": precision,
            "moe": moe}
```

### ② warmup 与余弦

warm = total × warmup_frac；warmup_frac = 0 时 warm 取 1，第一步就是峰值。

[`tinylm/train.py#L32-L40`](../../../projects/tinylm/train.py#L32-L40)

```python
def lr_at(step: int, total: int, cfg: dict) -> float:
    """线性 warmup，然后余弦降到峰值的 10%（cosine）或保持峰值不变（constant）。"""
    warm = max(1, int(total * cfg["warmup_frac"]))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    if cfg.get("lr_schedule", "cosine") == "constant":
        return cfg["lr"]
    p = (step - warm) / max(1, total - warm)
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))
```

### ③ 文档 mask

doc = EOS 的累计计数 [B, T]；mask = (doc_i == doc_j) & 因果 [B, 1, T, T]；pos = 到本文档开头的距离，喂给 RoPE。显式 mask 让 SDPA 走不了 flash kernel。

[`tinylm/model.py#L62-L72`](../../../projects/tinylm/model.py#L62-L72)

```python
def doc_layout(idx: torch.Tensor, eos_id: int):
    """按分隔符分段：idx [B, T] → 文档编号 doc [B, T]、文档内位置 pos [B, T]、注意力 mask [B, 1, T, T]（True = 可见）。
    token 流是 [EOS, 文档1, EOS, 文档2, ...]，EOS 当作下一篇文档的开头（它负责预测下一篇的第一个 token）。"""
    B, T = idx.shape
    doc = (idx == eos_id).long().cumsum(dim=1)                                  # [B, T]
    ar = torch.arange(T, device=idx.device).expand(B, T)
    start = torch.where(idx == eos_id, ar, torch.zeros_like(ar)).cummax(dim=1).values   # 每个位置所在文档的起点
    pos = ar - start                                                            # [B, T]
    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    mask = (doc[:, :, None] == doc[:, None, :]) & causal                        # [B, T, T]
    return doc, pos, mask[:, None]
```

### ④ 注意力

mask 为 None 时用 is_causal（flash 路径），否则传 attn_mask（稠密 [T, T] mask）。

[`tinylm/model.py#L143-L158`](../../../projects/tinylm/model.py#L143-L158)

```python
    def attn(self, x, cos, sin, mask):
        B, T, C = x.shape
        if self.R:
            return self.mla(x, cos, sin, mask)
        q, k, v = self.qkv(x).split([C, self.Hkv * self.D, self.Hkv * self.D], dim=-1)
        q = q.view(B, T, self.H, self.D).transpose(1, 2)                       # [B, H, T, D]
        k = k.view(B, T, self.Hkv, self.D).transpose(1, 2)                     # [B, Hkv, T, D]：KV cache 存的就是 k、v
        v = v.view(B, T, self.Hkv, self.D).transpose(1, 2)                     # [B, Hkv, T, D]
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        gqa = self.Hkv != self.H                                                # 每 H / Hkv 个查询头共享一组 KV
        if mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
        else:                                                                   # 显式 mask：走不了 flash kernel
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)
        return a.transpose(1, 2).reshape(B, T, C)                               # [B, T, C]
```

---

[← 追踪一步训练](03-trace.md) · [课程首页](../README.md) · [先估算 →](05-estimate.md)
