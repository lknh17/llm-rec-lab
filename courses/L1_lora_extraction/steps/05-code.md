<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 追踪一条样本](04-trace.md) · [课程首页](../README.md) · [基础模型（不训练） →](06-base-model.md)

# 第 5 步 / 共 15 步：代码导读

> 阶段 3 / 6：**看清内部**

每个环节一段关键代码，取自实际执行的版本（`projects/adextract/`）。读的时候只看三点：这段解决什么问题、关键的是哪几行、张量形状。

### ① 样本：messages

训练和推理调用同一个函数，保证 prompt 一致。

[`adextract/prompts.py#L30-L38`](../../../projects/adextract/prompts.py#L30-L38)

```python
def build_messages(text: str, prompt_mode: str) -> list[dict]:
    """一条广告 -> 聊天消息。训练和推理用同一个函数，保证 prompt 一致。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prompt_mode == "few_shot":
        for ad, target in FEW_SHOT:
            messages.append({"role": "user", "content": ad})
            messages.append({"role": "assistant", "content": to_json(target)})
    messages.append({"role": "user", "content": text})
    return messages
```

### ② 模板、tokenize、监督 mask

prompt 部分 labels = -100；超长样本按 truncation 规则丢弃或截断。

[`adextract/data.py#L58-L84`](../../../projects/adextract/data.py#L58-L84)

```python
def encode_example(tok, text: str, target: dict, cfg: dict) -> dict | None:
    """一条训练样本 -> input_ids 和 labels。超长且 truncation=drop_long 时返回 None（整条丢弃）。"""
    messages = build_messages(text, cfg["prompt_mode"])
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(messages + [{"role": "assistant", "content": to_json(target)}], tokenize=False)
    # 模板一致性：完整对话必须以 prompt 开头，否则"前 n_prompt 个 token 是题目"这个假设就不成立
    assert full.startswith(prompt), "聊天模板的 prompt 不是完整对话的前缀，监督位置会错位"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert input_ids[: len(prompt_ids)] == prompt_ids, "分词边界不一致"
    n_prompt = len(prompt_ids)

    if cfg["supervision"] == "assistant_only":
        labels = [IGNORE] * n_prompt + input_ids[n_prompt:]   # 只监督回答部分
    else:
        labels = list(input_ids)                              # 全部 token 都监督

    truncated = False
    if len(input_ids) > cfg["max_len"]:
        if cfg["truncation"] == "drop_long":
            return None
        # right：直接砍掉尾部。回答在尾部，最先被砍掉的就是监督目标
        input_ids, labels = input_ids[: cfg["max_len"]], labels[: cfg["max_len"]]
        truncated = True
    n_sup = sum(1 for x in labels[1:] if x != IGNORE)  # 第 0 位没有"前一个位置"去预测它，不计入
    return {"input_ids": input_ids, "labels": labels, "n_prompt": n_prompt, "n_supervised": n_sup,
            "truncated": truncated, "prompt_text": prompt, "full_text": full}
```

### ③ LoRA 挂载

LORA_TARGETS 决定挂在哪些线性层；B 初始化为 0。

[`adextract/modeling.py#L57-L73`](../../../projects/adextract/modeling.py#L57-L73)

```python
def attach_lora(model, cfg: dict):
    """冻结全部基础权重，只在指定线性层旁边挂低秩矩阵 A[r, in]、B[out, r]。

    前向：y = W x + (alpha / r) · B A x。B 初始化为 0，所以训练开始时模型输出与基础模型完全一致。
    可训练参数 = Σ r·(in + out)，0.5B 模型 r=16 挂全部 7 类线性层约 880 万，占 1.8%。
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
                        target_modules=LORA_TARGETS[cfg["lora_targets"]], task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    if cfg.get("grad_checkpointing"):
        # 冻结的基础模型 + 检查点：输入 embedding 必须 requires_grad，否则重算段里没有梯度可传
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model
```

### ④ loss 对齐与归一化

错一位对齐；只在被监督位置计算 lm_head；分母是被监督 token 数（clamp 到 1）。

[`adextract/train.py#L15-L42`](../../../projects/adextract/train.py#L15-L42)

```python
def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """位置 t 的 logits 预测位置 t+1 的 token，所以 logits 去掉最后一位、labels 去掉第一位再对齐。

    归一化：被监督 token 的 loss 求和 / 被监督 token 总数（整个 batch 一起平均）。
    长回答的样本在 batch 里权重更大；另一种做法是先每条样本内平均再跨样本平均。
    分母用 clamp(min=1)：没有任何监督 token 时 loss = 0 而不是 nan——训练不会报错，但也什么都没学。
    """
    shift_logits = logits[:, :-1, :]                     # [B, T-1, V]
    shift_labels = labels[:, 1:]                         # [B, T-1]
    n_sup = (shift_labels != IGNORE).sum()
    loss_sum = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)).float(), shift_labels.reshape(-1),
                               ignore_index=IGNORE, reduction="sum")
    return loss_sum / n_sup.clamp(min=1), n_sup


def forward_supervised(model, b: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """只在需要算 loss 的位置计算 lm_head。

    词表 V=151936：完整 logits [8, 320, V] 的 fp32 张量约 1.5 GB，而被监督的只有末尾约 40 个位置。
    start = batch 内最早的被监督位置 - 1（该位置的 logit 预测第一个被监督 token），之前的位置不算 logits。
    all_tokens 监督时 start=0，退化为完整计算。返回 (logits[B, T-start, V], labels[B, T-start], start)。
    """
    labels = b["labels"]                                         # [B, T]
    T = labels.shape[1]
    sup_pos = (labels != IGNORE).any(dim=0).nonzero()
    start = max(int(sup_pos[0]) - 1, 0) if len(sup_pos) else T - 1
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], logits_to_keep=T - start)
    return out.logits, labels[:, start:], start
```

### ⑤ 训练循环

AdamW、warmup + 线性衰减、梯度裁剪 1.0；每步记录 supervised_tokens、grad_norm、tokens/s。

[`adextract/train.py#L73-L134`](../../../projects/adextract/train.py#L73-L134)

```python
def train(ctx, cfg, model, tok, examples: list[dict], dev_examples: list[dict], device, on_epoch_end=None) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]          # 只有 LoRA 的 A、B
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    bs = cfg["batch_size"]
    steps_per_epoch = math.ceil(len(examples) / bs)
    total = steps_per_epoch * cfg["epochs"]
    if cfg["max_steps"]:
        total = min(total, cfg["max_steps"])
    warmup = max(1, int(0.05 * total))
    # 学习率：前 5% 步线性升温，之后线性降到 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    gen = torch.Generator().manual_seed(cfg["seed"])
    ctx.info("train", {"examples": len(examples), "steps_per_epoch": steps_per_epoch, "planned_steps": total,
                       "warmup_steps": warmup})
    b_norm0, base0 = lora_b_norm(model), base_weight_probe(model)
    step, seen_tokens, seen_sup, t_start = 0, 0, 0, time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(examples), generator=gen).tolist()
        for s in range(0, len(order), bs):
            if step >= total:
                break
            batch_ex = [examples[i] for i in order[s: s + bs]]
            b = collate(batch_ex, tok.pad_token_id, device)                       # 每个 [B, T]
            logits, labels, start = forward_supervised(model, b)                  # [B, T-start, V]
            loss, n_sup = causal_lm_loss(logits, labels)
            if step == 0:
                trace_first_batch(ctx, tok, batch_ex, b, logits, labels, start, loss, cfg)
            loss.backward()                                                        # 梯度只流到 LoRA 参数
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)                # 梯度裁剪，防止个别 batch 冲坏
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            loss_val, n_sup_val = loss.item(), int(n_sup.item())
            n_tok = int(b["attention_mask"].sum().item())
            seen_tokens += n_tok
            seen_sup += n_sup_val
            ctx.metric("train_loss", loss_val, step=step, epoch=epoch + 1, supervised_tokens=n_sup_val,
                       tokens=n_tok, lr=sched.get_last_lr()[0], grad_norm=round(float(grad_norm), 4),
                       tokens_per_s=round(seen_tokens / max(time.time() - t_start, 1e-6)))
            del logits, loss
            if device.type == "mps" and step % 10 == 0:
                # 每步序列长度不同，MPS 缓存分配器会为每种形状留一份显存；定期释放，避免占用一路涨到几十 GB
                torch.mps.empty_cache()
            if step % 10 == 0 or step == 1:
                ctx.log(f"step {step}/{total} loss={loss_val:.4f} 监督token={n_sup_val} "
                        f"lr={sched.get_last_lr()[0]:.2e} grad_norm={float(grad_norm):.3f}")
            ctx.check()
        dl = eval_loss(model, dev_examples, tok.pad_token_id, device)
        ctx.metric("dev_loss", dl, step=step, split="dev", epoch=epoch + 1)
        ctx.log(f"epoch {epoch + 1}: dev_loss={dl:.4f}")
        if on_epoch_end:
            on_epoch_end(epoch + 1, step)
        if step >= total:
            break
    result = {"steps": step, "train_tokens": seen_tokens, "supervised_tokens": seen_sup,
              "seconds": round(time.time() - t_start, 1),
              "lora_B_norm_before": b_norm0, "lora_B_norm_after": lora_b_norm(model),
              "base_weight_probe_before": base0, "base_weight_probe_after": base_weight_probe(model)}
    return result
```

### ⑥ 逐条评分

格式、品类、卖点、优惠、条件、无依据，每项的判定规则都在这里。

[`adextract/schema.py#L83-L133`](../../../projects/adextract/schema.py#L83-L133)

```python
def score_one(text: str, pred_raw: str, gold: dict) -> dict:
    """单条样本的评测明细和错误类型。"""
    pred, err = parse_output(pred_raw)
    gold_cond = [g for g in gold["promotions"] if g.get("condition")]
    r = {"format_ok": pred is not None, "format_error": err, "errors": [],
         "n_gold_points": len(gold["selling_points"]), "n_gold_promos": len(gold["promotions"]),
         "n_gold_cond": len(gold_cond), "point_tp": 0, "n_pred_points": 0, "promo_tp": 0, "cond_kept": 0,
         "n_pred_values": 0, "n_ungrounded": 0, "ungrounded_values": []}
    if pred is None:
        r["errors"].append("格式错误")
        r["category_ok"] = False
        return r
    r["category_ok"] = loose_match(pred["category"], gold["category"])
    if not r["category_ok"]:
        r["errors"].append("品类错误" if gold["category"] else "无品类却编造品类")
    pts = _match_lists(pred["selling_points"], gold["selling_points"])
    r["point_tp"], r["n_pred_points"] = len(pts), len(pred["selling_points"])
    if len(pts) < len(gold["selling_points"]):
        r["errors"].append("漏卖点")
    if len(pts) < len(pred["selling_points"]):
        r["errors"].append("多卖点")
    pairs = _match_lists(pred["promotions"], gold["promotions"], key=lambda x: x.get("offer"))
    r["promo_tp"] = len(pairs)
    if len(pairs) < len(gold["promotions"]):
        r["errors"].append("漏优惠")
    if len(pairs) < len(pred["promotions"]):
        r["errors"].append("多优惠（原文无此优惠或把否定句当优惠）")
    for pi, gi in pairs:
        g, p = gold["promotions"][gi], pred["promotions"][pi]
        if g.get("condition"):
            if loose_match(p.get("condition"), g["condition"]):
                r["cond_kept"] += 1
            else:
                r["errors"].append("漏条件")  # 例如把"指定款第二件半价"写成无条件的"第二件半价"
        elif p.get("condition"):
            r["errors"].append("多条件")
    missed_cond_promos = [g for g in gold_cond if not any(gold["promotions"][gi] is g for _, gi in pairs)]
    if missed_cond_promos and "漏条件" not in r["errors"]:
        r["errors"].append("漏条件")
    values = ([pred["category"]] if pred["category"] else []) + list(pred["selling_points"])
    for p in pred["promotions"]:
        values += [p.get("offer")] + ([p["condition"]] if p.get("condition") else [])
    src = norm(text)
    for v in values:
        r["n_pred_values"] += 1
        if norm(v) and norm(v) not in src:
            r["n_ungrounded"] += 1
            r["ungrounded_values"].append(v)
    if r["n_ungrounded"]:
        r["errors"].append("无依据内容")
    return r
```

---

[← 追踪一条样本](04-trace.md) · [课程首页](../README.md) · [基础模型（不训练） →](06-base-model.md)
