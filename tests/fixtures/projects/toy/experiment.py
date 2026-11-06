"""一维逻辑回归：x ~ U(-1, 1)，y = [x > 0]，按 noise 翻转标签。leak=True 时第二个特征就是标签本身。"""

import math
import random


def run(ctx, cfg: dict) -> None:
    ctx.stage("prepare")
    rng = random.Random(cfg["seed"])
    rows = []
    for _ in range(cfg["rows"]):
        x = rng.uniform(-1, 1)
        y = int(x > 0) if rng.random() >= cfg["noise"] else int(x <= 0)
        rows.append((x, float(y) if cfg["leak"] else 0.0, y))
    cut = int(len(rows) * 0.8)
    train, valid = rows[:cut], rows[cut:]
    ctx.info("data", {"rows": len(rows), "train_rows": len(train), "valid_rows": len(valid)})
    ctx.info("checks", {"data": {"label_in_features": bool(cfg["leak"])}})

    ctx.stage("train")
    w0 = w1 = b = 0.0
    step = 0
    for _ in range(cfg["epochs"]):
        for x0, x1, y in train:
            p = 1 / (1 + math.exp(-(w0 * x0 + w1 * x1 + b)))
            g = p - y
            w0 -= cfg["lr"] * g * x0
            w1 -= cfg["lr"] * g * x1
            b -= cfg["lr"] * g
            step += 1
        ctx.metric("train_loss", -math.log(max(1e-9, p if y else 1 - p)), step=step)
        ctx.check()

    ctx.stage("eval")
    acc = sum(int((w0 * x0 + w1 * x1 + b > 0) == bool(y)) for x0, x1, y in valid) / len(valid)
    ctx.summary({"valid": {"acc": acc}})
