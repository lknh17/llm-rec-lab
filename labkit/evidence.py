"""证据摘要 evidence.json：网页交互图和 GitHub 静态 SVG 共用的小文件（单文件 ≤ 200 KB）。

从一个运行目录的原始文件推导，不需要重跑：
- params：可训练参数 / 总参数（run.json 的 model.params）；
- curves：train_loss / grad_norm / lr / tokens_per_s（每条 ≤ 200 点）和 dev_loss（metrics.jsonl）；
- throughput：tokens/s、被监督 tokens/s、每步耗时的中位数，训练分钟数；
- memory：设备显存峰值（worker 写的 resources.json，只有 CUDA 上有）和按"权重 / LoRA 参数+梯度 / 优化器状态 /
  激活与其他"拆分的估算（估算方法写在 breakdown_method 里，网页和 README 都会显示）；
- metrics：summary.json 的 headline 和 dev 关键指标。

    python -m labkit evidence results/            # 给已有结果补 evidence.json
    python -m labkit evidence results/ --check    # 只检查大小和字段
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

MAX_POINTS = 200
MAX_BYTES = 200 * 1024
VERSION = 1
CURVES = ("train_loss", "grad_norm", "lr", "tokens_per_s", "dev_loss", "valid_auc", "valid_logloss")
MAX_CURVES = 24
GB = 1024 ** 3


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _metrics(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _downsample(pts: list[list[float]], limit: int = MAX_POINTS) -> list[list[float]]:
    if len(pts) <= limit:
        return pts
    step = len(pts) / limit
    keep = sorted({int(i * step) for i in range(limit)} | {len(pts) - 1})
    return [pts[i] for i in keep]


def _round(v: Any, nd: int = 5) -> Any:
    return round(v, nd) if isinstance(v, float) else v


def curves(rows: list[dict]) -> dict[str, list[list[float]]]:
    """name → [[step, value], ...]。train_loss 行里附带的 grad_norm / lr / tokens_per_s 也拆成独立曲线。"""
    out: dict[str, list[list[float]]] = {}
    for r in rows:
        name, v = r.get("name"), r.get("value")
        x = r.get("step") if r.get("step") is not None else r.get("epoch")
        # 已知曲线之外，项目自己记的数值指标（dpo_loss、grpo_reward、valid_loss……）也一并收进来
        if isinstance(name, str) and isinstance(v, (int, float)) and not isinstance(v, bool) and x is not None \
                and (name in CURVES or len(out) < MAX_CURVES):
            out.setdefault(name, []).append([x, _round(float(v))])
        if name == "train_loss":
            for extra in ("grad_norm", "lr", "tokens_per_s"):
                if isinstance(r.get(extra), (int, float)):
                    out.setdefault(extra, []).append([x, _round(float(r[extra]), 7)])
        if name == "dev_loss" and r.get("epoch") is not None and isinstance(v, (int, float)):
            out.setdefault("dev_loss_by_epoch", []).append([r["epoch"], _round(float(v))])
    return {k: _downsample(v) for k, v in out.items()}


def throughput(rows: list[dict], meta: dict) -> dict:
    tr = [r for r in rows if r.get("name") == "train_loss"]
    tps = [r["tokens_per_s"] for r in tr if isinstance(r.get("tokens_per_s"), (int, float))]
    sup = []
    step_s = []
    for a, b in zip(tr, tr[1:]):
        dt = (b.get("time") or 0) - (a.get("time") or 0)
        if dt > 0:
            step_s.append(dt)
            if isinstance(b.get("supervised_tokens"), (int, float)):
                sup.append(b["supervised_tokens"] / dt)
    train = meta.get("train") or {}
    return {"tokens_per_s_median": round(statistics.median(tps)) if tps else None,
            "supervised_tokens_per_s_median": round(statistics.median(sup)) if sup else None,
            "step_seconds_median": round(statistics.median(step_s), 3) if step_s else None,
            "train_minutes": round(train["seconds"] / 60, 2) if isinstance(train.get("seconds"), (int, float)) else None,
            "train_tokens": train.get("train_tokens"), "supervised_tokens": train.get("supervised_tokens")}


def _bytes_per_param(dtype: str | None) -> int:
    d = str(dtype or "")
    return 2 if ("bfloat16" in d or "float16" in d) else 4


def memory(meta: dict, resources: dict) -> dict:
    """显存峰值（实测）+ 构成（估算）。LoRA：基础权重按模型 dtype；可训练参数、梯度按 fp32（PEFT 默认把适配器
    升到 fp32），AdamW 两份状态 fp32；峰值减去前三项记为"激活与其他"（含 logits、临时缓冲、分配器碎片）。"""
    model = meta.get("model") or {}
    params = model.get("params") or {}
    if not isinstance(params, dict):
        params = {"total": params, "trainable": params}
    total, trainable = params.get("total"), params.get("trainable") or 0
    out: dict[str, Any] = {"device": resources.get("device") or model.get("device"),
                           "peak_allocated_gb": None, "peak_reserved_gb": None, "device_total_gb": None,
                           "breakdown": None, "breakdown_method": None}
    for k in ("peak_allocated", "peak_reserved", "device_total"):
        if isinstance(resources.get(k + "_bytes"), (int, float)):
            out[k + "_gb"] = round(resources[k + "_bytes"] / GB, 2)
    if not isinstance(total, (int, float)):
        return out
    frozen = total - trainable
    weights = frozen * _bytes_per_param(model.get("dtype"))
    trains = model.get("train_mode", "lora") != "none" and trainable > 0
    train_state = trainable * 4 * 2 if trains else 0   # 参数 + 梯度
    optim = trainable * 4 * 2 if trains else 0         # AdamW m、v
    parts = {"weights_gb": round(weights / GB, 2), "trainable_and_grads_gb": round(train_state / GB, 3),
             "optimizer_gb": round(optim / GB, 3)}
    if out["peak_allocated_gb"] is not None:
        parts["activations_other_gb"] = round(max(out["peak_allocated_gb"] - sum(parts.values()), 0.0), 2)
    out["breakdown"] = parts
    out["breakdown_method"] = ("估算：冻结权重 × dtype 字节；可训练参数与梯度各 4 字节/参数；AdamW m、v 各 4 字节/参数；"
                               "激活与其他 = 实测峰值 − 前三项" + ("" if out["peak_allocated_gb"] is not None
                                                                 else "（本次运行没有显存实测，只列静态部分）"))
    return out


def _display(summary: dict) -> dict:
    """项目自报的展示指标：summary.display = {标签: 数值}，summary.display_pct = [按百分比显示的标签]。

    adextract 之外的项目（tinylm / seqrec / kernels）都用这个约定，网页证据面板和揭晓卡片直接按标签画。"""
    disp = summary.get("display")
    if not isinstance(disp, dict):
        return {}
    keep = {str(k): _round(v) for k, v in disp.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    pct = [k for k in summary.get("display_pct") or [] if k in keep]
    return {"display": keep, "display_pct": pct}


def build(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "run.json")
    summary = _read_json(run_dir / "summary.json")
    resources = _read_json(run_dir / "resources.json")
    rows = _metrics(run_dir)
    params = ((meta.get("model") or {}).get("params")) or {}
    if not isinstance(params, dict):  # 推荐项目只记了总参数量（全部可训练）
        params = {"total": params, "trainable": params}
    dev = summary.get("dev") if isinstance(summary.get("dev"), dict) else {}
    keep_dev = {k: dev.get(k) for k in ("exact_all_ok", "format_ok", "category_acc", "point_f1", "condition_keep",
                                         "ungrounded_rate", "promo_recall", "n") if k in dev}
    if isinstance(dev.get("by_tag"), dict):
        keep_dev["by_tag"] = {t: {"n": v.get("n"), "all_ok": v.get("all_ok")} for t, v in dev["by_tag"].items()
                              if isinstance(v, dict)}
    out = {"version": VERSION, "run_id": meta.get("run_id"), "scenario": meta.get("scenario"), "seed": meta.get("seed"),
           "profile": meta.get("profile"), "status": meta.get("status"),
           "params": {k: params.get(k) for k in ("trainable", "total", "trainable_pct", "trainable_tensors")},
           "lora": (meta.get("model") or {}).get("lora"),
           "data": {k: ((meta.get("data") or {}).get("tokenized") or {}).get(k)
                    for k in ("tokens_total", "supervised_tokens_total", "zero_supervision_examples", "prompt_tokens")},
           "curves": curves(rows), "throughput": throughput(rows, meta), "memory": memory(meta, resources),
           "metrics": {"headline": summary.get("headline"), "dev": keep_dev, **_display(summary)}}
    pref = _pref(summary)
    if pref:
        out["pref"] = pref
    return out


PREF_KEYS = ("method", "steps", "seconds", "gen_seconds", "train_seconds", "gen_share", "zero_adv_frac", "tokens", "beta",
             "reward", "kl_coef")
PAIR_KEYS = ("source", "prompts", "pairs", "pair_yield", "group_all_correct_frac", "sample_reward_mean", "gen_seconds",
             "flipped", "kinds")


def _pref(summary: dict) -> dict:
    """偏好阶段（adpref）：方法、步数、耗时与生成占比、偏好对数量与产出率。"""
    p = summary.get("pref") if isinstance(summary.get("pref"), dict) else {}
    res, pairs = p.get("result") or {}, p.get("pairs") or {}
    out = {k: res[k] for k in PREF_KEYS if k in res}
    if pairs:
        out["pairs"] = {k: pairs[k] for k in PAIR_KEYS if k in pairs}
    return out


def write(run_dir: Path, dst_dir: Path | None = None) -> Path:
    ev = build(run_dir)
    text = json.dumps(ev, ensure_ascii=False, separators=(",", ":"))
    if len(text.encode()) > MAX_BYTES:  # 曲线再砍一半直到放得下
        for _ in range(4):
            ev["curves"] = {k: _downsample(v, max(len(v) // 2, 20)) for k, v in ev["curves"].items()}
            text = json.dumps(ev, ensure_ascii=False, separators=(",", ":"))
            if len(text.encode()) <= MAX_BYTES:
                break
    path = Path(dst_dir or run_dir) / "evidence.json"
    path.write_text(text)
    return path


def rebuild_all(root: Path) -> list[Path]:
    return [write(p.parent) for p in sorted(Path(root).glob("**/run.json"))]


def check_all(root: Path) -> list[str]:
    problems = []
    for p in sorted(Path(root).glob("**/evidence.json")):
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"{p}: {p.stat().st_size} 字节超过 {MAX_BYTES}")
        ev = _read_json(p)
        for k in ("version", "curves", "throughput", "memory", "params"):
            if k not in ev:
                problems.append(f"{p}: 缺字段 {k}")
    return problems
