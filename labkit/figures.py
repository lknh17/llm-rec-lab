"""课程 README 的静态证据图（纯 Python 写 SVG，不依赖 matplotlib；GitHub 页面直接显示）。

    courses/<课程目录>/figures/<档位>-<图名>.svg

每个档位有结果才画；输出是确定性的（不含时间戳、随机数），`labkit docs --check` 能判断是否最新。
网页上的交互图（悬停、切换种子）在私有仓库的 lab/web 里，读的是同一份 evidence.json。

图：
- metrics：每个场景的主指标（均值，误差线 = 种子间最小到最大，圆点 = 各种子）；
- loss：train_loss 曲线（多种子取均值，阴影 = 最小到最大），基线 / 整段监督 / 截断故障 / 学习率；
- memory：显存构成堆叠柱（只有 CUDA 档有实测峰值）；
- params：可训练参数量（rank、挂载层）。
"""

from __future__ import annotations

import json
import statistics
from html import escape
from pathlib import Path

W, PAD_L, PAD_R, PAD_T = 760, 190, 90, 46
FONT = "font-family=\"-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif\""
C = {"primary": "#3B5BDB", "light": "#EDF1FF", "grid": "#E3E6EC", "text": "#1D2330", "muted": "#6B7385",
     "green": "#2B8A3E", "orange": "#E67700", "red": "#C92A2A", "blue": "#1971C2"}
ROLE_COLOR = {"reference": C["muted"], "baseline": C["primary"], "variant": C["blue"], "fault": C["red"],
              "fix": C["green"], "final": C["text"]}
PALETTE = [C["primary"], C["orange"], C["red"], C["green"], C["blue"], "#7048E8", "#0C8599"]
METRIC = {"adextract": ("dev.exact_all_ok", "dev 全对率")}
LOSS_SCENES = {"adextract": ["baseline", "all_tokens", "fault_truncation", "lr_5e-5", "lr_1e-3"]}
MEMORY_SCENES = {"adextract": ["base_model", "lora_qv", "lora_attn", "baseline", "lora_r64"]}
PARAM_SCENES = {"adextract": ["lora_qv", "lora_attn", "lora_r4", "lora_r8", "baseline", "lora_r64"]}
MEM_PARTS = [("weights_gb", "冻结权重", C["muted"]), ("trainable_and_grads_gb", "LoRA 参数+梯度", C["primary"]),
             ("optimizer_gb", "AdamW 状态", C["blue"]), ("activations_other_gb", "激活与其他", C["orange"])]


def _get(d, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None


def _svg(h: int, body: list[str], title: str, subtitle: str = "") -> str:
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" {FONT}>',
            f'<rect width="{W}" height="{h}" rx="12" fill="#FFFFFF" stroke="{C["grid"]}"/>',
            f'<text x="20" y="28" font-size="15" font-weight="600" fill="{C["text"]}">{escape(title)}</text>']
    if subtitle:
        head.append(f'<text x="{W - 20}" y="28" font-size="11" fill="{C["muted"]}" text-anchor="end">{escape(subtitle)}</text>')
    return "\n".join(head + body + ["</svg>"]) + "\n"


def _nice(v: float) -> float:
    """向上取到 1 / 2 / 2.5 / 5 × 10^k，坐标刻度好读。"""
    import math

    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    return next(m * e for m in (1, 2, 2.5, 5, 10) if m * e >= v)


def _fmt(v: float, pct: bool) -> str:
    return f"{v * 100:.1f}%" if pct else (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}")


def bars(rows: list[dict], title: str, subtitle: str, pct: bool = False, unit: str = "") -> str:
    """横向柱状图。rows: {label, values: [每个种子], color}。"""
    rows = [r for r in rows if r["values"]]
    if not rows:
        return ""
    bh, gap = 22, 10
    h = PAD_T + len(rows) * (bh + gap) + 34
    vmax = max(max(r["values"]) for r in rows) or 1
    vmax = 1.0 if pct and vmax <= 1 else _nice(vmax * 1.05)
    plot_w = W - PAD_L - PAD_R
    x = lambda v: PAD_L + plot_w * max(v, 0) / vmax  # noqa: E731
    out = []
    for i in range(5):
        gx = PAD_L + plot_w * i / 4
        out.append(f'<line x1="{gx:.1f}" y1="{PAD_T - 6}" x2="{gx:.1f}" y2="{h - 28}" stroke="{C["grid"]}"/>')
        out.append(f'<text x="{gx:.1f}" y="{h - 12}" font-size="10" fill="{C["muted"]}" text-anchor="middle">'
                   f'{escape(_fmt(vmax * i / 4, pct))}</text>')
    for i, r in enumerate(rows):
        y = PAD_T + i * (bh + gap)
        vals = r["values"]
        m = statistics.fmean(vals)
        col = r.get("color", C["primary"])
        out.append(f'<text x="{PAD_L - 10}" y="{y + bh * 0.68:.1f}" font-size="12" fill="{C["text"]}" text-anchor="end">'
                   f'{escape(r["label"])}</text>')
        out.append(f'<rect x="{PAD_L}" y="{y}" width="{x(m) - PAD_L:.1f}" height="{bh}" rx="4" fill="{col}" fill-opacity="0.82">'
                   f'<title>{escape(r["label"])}：{escape(_fmt(m, pct))}{unit}（{len(vals)} 个种子）</title></rect>')
        if len(vals) > 1:
            lo, hi = min(vals), max(vals)
            cy = y + bh / 2
            out.append(f'<line x1="{x(lo):.1f}" y1="{cy}" x2="{x(hi):.1f}" y2="{cy}" stroke="{C["text"]}" stroke-width="1.4"/>')
            for v in vals:
                out.append(f'<circle cx="{x(v):.1f}" cy="{cy}" r="3" fill="#FFFFFF" stroke="{C["text"]}" stroke-width="1.2"/>')
        tail = _fmt(m, pct) + unit + (f" ±{statistics.stdev(vals) * (100 if pct else 1):.1f}" if len(vals) > 1 else "")
        out.append(f'<text x="{x(max(vals)) + 8:.1f}" y="{y + bh * 0.68:.1f}" font-size="11" fill="{C["muted"]}">'
                   f'{escape(tail)}</text>')
    return _svg(h, out, title, subtitle)


def stacked(rows: list[dict], parts: list[tuple[str, str, str]], title: str, subtitle: str, unit: str = " GB") -> str:
    """横向堆叠柱。rows: {label, parts: {key: value}}；parts: [(key, 名称, 颜色)]。"""
    rows = [r for r in rows if r["parts"]]
    if not rows:
        return ""
    bh, gap = 22, 10
    h = PAD_T + len(rows) * (bh + gap) + 56
    vmax = _nice(max(sum(r["parts"].values()) for r in rows) * 1.05)
    plot_w = W - PAD_L - PAD_R
    out = []
    for i in range(5):
        gx = PAD_L + plot_w * i / 4
        out.append(f'<line x1="{gx:.1f}" y1="{PAD_T - 6}" x2="{gx:.1f}" y2="{h - 50}" stroke="{C["grid"]}"/>')
        out.append(f'<text x="{gx:.1f}" y="{h - 34}" font-size="10" fill="{C["muted"]}" text-anchor="middle">'
                   f'{vmax * i / 4:.0f}{unit}</text>')
    for i, r in enumerate(rows):
        y = PAD_T + i * (bh + gap)
        out.append(f'<text x="{PAD_L - 10}" y="{y + bh * 0.68:.1f}" font-size="12" fill="{C["text"]}" text-anchor="end">'
                   f'{escape(r["label"])}</text>')
        x0 = PAD_L
        for key, name, col in parts:
            v = r["parts"].get(key) or 0
            w = plot_w * v / vmax
            if w > 0:
                out.append(f'<rect x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{bh}" fill="{col}">'
                           f'<title>{escape(r["label"])} · {escape(name)}：{v:.2f}{unit}</title></rect>')
            x0 += w
        out.append(f'<text x="{x0 + 8:.1f}" y="{y + bh * 0.68:.1f}" font-size="11" fill="{C["muted"]}">'
                   f'{sum(r["parts"].values()):.1f}{unit}</text>')
    lx = PAD_L
    for _key, name, col in parts:
        out.append(f'<rect x="{lx}" y="{h - 22}" width="10" height="10" rx="2" fill="{col}"/>')
        out.append(f'<text x="{lx + 14}" y="{h - 13}" font-size="11" fill="{C["text"]}">{escape(name)}</text>')
        lx += 28 + 12 * len(name)
    return _svg(h, out, title, subtitle)


def lines(series: list[dict], title: str, subtitle: str, ylabel: str = "") -> str:
    """折线 + 种子范围阴影。series: {label, color, runs: [[[x, y], ...] 每个种子]}。"""
    series = [s for s in series if s["runs"] and any(s["runs"])]
    if not series:
        return ""
    h, pb = 330, 64
    plot_w, plot_h = W - 70 - 30, h - PAD_T - pb
    allx = [p[0] for s in series for r in s["runs"] for p in r]
    ally = [p[1] for s in series for r in s["runs"] for p in r]
    x0, x1 = min(allx), max(allx) or 1
    y0, y1 = 0.0, _nice(max(ally))
    sx = lambda v: 70 + plot_w * (v - x0) / ((x1 - x0) or 1)  # noqa: E731
    sy = lambda v: PAD_T + plot_h * (1 - (v - y0) / ((y1 - y0) or 1))  # noqa: E731
    out = []
    for i in range(5):
        v = y0 + (y1 - y0) * i / 4
        out.append(f'<line x1="70" y1="{sy(v):.1f}" x2="{70 + plot_w}" y2="{sy(v):.1f}" stroke="{C["grid"]}"/>')
        out.append(f'<text x="62" y="{sy(v) + 3:.1f}" font-size="10" fill="{C["muted"]}" text-anchor="end">{v:g}</text>')
    out.append(f'<text x="{70 + plot_w / 2}" y="{h - pb + 30}" font-size="10" fill="{C["muted"]}" text-anchor="middle">'
               f'训练步（{x0:g} – {x1:g}）</text>')
    if ylabel:
        out.append(f'<text x="20" y="{PAD_T - 8}" font-size="10" fill="{C["muted"]}">{escape(ylabel)}</text>')
    lx = 70
    for s in series:
        runs = [r for r in s["runs"] if r]
        n = min(len(r) for r in runs)
        xs = [runs[0][i][0] for i in range(n)]
        mean = [statistics.fmean(r[i][1] for r in runs) for i in range(n)]
        if len(runs) > 1:
            lo = [min(r[i][1] for r in runs) for i in range(n)]
            hi = [max(r[i][1] for r in runs) for i in range(n)]
            poly = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, hi)) + " " + \
                " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(reversed(xs), reversed(lo)))
            out.append(f'<polygon points="{poly}" fill="{s["color"]}" fill-opacity="0.14"/>')
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, mean))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="1.8">'
                   f'<title>{escape(s["label"])}：最终 {mean[-1]:.4g}（{len(runs)} 个种子）</title></polyline>')
        out.append(f'<rect x="{lx}" y="{h - 24}" width="14" height="3" fill="{s["color"]}"/>')
        out.append(f'<text x="{lx + 18}" y="{h - 19}" font-size="11" fill="{C["text"]}">{escape(s["label"])}</text>')
        lx += 34 + 12 * len(s["label"])
    return _svg(h, out, title, subtitle)


# ------------------------------------------------------------------ 课程 → 图
def _evidence(run_dir: Path) -> dict:
    p = run_dir / "evidence.json"
    if p.exists():
        return json.loads(p.read_text())
    from . import evidence

    return evidence.build(run_dir)


def course_figures(course: dict, got: dict[str, list[dict]], profile: str) -> dict[str, str]:
    """图名 → SVG 文本。got 来自 docs.collect（场景 → 各种子 {seed, summary, meta, dir}）。"""
    project = course["project"]
    by_id = {s["id"]: s for s in course["scenarios"]}
    n_seeds = max((len(v) for v in got.values()), default=0)
    sub = f"{profile} · 每个场景最多 {n_seeds} 个种子"
    figs: dict[str, str] = {}
    key, name = METRIC[project]
    rows = []
    for s in course["scenarios"]:
        runs = got.get(s["id"]) or []
        vals = [v for r in runs if (v := _get(r["summary"], key)) is not None]
        rows.append({"label": s["id"], "values": vals, "color": ROLE_COLOR.get(s["role"], C["primary"])})
    figs["metrics"] = bars(rows, f"{name}（各场景）", sub, pct=project == "adextract")
    loss = []
    for i, sid in enumerate(LOSS_SCENES.get(project, [])):
        runs = [_evidence(r["dir"]).get("curves", {}).get("train_loss") or [] for r in got.get(sid) or []]
        if runs:
            loss.append({"label": sid, "color": PALETTE[i % len(PALETTE)], "runs": runs})
    figs["loss"] = lines(loss, "train loss（均值，阴影为种子间范围）", sub)
    mem = []
    for sid in MEMORY_SCENES.get(project, []):
        evs = [_evidence(r["dir"]).get("memory") or {} for r in got.get(sid) or []]
        evs = [e for e in evs if e.get("breakdown") and e.get("peak_allocated_gb") is not None]
        if evs:
            keys = evs[0]["breakdown"].keys()
            mem.append({"label": sid, "parts": {k: statistics.fmean(e["breakdown"].get(k) or 0 for e in evs) for k in keys}})
    figs["memory"] = stacked(mem, MEM_PARTS, "峰值显存构成（实测峰值，分项为估算）", sub)
    prm = []
    for sid in PARAM_SCENES.get(project, []):
        vals = [v for r in got.get(sid) or [] if (v := _get(r["meta"], "model.params.trainable")) is not None]
        if vals:
            prm.append({"label": f"{sid}", "values": vals[:1], "color": ROLE_COLOR.get(by_id.get(sid, {}).get("role"), C["blue"])})
    figs["params"] = bars(prm, "可训练参数量（rank × 挂载层）", sub) if len(prm) > 1 else ""
    return {k: v for k, v in figs.items() if v}
