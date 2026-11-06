"""GitHub 上一步一页的课程讲义，以及作业看板。只看 GitHub 也能按步骤学完一门课、看清作业进度。

    courses/<课程目录>/lesson.yaml          公开讲义源（问题、选项、要看的证据；不含答案和解析）
      → courses/<课程目录>/steps/NN-*.md     每步一页：问题 → 先预测 → 折叠的实测（表、图、运行目录链接）
      → courses/<课程目录>/steps/figures/*.svg
      → README.md 的 <!-- labkit:path --> 区：步骤目录和每步的 H20 结果数
    jobs/README.md                          作业看板：运行中 / 排队 / 已完成 / 已作废、H20 心跳、场景 × 种子矩阵

课程页只依赖 lesson.yaml、场景清单、results/ 和项目代码，输出是确定性的（`labkit docs --check` 能判断是否最新）。
作业看板含心跳时间，只由 GitHub Actions（.github/workflows/pages.yml）生成并提交：Mac 和 H20 都不写它，避免推送冲突。
"""

from __future__ import annotations

import datetime as dt
import json
import re
import statistics
from pathlib import Path

import yaml

from . import docs, figures, paths, scenarios

LESSON_FILE = "lesson.yaml"
PENDING = "〔待 H20 生成〕"
PLACEHOLDER = re.compile(r"\{\{v:([\w.-]+):([^:}]+)(?::([^}]+))?\}\}")
OPTION_KEYS = "ABCDEFGH"
STATE_NAMES = {"pending": "排队中", "running": "运行中", "done": "已完成", "cancelled": "已作废"}
JOB_STATUS = {"done": "全部成功", "partial": "部分成功", "failed": "失败", "rejected": "被拒绝（校验不通过）"}


# ------------------------------------------------------------------ 取数
class Book:
    """一门课的讲义 + 各档位已发布的结果。"""

    def __init__(self, course: dict, lesson: dict, course_dir: Path):
        self.course, self.lesson, self.dir = course, lesson, course_dir
        self.cid = course["course"]
        self.by_id = {s["id"]: s for s in course["scenarios"]}
        self.labels: dict[str, str] = lesson.get("profiles") or {}
        self.profiles = [p for p in self.labels
                         if p in course["profiles"] and not course["profiles"][p].get("skip_expect")]
        self.primary = self.profiles[0]
        self.got = {p: docs.collect(self.cid, p) for p in self.profiles}
        self._ev: dict[Path, dict] = {}
        self.steps = lesson["steps"]
        self.stage_title = {s["id"]: s["title"] for s in lesson.get("stages", [])}
        self.anchors = _anchors(course["project"])

    # 单个运行
    def evidence(self, run: dict) -> dict:
        if run["dir"] not in self._ev:
            self._ev[run["dir"]] = figures._evidence(run["dir"])
        return self._ev[run["dir"]]

    def raw(self, run: dict, path: str) -> float | None:
        if "/" in path:
            a, b = (self.raw(run, p.strip()) for p in path.split("/", 1))
            return a / b if a is not None and b else None
        src, _, rest = path.partition(".")
        d = {"summary": lambda: run["summary"], "meta": lambda: run["meta"],
             "evidence": lambda: self.evidence(run)}.get(src, lambda: None)()
        for k in rest.split("."):
            if k == "last" and isinstance(d, list):
                d = d[-1] if d else None
                d = d[-1] if isinstance(d, list) and d else d
                continue
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None

    # 一个场景
    def runs(self, profile: str, sid: str) -> list[dict]:
        return sorted(self.got[profile].get(sid) or [], key=lambda r: (r["seed"] is None, r["seed"] or 0, str(r["dir"])))

    def values(self, profile: str, sid: str, path: str) -> list[float]:
        return [v for r in self.runs(profile, sid) if (v := self.raw(r, path)) is not None]

    def pick(self, sid: str, path: str) -> tuple[str | None, list[float]]:
        for p in self.profiles:
            vals = self.values(p, sid, path)
            if vals:
                return p, vals
        return None, []

    def expected(self, profile: str, sids: list[str]) -> int:
        try:
            return sum(1 for s in scenarios.expand(self.cid, profile) if s.scenario in sids)
        except (KeyError, ValueError):
            return 0

    def have(self, profile: str, sids: list[str]) -> int:
        return sum(len(self.got[profile].get(s) or []) for s in sids)

    def short(self, profile: str) -> str:
        return self.labels.get(profile, profile).split(" · ")[0]

    def render(self, text: str) -> str:
        """替换 {{v:..}}。用了非主档位的数字时，在句末（多段文字则另起一段）统一注明来源档位。"""
        fallback: set[str] = set()

        def sub(m: re.Match) -> str:
            sid, path, spec = m.group(1), m.group(2), m.group(3)
            prof, vals = self.pick(sid, path)
            if not vals:
                return PENDING
            if prof != self.primary:
                fallback.add(prof)
            return fmt_agg(vals, spec)

        out = PLACEHOLDER.sub(sub, text or "")
        if not fallback:
            return out
        src = "、".join(self.short(p) for p in self.profiles if p in fallback)
        if "\n" in out.strip():
            return out.rstrip() + f"\n\n_（文中数字来自 {src}，{self.short(self.primary)} 结果推回后自动替换）_\n"
        return out + f"（数字来自 {src}）"


# ------------------------------------------------------------------ 格式化
def fmt(v: float, spec: str | None = None, pct: bool = False, signed: bool = False) -> str:
    sign = "+" if signed else ""
    if pct or spec == "pct":
        return f"{v * 100:{sign}.1f}%"
    if spec == "params":
        a = abs(v)
        s = f"{v / 1e8:{sign}.2f} 亿" if a >= 1e8 else f"{v / 1e4:{sign},.0f} 万" if a >= 1e4 else f"{v:{sign},.0f}"
        return s
    if spec:
        if spec.endswith("d"):
            return format(int(round(v)), sign + spec)
        return format(v, sign + spec)
    return f"{v:{sign}.4g}"


def fmt_agg(vals: list[float], spec: str | None = None, pct: bool = False) -> str:
    if not vals:
        return "—"
    m = statistics.fmean(vals)
    if len(vals) == 1 or (spec == "params" and len(set(vals)) == 1):  # 参数量由配置决定，种子间不变，不写 ± 0
        return fmt(m, spec, pct)
    sd = statistics.stdev(vals)
    sd_s = f"{sd * 100:.1f}" if (pct or spec == "pct") else fmt(sd, spec)
    return f"{fmt(m, spec, pct)} ± {sd_s}"


def fmt_delta(d: float, col: dict) -> str:
    if col.get("pct"):
        return f"{d * 100:+.1f} 个点"
    return fmt(d, col.get("fmt"), signed=True)


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


# ------------------------------------------------------------------ 代码锚点
ANCHOR_START = re.compile(r"#\s*--8<--\s*\[start:([\w-]+)\]")


def _anchors(project: str) -> dict[tuple[str, str], tuple[Path, int, int]]:
    """(相对 projects/ 的文件, 锚点名) → (文件, 起始行, 结束行)，行号 1 起，不含标记行。"""
    out = {}
    root = paths.PROJECTS_DIR / project
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            m = ANCHOR_START.search(line)
            if not m:
                continue
            name = m.group(1)
            end = next((j for j in range(i + 1, len(lines) + 1)
                        if re.search(rf"#\s*--8<--\s*\[end:{re.escape(name)}\]", lines[j - 1])), len(lines) + 1)
            out[(f.relative_to(paths.PROJECTS_DIR).as_posix(), name)] = (f, i + 1, end - 1)
    return out


def _code_link(book: Book, file: str, region: str) -> str:
    hit = book.anchors.get((file, region))
    if not hit:
        return f"`{file}`"
    f, a, b = hit
    rel = Path("../../..") / f.relative_to(paths.ROOT)
    return f"[`{f.relative_to(paths.PROJECTS_DIR).as_posix()}#L{a}-L{b}`]({rel.as_posix()}#L{a}-L{b})"


def _code_snippet(book: Book, file: str, region: str) -> str:
    hit = book.anchors.get((file, region))
    if not hit:
        return ""
    f, a, b = hit
    lines = f.read_text().splitlines()[a - 1:b]
    return "```python\n" + "\n".join(lines) + "\n```"


# ------------------------------------------------------------------ 一步一页
def _file(step: dict) -> str:
    return f"{step['file']}.md"


def _question_text(step: dict) -> str:
    if step.get("question"):
        return step["question"]["text"]
    return step.get("question_short") or (step.get("questions") or [{}])[0].get("text", "")


def _options(q: dict) -> list[str]:
    return [f"- **{OPTION_KEYS[i]}.** {o}" for i, o in enumerate(q.get("options") or [])]


def _run_link(run: dict, from_dir: Path) -> str:
    rel = Path("../../..") / run["dir"].relative_to(paths.ROOT)
    return f"[s{run['seed']}]({rel.as_posix()}/)"


def _scenario_table(book: Book, step: dict) -> list[str]:
    rows = ["| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（" + book.short(book.primary) + "） |", "| - | - | - | - | - |"]
    for sid, role in _rows(step):
        s = book.by_id.get(sid)
        if not s:
            continue
        diff = ", ".join(f"`{k}={v}`" for k, v in (s.get("params") or {}).items()) or "—"
        try:
            n = sum(1 for x in scenarios.expand(book.cid, book.primary) if x.scenario == sid)
        except (KeyError, ValueError):
            n = 0
        rows.append(f"| `{sid}` | {role} | {_cell(s.get('title', ''))} | {diff} | {n or '本档不跑'} |")
    return rows


def _rows(step: dict) -> list[tuple[str, str]]:
    out = []
    if step.get("compare_to"):
        out.append((step["compare_to"], "对照"))
    for sid in step.get("also") or []:
        out.append((sid, "参照"))
    for sid in step.get("scenarios") or []:
        out.append((sid, "本步"))
    return out


def _results_table(book: Book, step: dict, profile: str) -> list[str]:
    cols = step.get("columns") or [book.lesson["metric"]]
    dcol = next((c for c in cols if c.get("delta")), cols[0])
    ref = step.get("compare_to")
    ref_vals = book.values(profile, ref, dcol["path"]) if ref else []
    head = "| 场景 | 类型 | " + " | ".join(c["label"] for c in cols)
    if ref:
        head += f" | {dcol['label']}：与 `{ref}` 差"
    head += f" | {dcol['label']}：各种子 | 断言 |"
    out = [head, "|" + " - |" * (head.count("|") - 1)]
    for sid, _role in _rows(step):
        s = book.by_id.get(sid) or {}
        runs = book.runs(profile, sid)
        if not runs:
            if not s.get("profiles") or profile in s["profiles"]:
                out.append(f"| `{sid}` | {docs.ROLE_NAMES.get(s.get('role'), '')} | "
                           + " | ".join(PENDING for _ in cols) + (" | " if ref else "") + " | | |")
            continue
        cells = [fmt_agg(book.values(profile, sid, c["path"]), c.get("fmt"), c.get("pct")) for c in cols]
        row = f"| `{sid}` | {docs.ROLE_NAMES.get(s.get('role'), '')} | " + " | ".join(cells)
        dvals = book.values(profile, sid, dcol["path"])
        if ref:
            row += " | " + ("—" if sid == ref or not dvals or not ref_vals
                            else fmt_delta(statistics.fmean(dvals) - statistics.fmean(ref_vals), dcol))
        seeds = " / ".join(f"s{r['seed']} {fmt(v, dcol.get('fmt'), dcol.get('pct'))}"
                           for r in runs if (v := book.raw(r, dcol["path"])) is not None)
        oks = [r["meta"].get("expect_ok") for r in runs if s.get("expect")]
        mark = "" if not oks else ("通过" if all(o is True for o in oks) else "未通过" if any(o is False for o in oks) else "待定")
        out.append(row + f" | {seeds or '—'} | {mark} |")
    return out


def _step_figures(book: Book, step: dict, profile: str, want: dict[Path, str]) -> list[str]:
    sids = [sid for sid, _ in _rows(step)]
    got = book.got[profile]
    sub = f"{profile} · {step['title']}"
    cols = step.get("columns") or [book.lesson["metric"]]
    out = []
    for chart in step.get("charts") or []:
        svg, title = "", ""
        if chart == "metrics":
            c = cols[0]
            rows = [{"label": sid, "values": book.values(profile, sid, c["path"]),
                     "color": figures.ROLE_COLOR.get(book.by_id.get(sid, {}).get("role"), figures.C["primary"])} for sid in sids]
            title = f"{c['label']}（均值，误差线为种子间最小到最大）"
            svg = figures.bars(rows, title, sub, pct=bool(c.get("pct")))
        elif chart == "loss":
            series = []
            for i, sid in enumerate(sids):
                runs = [book.evidence(r).get("curves", {}).get("train_loss") or [] for r in got.get(sid) or []]
                if any(runs):
                    series.append({"label": sid, "color": figures.PALETTE[i % len(figures.PALETTE)], "runs": runs})
            title = "train loss（均值，阴影为种子间范围）"
            svg = figures.lines(series, title, sub)
        elif chart == "memory":
            mem = []
            for sid in sids:
                evs = [book.evidence(r).get("memory") or {} for r in got.get(sid) or []]
                evs = [e for e in evs if e.get("breakdown") and e.get("peak_allocated_gb") is not None]
                if evs:
                    keys = evs[0]["breakdown"].keys()
                    mem.append({"label": sid, "parts": {k: statistics.fmean(e["breakdown"].get(k) or 0 for e in evs) for k in keys}})
            title = "峰值显存构成（实测峰值，分项为估算）"
            svg = figures.stacked(mem, figures.MEM_PARTS, title, sub)
        elif chart in ("params", "throughput"):
            path = "meta.model.params.trainable" if chart == "params" else "evidence.throughput.tokens_per_s_median"
            rows = []
            for sid in sids:
                vals = book.values(profile, sid, path)
                rows.append({"label": sid, "values": vals[:1] if chart == "params" else vals,
                             "color": figures.ROLE_COLOR.get(book.by_id.get(sid, {}).get("role"), figures.C["blue"])})
            title = "可训练参数量" if chart == "params" else "训练吞吐（tokens/s，中位数）"
            svg = figures.bars(rows, title, sub, unit="" if chart == "params" else " tok/s")
        if svg:
            p = book.dir / "steps" / "figures" / f"{profile}-{step['id']}-{chart}.svg"
            want[p] = svg
            out.append(f"![{title}](figures/{p.name})")
    return out


def _errors_block(book: Book, step: dict, profile: str) -> list[str]:
    sids = [sid for sid, _ in _rows(step) if book.runs(profile, sid)]
    counts: dict[str, dict[str, float]] = {}
    for sid in sids:
        runs = book.runs(profile, sid)
        keys = {k for r in runs for k in ((r["summary"].get("dev") or {}).get("error_counts") or {})}
        counts[sid] = {k: statistics.fmean(((r["summary"].get("dev") or {}).get("error_counts") or {}).get(k, 0) for r in runs)
                       for k in keys}
    keys = sorted({k for c in counts.values() for k in c}, key=lambda k: (-max(c.get(k, 0) for c in counts.values()), k))
    if not keys:
        return []
    out = ["**dev 错误类型分布**（每个运行的错误条数，多种子取均值；dev 共 30 条，一条样本可能有多个错误）", "",
           "| 错误类型 | " + " | ".join(f"`{s}`" for s in sids) + " |", "|" + " - |" * (len(sids) + 1)]
    for k in keys:
        out.append(f"| {_cell(k)} | " + " | ".join(f"{counts[s].get(k, 0):.1f}" for s in sids) + " |")
    return out


def _predictions(run: dict) -> dict[str, dict]:
    p = run["dir"] / "predictions.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("set") == "dev":
                out[d["id"]] = d
    return out


def _ok(pred: dict) -> bool:
    return not pred.get("errors") and not pred.get("format_error")


def _flips_block(book: Book, step: dict, profile: str) -> list[str]:
    ref, sids = step.get("compare_to"), step.get("scenarios") or []
    if not ref or not sids:
        return []
    ref_runs = {r["seed"]: r for r in book.runs(profile, ref)}
    runs = book.runs(profile, sids[0])
    if not ref_runs or not runs:
        return []
    out = [f"**逐样本：`{ref}` → `{sids[0]}`**（dev 30 条；修好 = 对照错、本步对；弄坏 = 对照对、本步错）", "",
           "| 种子 | 修好 | 弄坏 | 两边都对 | 两边都错 |", "| - | - | - | - | - |"]
    detail: list[str] = []
    for r in runs:
        base = ref_runs.get(r["seed"]) or ref_runs[min(ref_runs)]
        a, b = _predictions(base), _predictions(r)
        ids = sorted(set(a) & set(b))
        fixed = [i for i in ids if not _ok(a[i]) and _ok(b[i])]
        broken = [i for i in ids if _ok(a[i]) and not _ok(b[i])]
        both = sum(1 for i in ids if _ok(a[i]) and _ok(b[i]))
        neither = len(ids) - len(fixed) - len(broken) - both
        out.append(f"| s{r['seed']}（对照 s{base['seed']}） | {len(fixed)} | {len(broken)} | {both} | {neither} |")
        if not detail:
            for i in fixed[:8]:
                detail.append(f"| 修好 | `{i}` | {_cell(a[i].get('text', ''))} | {_cell('、'.join(a[i].get('errors') or []) or a[i].get('format_error') or '')} |")
            for i in broken[:8]:
                detail.append(f"| 弄坏 | `{i}` | {_cell(b[i].get('text', ''))} | {_cell('、'.join(b[i].get('errors') or []) or b[i].get('format_error') or '')} |")
    if detail:
        out += ["", "第一个种子里的例子（最多各 8 条；最后一列是出错一方的错误类型）：", "",
                "| | 样本 | 文案 | 错误 |", "| - | - | - | - |", *detail]
    return out


def _results_section(book: Book, step: dict, want: dict[Path, str]) -> list[str]:
    sids = [sid for sid, _ in _rows(step)]
    out = ["## 实测", ""]
    shown = False
    for profile in book.profiles:
        have, exp = book.have(profile, sids), book.expected(profile, sids)
        if profile != book.primary and not have:
            continue
        label = book.labels.get(profile, profile)
        note = "" if profile == book.primary else "，只作对照，不能推断 7B 上的结论"
        out.append(f"<details><summary>展开实测：{label}（已有 {have} / {exp} 个运行{note}）</summary>")
        out.append("")
        if not have:
            out += ["_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_", ""]
            out += ["</details>", ""]
            continue
        shown = True
        out += _results_table(book, step, profile) + [""]
        for block in step.get("blocks") or []:
            lines = {"errors": _errors_block, "flips": _flips_block}.get(block, lambda *_: [])(book, step, profile)
            if lines:
                out += lines + [""]
        figs = _step_figures(book, step, profile, want)
        out += [f + "\n" for f in figs]
        links = []
        for sid in sids:
            rs = book.runs(profile, sid)
            if rs:
                links.append(f"`{sid}` " + " · ".join(_run_link(r, book.dir) for r in rs))
        if links:
            out += ["运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：", ""]
            out += [f"- {x}" for x in links] + [""]
        out += ["</details>", ""]
    if not shown and len(out) == 2:
        out.append(PENDING)
    return out


def _trace_block(book: Book) -> list[str]:
    for profile in book.profiles:
        runs = book.runs(profile, "baseline")
        if runs and (runs[0]["dir"] / "trace.json").exists():
            run = runs[0]
            break
    else:
        return [PENDING]
    t = json.loads((run["dir"] / "trace.json").read_text())
    rel = (Path("../../..") / run["dir"].relative_to(paths.ROOT) / "trace.json").as_posix()
    s, b, tt = t.get("sample") or {}, t.get("batch") or {}, t.get("train_tokens") or {}
    out = [f"来源：{book.labels.get(profile, profile)}，`baseline` s{run['seed']} 的 [`trace.json`]({rel})。", ""]
    raw = s.get("raw") or {}
    out += ["### 样本", "", f"- 文案：{raw.get('text', '')}",
            f"- 标注：`{json.dumps(raw.get('gold'), ensure_ascii=False)}`", ""]
    msgs = s.get("messages") or []
    if msgs:
        out += ["| 角色 | 字符数 | 内容（开头） |", "| - | - | - |"]
        for m in msgs:
            c = m.get("content") or ""
            out.append(f"| {m.get('role')} | {len(c)} | {_cell(c[:60])}{'…' if len(c) > 60 else ''} |")
        out.append("")
    n_p, n_t, n_s = s.get("n_prompt"), s.get("n_total"), s.get("n_supervised")
    if n_t:
        out += ["### token 与监督 mask", "",
                f"prompt {n_p} 个 token，全长 {n_t}，**参与 loss 的 {n_s} 个**（{(n_s or 0) / n_t:.1%}），截断：{s.get('truncated')}。", ""]
    toks = tt.get("tokens") or []
    if toks:
        first_sup = next((i for i, x in enumerate(toks) if x.get("supervised")), len(toks))
        pick = list(range(max(0, first_sup - 6), min(len(toks), first_sup + 6))) + \
            [i for i in range(max(first_sup + 6, len(toks) - 4), len(toks))]
        out += ["监督边界附近的 token（✅ = 参与 loss；中间省略的都是 ✅）：", "",
                "| 位置 | token id | token | 区域 | 监督 |", "| - | - | - | - | - |"]
        last = None
        for i in pick:
            if last is not None and i != last + 1:
                out.append("| … | | | | |")
            x = toks[i]
            out.append(f"| {x.get('i')} | {x.get('id')} | `{_cell(json.dumps(x.get('text'), ensure_ascii=False)[1:-1])}` | "
                       f"{x.get('region', '')} | {'✅' if x.get('supervised') else ''} |")
            last = i
        out.append("")
    if tt.get("supervised_text"):
        out += ["被监督的文本（loss 只算这一段）：", "", "```text", tt["supervised_text"].rstrip("\n") + "⏎", "```", ""]
    tensors = [(k, v) for k, v in b.items() if isinstance(v, dict) and v.get("shape")]
    if tensors:
        out += ["### 第 1 个 batch 的张量", "", "| 张量 | 形状 | dtype | 备注 |", "| - | - | - | - |"]
        for k, v in tensors:
            note = ""
            if v.get("start_position") is not None:
                note = f"从第 {v['start_position']} 个位置开始算；不切片时是 `{v.get('full_shape_if_not_sliced')}`"
            out.append(f"| `{k}` | `{v['shape']}` | {v.get('dtype', '')} | {note} |")
        out.append("")
        extra = [(k, b[k]) for k in ("per_sample_len", "per_sample_supervised", "pad_tokens", "batch_loss") if k in b]
        out += [f"- `{k}`：{v}" for k, v in extra] + [""]
    uc = t.get("update_check") or {}
    if uc:
        out += ["### 参数有没有真的更新", "", "| 检查 | 训练前 | 训练后 |", "| - | - | - |",
                f"| LoRA B 的范数（初始化为 0） | {uc.get('lora_B_norm_before')} | {fmt(uc['lora_B_norm_after'], '.4g') if isinstance(uc.get('lora_B_norm_after'), (int, float)) else uc.get('lora_B_norm_after')} |",
                f"| 基础权重指纹（必须不变） | {uc.get('base_weight_probe_before')} | {uc.get('base_weight_probe_after')} |", ""]
    rl = t.get("reload") or {}
    if rl:
        out += ["### 保存 → 重新加载核对", "", f"重新加载基础模型 + 适配器后，抽查 {rl.get('checked')} 条，输出完全一致 {rl.get('identical')} 条。", ""]
    return out


def _code_block(book: Book, step: dict) -> list[str]:
    out = []
    for c in step.get("code") or []:
        out += [f"### {c['title']}", "", c.get("note", ""), "", _code_link(book, c["file"], c["region"]), "",
                _code_snippet(book, c["file"], c["region"]), ""]
    return out


def _overview_block(book: Book, step_of: dict[str, dict]) -> list[str]:
    m = book.lesson["metric"]
    out = []
    for profile in book.profiles:
        got = book.got[profile]
        if not got:
            if profile == book.primary:
                out += [f"_{book.labels.get(profile, profile)}：还没有结果，进度见 [作业看板](../../../jobs/README.md)。_", ""]
            continue
        base = book.values(profile, "baseline", m["path"])
        out += [f"<details><summary>{book.labels.get(profile, profile)}</summary>", "",
                f"| 场景 | 类型 | 在哪一步 | {m['label']} | 与 `baseline` 差 | 种子数 |", "| - | - | - | - | - | - |"]
        for s in book.course["scenarios"]:
            vals = book.values(profile, s["id"], m["path"])
            if not vals:
                continue
            st = step_of.get(s["id"])
            where = f"[{st['title']}]({_file(st)})" if st else ""
            d = fmt_delta(statistics.fmean(vals) - statistics.fmean(base), m) if base and s["id"] != "baseline" else "—"
            out.append(f"| `{s['id']}` | {docs.ROLE_NAMES.get(s['role'], '')} | {where} | {fmt_agg(vals, None, m.get('pct'))} | {d} | {len(vals)} |")
        out += ["", "</details>", ""]
    out += ["全部图和结果表也汇总在 [课程首页](../README.md) 的“场景与结果”“证据图”两节。", ""]
    return out


def step_page(book: Book, i: int, want: dict[Path, str]) -> str:
    steps = book.steps
    step = steps[i]
    n = len(steps)
    stage_ids = [s["id"] for s in book.lesson.get("stages", [])]
    stage = step.get("stage")
    nav = ["[课程首页](../README.md)"]
    if i > 0:
        nav.insert(0, f"[← {steps[i - 1]['title']}]({_file(steps[i - 1])})")
    if i + 1 < n:
        nav.append(f"[{steps[i + 1]['title']} →]({_file(steps[i + 1])})")
    head = ["<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->", "",
            " · ".join(nav), "",
            f"# 第 {i + 1} 步 / 共 {n} 步：{step['title']}", ""]
    if stage:
        head += [f"> 阶段 {stage_ids.index(stage) + 1} / {len(stage_ids)}：**{book.stage_title.get(stage, stage)}**"
                 + (" · 场景：" + " ".join(f"`{s}`" for s in step["scenarios"]) if step.get("scenarios") else "")
                 + (f"（对照 `{step['compare_to']}`）" if step.get("compare_to") else ""), ""]
    out = head + [book.render(step.get("body", "")).rstrip(), ""]

    if step.get("nodes"):
        out += ["| 环节 | 输入输出 / 形状 | 说明 | 代码 |", "| - | - | - | - |"]
        for k, nd in enumerate(step["nodes"], 1):
            code = _code_link(book, nd["code"]["file"], nd["code"]["region"]) if nd.get("code") else ""
            out.append(f"| {k}. {_cell(nd['title'])} | {_cell(nd.get('shape', ''))} | {_cell(nd.get('detail', ''))} | {code} |")
        out.append("")

    if step.get("questions"):
        out += ["## 先估算", ""]
        for k, q in enumerate(step["questions"], 1):
            out += [f"**问题 {k}**：{q['text']}", ""] + _options(q) + [""]
        out += ["> 每题先选定一个，写下你的估算过程，再展开下面的核对。", ""]
    if step.get("check"):
        out += ["## 核对", "", "<details><summary>展开核对用的实测数</summary>", "",
                "| 项目 | 实测 |", "| - | - |"]
        out += [f"| {_cell(c['label'])} | {_cell(book.render(c['value']))} |" for c in step["check"]]
        out += ["", "</details>", ""]

    if "trace" in (step.get("blocks") or []):
        out += _trace_block(book)
    if step.get("code"):
        out += _code_block(book, step)

    if step.get("scenarios"):
        out += ["## 这一步跑了什么", ""] + _scenario_table(book, step) + [""]
    if step.get("clues"):
        out += ["## 线索", "", "逐条展开，看到能下判断为止。", ""]
        for k, c in enumerate(step["clues"], 1):
            out += [f"<details><summary>线索 {k}</summary>", "", book.render(c), "", "</details>", ""]
    if step.get("question"):
        out += ["## 先预测", "", f"**{step['question']['text']}**", ""] + _options(step["question"]) + \
            ["", "> 先选定一个（记下来），再展开下面的实测。差异小于 3.3 个点（dev 一条样本）或小于种子间波动，都按“差不多”算。", ""]
    if step.get("scenarios"):
        out += _results_section(book, step, want)
    if step.get("look"):
        out += ["## 看完实测再想一步", "", book.render(step["look"]), ""]

    if "overview" in (step.get("blocks") or []):
        step_of = {}
        for st in steps:
            for sid in st.get("scenarios") or []:
                step_of.setdefault(sid, st)
        out += _overview_block(book, step_of)
    if step.get("selfcheck"):
        out += ["## 自查", "", "能不能不看结果，用自己的话讲清下面每一条的“为什么”？讲不清的回到对应步骤重看。", ""]
        out += [f"{k}. {q}" for k, q in enumerate(step["selfcheck"], 1)] + [""]

    out += ["---", "", " · ".join(nav), ""]
    return "\n".join(out)


def path_table(book: Book) -> str:
    """README 的步骤目录：阶段、步骤链接、问题、场景、主档位已有结果数。"""
    rows = [f"| # | 阶段 | 步骤 | 这一步回答的问题 | 场景 | {book.short(book.primary)} 结果 |", "| - | - | - | - | - | - |"]
    for i, st in enumerate(book.steps, 1):
        sids = st.get("scenarios") or []
        res = f"{book.have(book.primary, sids)} / {book.expected(book.primary, sids)}" if sids else "—"
        rows.append(f"| {i} | {book.stage_title.get(st.get('stage'), '')} | [{st['title']}](steps/{_file(st)}) | "
                    f"{_cell(_question_text(st))} | {' '.join(f'`{s}`' for s in sids) or '—'} | {res} |")
    return "\n".join(rows)


def build_course(course: dict, course_dir: Path, check: bool, changed: list[str]) -> str | None:
    """生成（或检查）一门课的 steps/。返回 README path 区的内容；没有 lesson.yaml 时返回 None。"""
    lf = course_dir / LESSON_FILE
    if not lf.exists():
        return None
    lesson = yaml.safe_load(lf.read_text())
    book = Book(course, lesson, course_dir)
    steps_dir = course_dir / "steps"
    want: dict[Path, str] = {}
    for i, st in enumerate(book.steps):
        want[steps_dir / _file(st)] = step_page(book, i, want)
    existing = [p for p in steps_dir.rglob("*") if p.is_file() and p.suffix in (".md", ".svg")] if steps_dir.exists() else []
    for p, text in want.items():
        if not p.exists() or p.read_text() != text:
            changed.append(str(p.relative_to(paths.ROOT)))
            if not check:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
    for p in existing:
        if p not in want:
            changed.append(str(p.relative_to(paths.ROOT)))
            if not check:
                p.unlink()
    return path_table(book)


# ------------------------------------------------------------------ 作业看板（只由 GitHub Actions 生成）
def _load_jobs() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {k: [] for k in STATE_NAMES}
    for state in STATE_NAMES:
        d = paths.JOBS_DIR / state
        for f in sorted(d.glob("*.yaml"), reverse=True) if d.exists() else []:
            try:
                job = yaml.safe_load(f.read_text()) or {}
            except yaml.YAMLError:
                job = {}
            job["_file"] = f
            job.setdefault("job_id", f.stem)
            out[state].append(job)
    return out


def _n_runs(job: dict) -> int:
    n = 0
    for c in job.get("courses") or []:
        try:
            n += len(scenarios.expand(c, job["profile"], job.get("only") or None, job.get("seeds") or None))
        except (KeyError, ValueError):
            pass
    return n


def _utc(ts) -> str:
    if ts in (None, ""):
        return "—"
    if isinstance(ts, (int, float)):
        t = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    else:
        try:
            t = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            return str(ts)
    bj = t.astimezone(dt.timezone(dt.timedelta(hours=8)))
    return f"{t:%m-%d %H:%M} UTC（北京 {bj:%H:%M}）"


def _job_link(job: dict) -> str:
    f: Path = job["_file"]
    return f"[`{job['job_id']}`]({f.relative_to(paths.JOBS_DIR).as_posix()})"


def _note(job: dict, n: int = 80) -> str:
    s = str(job.get("notes") or job.get("note") or "").strip().splitlines()
    s = s[0] if s else ""
    return _cell(s[:n] + ("…" if len(s) > n else ""))


def jobs_board(now: float | None = None) -> str:
    jobs_by = _load_jobs()
    out = ["<!-- 本页由 GitHub Actions（.github/workflows/pages.yml）运行 python -m labkit docs --board 生成，不要手改 -->", "",
           "# 作业看板", "",
           "作业单是 Mac 端和 H20 之间唯一的通道：Mac 写进 [`pending/`](pending/) → H20 agent 领取移到 [`running/`](running/) → "
           "跑完移到 [`done/`](done/)，并写回每个运行的状态和耗时；作废的放在 [`cancelled/`](cancelled/)。"
           "结果在 [`results/`](../results/)，按步骤看结果请从课程首页进入（例如 [L05 SFT 与 LoRA](../courses/L1_lora_extraction/README.md)）。", ""]
    status_files = sorted((paths.ROOT / "status").glob("*.json")) if (paths.ROOT / "status").exists() else []
    if status_files:
        out += ["## H20 实时状态", "", "| 档位 | 状态 | 当前作业 | 进度 | GPU | 最后心跳 |", "| - | - | - | - | - | - |"]
        for f in status_files:
            try:
                st = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            state = {"idle": "空闲", "running": "运行中", "publishing": "推送结果中", "error": "出错"}.get(st.get("state"), st.get("state"))
            if st.get("last_error"):
                state += f"：{_cell(str(st['last_error'])[:60])}"
            gpu = f"{st.get('gpu_count') or '?'} × {st.get('gpu') or '?'}"
            out.append(f"| {st.get('profile') or f.stem} | {state} | {st.get('job_id') or '—'} | {st.get('progress') or '—'} | "
                       f"{gpu} | {_utc(st.get('updated_at'))} |")
        out += ["", "> agent 状态变化时立即推送心跳；状态不变时，运行中每 15 分钟、空闲每 3 小时推送一次。"
                "最后心跳超过两个间隔（运行中 30 分钟、空闲 6 小时）没更新，说明 agent 可能停了。"
                "进度是已结束的运行数 / 本作业的运行总数；结果在整张作业跑完后一次性推回。", ""]

    out += ["## 运行中", ""]
    if jobs_by["running"]:
        out += ["| 作业 | 课程 · 档位 | 运行数 | 领取时间 | 说明 |", "| - | - | - | - | - |"]
        for j in jobs_by["running"]:
            out.append(f"| {_job_link(j)} | {' '.join(j.get('courses') or [])} · {j.get('profile')} | {_n_runs(j)} | "
                       f"{_utc(j.get('claimed_at') or j.get('started_at'))} | {_note(j)} |")
    else:
        out.append("_没有运行中的作业。_")
    out += ["", "## 排队中", ""]
    if jobs_by["pending"]:
        out += ["| 作业 | 课程 · 档位 | 运行数 | 下发时间 | 说明 |", "| - | - | - | - | - |"]
        for j in jobs_by["pending"]:
            out.append(f"| {_job_link(j)} | {' '.join(j.get('courses') or [])} · {j.get('profile')} | {_n_runs(j)} | "
                       f"{_utc(j.get('created_at'))} | {_note(j)} |")
    else:
        out.append("_没有排队的作业。_")
    out += ["", "## 已完成", ""]
    if jobs_by["done"]:
        out += ["| 作业 | 结果 | 运行：成功 / 断言未通过 / 失败 | 耗时 | 代码版本 | 说明 |", "| - | - | - | - | - | - |"]
        for j in jobs_by["done"]:
            runs = j.get("runs") or []
            ok = sum(1 for r in runs if r.get("status") == "succeeded" and r.get("expect_ok") is not False)
            warn = sum(1 for r in runs if r.get("status") == "succeeded" and r.get("expect_ok") is False)
            bad = len(runs) - ok - warn
            hours = f"{j['hours']:.1f} 小时" if isinstance(j.get("hours"), (int, float)) else "—"
            commit = f"`{str(j.get('ran_commit') or j.get('requested_commit') or '')[:7]}`"
            out.append(f"| {_job_link(j)} | {JOB_STATUS.get(j.get('status'), j.get('status') or '—')} | "
                       f"{ok} / {warn} / {bad}（共 {_n_runs(j) or len(runs)}） | {hours} | {commit} | {_note(j)} |")
    else:
        out.append("_还没有完成的作业。_")
    if jobs_by["cancelled"]:
        out += ["", "## 已作废", "", "| 作业 | 原因 |", "| - | - |"]
        for j in jobs_by["cancelled"]:
            out.append(f"| {_job_link(j)} | {_cell(str(j.get('cancel_reason') or '')[:80])} |")

    # 场景 × 种子矩阵：每个出现过作业的 (课程, 档位)
    pairs = sorted({(c, j.get("profile")) for js in jobs_by.values() for j in js for c in j.get("courses") or []
                    if j.get("profile")})
    for cid, profile in pairs:
        try:
            course = scenarios.load_course(cid)
            specs = scenarios.expand(cid, profile)
        except (KeyError, ValueError):
            continue
        got = docs.collect(cid, profile)
        waiting = set()
        for state in ("running", "pending"):
            for j in jobs_by[state]:
                if cid in (j.get("courses") or []) and j.get("profile") == profile:
                    try:
                        waiting |= {(s.scenario, s.seed) for s in scenarios.expand(cid, profile, j.get("only") or None, j.get("seeds") or None)}
                    except (KeyError, ValueError):
                        pass
        failed = set()
        for j in reversed(jobs_by["done"]):  # 旧 → 新，新的覆盖旧的
            for r in j.get("runs") or []:
                key = (r.get("scenario"), r.get("seed"))
                if r.get("status") == "succeeded":
                    failed.discard(key)
                elif r.get("course", cid) == cid and j.get("profile") == profile:
                    failed.add(key)
        seeds = sorted({s.seed for s in specs})
        out += ["", f"## {cid} · {profile} 运行矩阵", "",
                "✅ 已出结果且断言通过 · ⚠️ 已出结果但断言未通过 · ❌ 失败 · ⏳ 在运行中或排队的作业里 · ▫️ 还没下发 · 空白 = 本场景不跑这个种子", "",
                "| 场景 | 类型 | " + " | ".join(f"s{s}" for s in seeds) + " |", "|" + " - |" * (len(seeds) + 2)]
        want = {(s.scenario, s.seed) for s in specs}
        for sc in course["scenarios"]:
            cells = []
            for seed in seeds:
                key = (sc["id"], seed)
                if key not in want:
                    cells.append("")
                    continue
                runs = [r for r in got.get(sc["id"]) or [] if r["seed"] == seed]
                if runs:
                    cells.append("⚠️" if any(r["meta"].get("expect_ok") is False for r in runs) else "✅")
                elif key in waiting:
                    cells.append("⏳")
                elif key in failed:
                    cells.append("❌")
                else:
                    cells.append("▫️")
            if any(cells):
                out.append(f"| `{sc['id']}` | {docs.ROLE_NAMES.get(sc['role'], '')} | " + " | ".join(cells) + " |")

    out += ["", "## 作业单字段", "", "| 字段 | 含义 |", "| - | - |",
            "| `job_id` | 作业编号：下发时间-课程-档位 |",
            "| `courses` / `profile` | 跑哪门课、哪个档位（档位决定模型和规模，见课程的 `scenarios.yaml`） |",
            "| `only` / `seeds` | 只跑哪些场景、哪些种子；空表示全部 |",
            "| `requested_commit` | Mac 端下发时的代码版本；H20 必须在它之后的代码上跑 |",
            "| `claimed_by` / `claimed_at` | 领取它的 agent（随机编号，不是主机名）和领取时间 |",
            "| `status` | done 全部成功 · partial 部分失败或断言未通过 · failed 全部失败 · rejected 校验不通过没执行 |",
            "| `runs` | 每个运行一行：场景、种子、结果目录、状态、耗时、断言是否通过 |",
            "| `problems` / `notes` | 失败或断言未通过的运行；H20 端的观察和说明 |",
            "| `host_summary` | GPU 型号、数量、显存、驱动/CUDA/torch 版本（不含主机名和 IP） |", ""]
    return "\n".join(out)


def write_jobs_board() -> bool:
    p = paths.JOBS_DIR / "README.md"
    text = jobs_board()
    if p.exists() and p.read_text() == text:
        return False
    p.write_text(text)
    return True
