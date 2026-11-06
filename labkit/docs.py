"""课程 README 的自动生成部分：代码链接表和场景结果表。

README 里手写的讲解保持不动，只替换两个标记区之间的内容：

    <!-- labkit:code -->     ……     <!-- /labkit:code -->
    <!-- labkit:results -->  ……     <!-- /labkit:results -->
    <!-- labkit:figures -->  ……     <!-- /labkit:figures -->
    <!-- labkit:path -->     ……     <!-- /labkit:path -->     步骤目录（来自 lesson.yaml，见 labkit.lessonbook）

有 lesson.yaml 的课程还会生成一步一页的 steps/*.md 和 steps/figures/*.svg（labkit.lessonbook）。

图：figures/<档位>-<图名>.svg（labkit.figures 纯 Python 生成），README 的 figures 区引用它们。
冒烟档（skip_expect）不进表也不画图：规模太小，数字没有教学意义。

代码表：扫描项目里的 `# --8<-- [start:name]` 锚点，生成"锚点 → 文件#行号"的 GitHub 相对链接。
结果表：每个档位一张表，行是场景，列是该课程的关键指标（多种子时给 均值 ± 标准差 和种子数）。
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from . import paths, scenarios

ANCHOR = re.compile(r"#\s*--8<--\s*\[start:([\w-]+)\]")
HEADLINE = {
    "adextract": [("dev.exact_all_ok", "dev 全对率"), ("dev.format_ok", "dev 格式合法率"),
                  ("synth_valid.exact_all_ok", "合成验证全对率"), ("test.exact_all_ok", "test 全对率")],
}
ROLE_NAMES = {"reference": "参照", "baseline": "基线", "variant": "候选改动", "fault": "教学故障", "fix": "修复",
              "final": "最终测试"}


def code_table(project: str, course_dir: Path) -> str:
    proj = paths.PROJECTS_DIR / project
    rows = ["| 代码段 | 位置 |", "| - | - |"]
    for f in sorted(proj.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            m = ANCHOR.search(line)
            if m:
                rel = Path("..") / ".." / f.relative_to(paths.ROOT)
                rows.append(f"| `{m.group(1)}` | [{f.relative_to(paths.ROOT)}#L{i}]({rel.as_posix()}#L{i}) |")
    return "\n".join(rows)


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float)) else None


def collect(course_id: str, profile: str) -> dict[str, list[dict]]:
    """场景 → 该档位下每个种子的 {seed, summary, meta}（只取 results/ 里已发布、成功的运行）。"""
    out: dict[str, list[dict]] = {}
    base = paths.RESULTS_DIR / profile / course_id
    if not base.exists():
        return out
    for run_json in sorted(base.glob("*/*/run.json")):
        meta = json.loads(run_json.read_text())
        if meta.get("status") != "succeeded":
            continue
        summ = json.loads((run_json.parent / "summary.json").read_text()) if (run_json.parent / "summary.json").exists() else {}
        out.setdefault(meta.get("scenario") or run_json.parent.parent.name, []).append(
            {"seed": meta.get("seed"), "summary": summ, "meta": meta, "dir": run_json.parent})
    return out


def fmt_values(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f} (n={len(vals)})"


def results_table(course: dict) -> str:
    cols = HEADLINE[course["project"]]
    parts = []
    for profile, prof in course["profiles"].items():
        if prof.get("skip_expect"):
            continue
        got = collect(course["course"], profile)
        n_runs = sum(len(v) for v in got.values())
        expected = len(scenarios.expand(course["course"], profile))
        parts.append(f"**{profile}**：{prof.get('note', '')}（已有 {n_runs} / {expected} 个运行）\n")
        if not got:
            parts.append("_还没有结果：等待批量运行。_\n")
            continue
        used = [(k, n) for k, n in cols if any(_get(r["summary"], k) is not None for rs in got.values() for r in rs)]
        parts.append("| 场景 | 类型 | " + " | ".join(n for _, n in used) + " | 断言 |")
        parts.append("| - | - | " + " | ".join("-" for _ in used) + " | - |")
        for s in course["scenarios"]:
            runs = got.get(s["id"])
            if not runs:
                if not s.get("profiles") or profile in s["profiles"]:
                    parts.append(f"| `{s['id']}` | {ROLE_NAMES[s['role']]} | " + " | ".join("待生成" for _ in used) + " | |")
                continue
            vals = [fmt_values([v for r in runs if (v := _get(r["summary"], k)) is not None]) for k, _ in used]
            oks = [r["meta"].get("expect_ok") for r in runs if s.get("expect")]
            mark = "" if not oks else ("通过" if all(o is True for o in oks) else ("未通过" if any(o is False for o in oks) else "待定"))
            parts.append(f"| `{s['id']}` | {ROLE_NAMES[s['role']]} | " + " | ".join(vals) + f" | {mark} |")
        parts.append("")
    return "\n".join(parts)


def _replace(text: str, tag: str, body: str) -> str:
    start, end = f"<!-- labkit:{tag} -->", f"<!-- /labkit:{tag} -->"
    if start not in text:
        return text
    a = text.index(start) + len(start)
    b = text.index(end)
    return text[:a] + "\n" + body.strip() + "\n" + text[b:]


FIG_TITLES = {"metrics": "主指标", "loss": "train loss 曲线", "memory": "峰值显存构成", "params": "可训练参数量"}


def figures_md(course: dict, course_dir: Path, check: bool, changed: list[str]) -> str:
    """生成（或检查）figures/*.svg，返回 README figures 区的 Markdown。"""
    from . import figures

    fig_dir = course_dir / "figures"
    want: dict[Path, str] = {}
    parts = []
    for profile, prof in course["profiles"].items():
        if prof.get("skip_expect"):
            continue
        got = collect(course["course"], profile)
        if not got:
            parts.append(f"- **{profile}**：还没有结果，图在结果推回后自动生成。")
            continue
        figs = figures.course_figures(course, got, profile)
        links = []
        for name, svg in figs.items():
            p = fig_dir / f"{profile}-{name}.svg"
            want[p] = svg
            links.append(f"[{FIG_TITLES.get(name, name)}](figures/{p.name})")
        parts.append(f"- **{profile}**：" + "、".join(links))
        for name in figs:
            parts.append(f"\n  ![{profile} {FIG_TITLES.get(name, name)}](figures/{profile}-{name}.svg)")
    stale = [p for p in fig_dir.glob("*.svg") if p not in want] if fig_dir.exists() else []
    for p, svg in want.items():
        if not p.exists() or p.read_text() != svg:
            changed.append(str(p.relative_to(paths.ROOT)))
            if not check:
                fig_dir.mkdir(exist_ok=True)
                p.write_text(svg)
    for p in stale:
        changed.append(str(p.relative_to(paths.ROOT)))
        if not check:
            p.unlink()
    return "\n".join(parts)


def build_all(check: bool = False) -> list[str]:
    changed: list[str] = []
    for cid, path in scenarios.course_files().items():
        course = scenarios.load_course(cid)
        readme = path.parent / "README.md"
        if not readme.exists():
            continue
        old = readme.read_text()
        from . import lessonbook

        path_md = lessonbook.build_course(course, path.parent, check, changed)
        new = _replace(old, "code", code_table(course["project"], path.parent))
        if path_md is not None:
            new = _replace(new, "path", path_md)
        details = "<details><summary>展开结果（先自己预测，再看）</summary>\n\n" + results_table(course) + "\n</details>"
        new = _replace(new, "results", details)
        if "<!-- labkit:figures -->" in new:
            figs = figures_md(course, path.parent, check, changed)
            new = _replace(new, "figures", "<details><summary>展开证据图（先自己预测，再看）</summary>\n\n" + figs
                           + "\n\n</details>")
        if new != old:
            changed.append(str(readme.relative_to(paths.ROOT)))
            if not check:
                readme.write_text(new)
    return changed


def outputs() -> list[str]:
    """docs 会改动的路径（相对仓库根）：H20 端 publish 时一并提交。"""
    out = []
    for path in scenarios.course_files().values():
        for p in (path.parent / "README.md", path.parent / "figures", path.parent / "steps"):
            if p.exists():
                out.append(str(p.relative_to(paths.ROOT)))
    return out
