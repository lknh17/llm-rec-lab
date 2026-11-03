"""labkit 命令行。H20 端的 agent 只需要用到这里的命令（见 AGENT_RUNBOOK.md）。

    python -m labkit assets [名称 ...] [--profile h20x8]     下载/检查公开数据集和模型
    python -m labkit probe [--compute]                       环境预检（--compute 做一次真实前向反向）
    python -m labkit scenarios [课程] [--profile P]           列出场景和每个场景展开后的运行
    python -m labkit run R1 baseline [--profile P] [--seed 0] 前台跑单个场景（调试用）
    python -m labkit batch R1 L1 --profile h20x8 [--only ..] [--seeds ..] [--dry-run] [--max-slots N]
    python -m labkit export [--profile P]                    batch/ → results/（脱敏白名单）
    python -m labkit redact --check [路径 ...]                推送前的脱敏检查
    python -m labkit jobs list | new R1 L1 --profile h20x8 | run <job_id> | show <job_id>
    python -m labkit publish -m "说明"                        检查 + 提交 results/ jobs/ + 推送
    python -m labkit docs [--check]                          重新生成课程 README 里的代码链接和结果表
"""

from __future__ import annotations

import argparse
import json
import sys

from . import paths


def _courses(args_courses: list[str], all_flag: bool = False) -> list[str]:
    from .scenarios import course_files

    known = list(course_files())
    if all_flag or not args_courses:
        return known
    bad = [c for c in args_courses if c not in known]
    if bad:
        sys.exit(f"未知课程 {bad}，可选 {known}")
    return args_courses


def cmd_assets(a) -> None:
    from . import assets, scenarios

    names = list(a.names)
    if a.profile:
        for c in _courses([], True):
            prof = scenarios.load_course(c)["profiles"].get(a.profile)
            if prof:
                names += [n for n in prof.get("assets", []) if n not in names]
    if not names:
        print(json.dumps(assets.status(), ensure_ascii=False, indent=2))
        return
    assets.ensure(names)


def cmd_probe(a) -> None:
    from .probe import probe

    print(json.dumps(probe(compute=a.compute), ensure_ascii=False, indent=2))


def cmd_scenarios(a) -> None:
    from . import scenarios

    for c in _courses(a.courses):
        course = scenarios.load_course(c)
        profiles = [a.profile] if a.profile else list(course["profiles"])
        print(f"== {c} {course.get('title', '')}")
        for p in profiles:
            if p not in course["profiles"]:
                continue
            specs = scenarios.expand(c, p)
            print(f"  [{p}] {len(specs)} 个运行  {course['profiles'][p].get('note', '')}")
            for s in specs:
                changed = {k: v for k, v in s.cfg.items()
                           if k in {**next(x for x in course['scenarios'] if x['id'] == s.scenario).get('params', {})}}
                print(f"    {s.scenario:<22} s{s.seed} {s.role:<9} {s.rel_dir}  {changed or ''}")


def cmd_run(a) -> None:
    from . import batch, scenarios

    specs = scenarios.expand(a.course, a.profile, [a.scenario], [a.seed])
    if not specs:
        sys.exit(f"{a.course}/{a.scenario} 在档位 {a.profile} 下没有种子 {a.seed}")
    spec = specs[0]
    run = batch.prepare_run(spec, batch.git_commit())
    print(f"运行目录 {run.dir}", flush=True)
    import threading

    status = batch.execute(run, batch.Slot("fg", None if not a.gpu else a.gpu), threading.Event())
    ok = batch.check_expect(run, spec, allow_missing_base=False) if status == "succeeded" else None
    print(run.log_tail(3000))
    print(json.dumps({"status": status, "expect_ok": ok, "expect": run.meta().get("expect_results")},
                     ensure_ascii=False, indent=2))


def cmd_batch(a) -> None:
    from . import batch

    report = batch.run_batch(_courses(a.courses, a.all), a.profile, a.only or None, a.seeds or None,
                             dry_run=a.dry_run, max_slots=a.max_slots, force=a.force)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


def cmd_export(a) -> None:
    from . import redact

    out = redact.export_all(a.profile, a.courses or None)
    total = sum(r["bytes"] for r in out)
    print(f"导出 {len(out)} 个运行到 results/，共 {total / 1e6:.1f} MB")
    for r in out:
        print(f"  {r['dir']}  {r['status']}  {r['bytes'] / 1e3:.0f} KB  未导出：{r['skipped'] or '无'}")


def cmd_redact(a) -> None:
    from pathlib import Path

    from . import redact

    targets = [Path(p).resolve() for p in a.paths] or None
    problems = redact.check(targets)
    if problems:
        for p in problems[:50]:
            print(f"✗ {p['path']}: {p['problem']}")
        sys.exit(f"脱敏检查不通过：{len(problems)} 处")
    print("脱敏检查通过")


def cmd_jobs(a) -> None:
    from . import jobs

    if a.action == "list":
        for j in jobs.list_jobs():
            print(f"{j['_state']:<8} {j['job_id']:<40} {j.get('profile')} {j.get('courses')} {j.get('status', '')}")
    elif a.action == "show":
        print(json.dumps(jobs.load(a.args[0]), ensure_ascii=False, indent=2, default=str))
    elif a.action == "new":
        from .batch import git_commit

        p = jobs.new_job(_courses(a.args), a.profile, a.only or None, a.seeds or None, a.note,
                         requested_commit=git_commit()["commit"])
        print(f"已写入 {p.relative_to(paths.ROOT)}。提交并推送后，H20 端执行 python -m labkit jobs run {p.stem}")
    elif a.action == "run":
        from .publish import publish

        job_id = a.args[0] if a.args else next((j["job_id"] for j in jobs.list_jobs() if j["_state"] == "pending"), None)
        if not job_id:
            sys.exit("没有待处理的作业（jobs/pending/ 为空）")
        res = jobs.run_job(job_id, a.max_slots, publish_each=not a.no_publish)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if not a.no_publish:
            publish(f"results: {job_id} {res['status']}")
    else:
        sys.exit(f"未知动作 {a.action}")


def cmd_publish(a) -> None:
    from .publish import publish

    publish(a.message, push=not a.no_push)


def cmd_docs(a) -> None:
    from . import docs

    changed = docs.build_all(check=a.check)
    if a.check and changed:
        sys.exit(f"课程 README 不是最新的：{changed}。运行 python -m labkit docs")
    print("课程 README 已是最新" if not changed else f"已更新 {changed}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="labkit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("assets")
    p.add_argument("names", nargs="*")
    p.add_argument("--profile")
    p.set_defaults(fn=cmd_assets)

    p = sub.add_parser("probe")
    p.add_argument("--compute", action="store_true")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("scenarios")
    p.add_argument("courses", nargs="*")
    p.add_argument("--profile")
    p.set_defaults(fn=cmd_scenarios)

    p = sub.add_parser("run")
    p.add_argument("course")
    p.add_argument("scenario")
    p.add_argument("--profile", default="mac-m4pro")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES，例如 0")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("batch")
    p.add_argument("courses", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--profile", required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-slots", type=int)
    p.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    p.add_argument("--report", help="把逐运行报告写到这个 json 文件")
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("export")
    p.add_argument("--profile")
    p.add_argument("--courses", nargs="*")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("redact")
    p.add_argument("--check", action="store_true", required=True)
    p.add_argument("paths", nargs="*")
    p.set_defaults(fn=cmd_redact)

    p = sub.add_parser("jobs")
    p.add_argument("action", choices=("list", "show", "new", "run"))
    p.add_argument("args", nargs="*")
    p.add_argument("--profile", default="h20x8")
    p.add_argument("--only", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--note", default="")
    p.add_argument("--max-slots", type=int)
    p.add_argument("--no-publish", action="store_true")
    p.set_defaults(fn=cmd_jobs)

    p = sub.add_parser("publish")
    p.add_argument("-m", "--message", required=True)
    p.add_argument("--no-push", action="store_true")
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("docs")
    p.add_argument("--check", action="store_true")
    p.set_defaults(fn=cmd_docs)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
