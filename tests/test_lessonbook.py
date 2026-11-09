"""GitHub 一步一页讲义（lessonbook）和作业看板。"""

import json

import pytest
import yaml

from labkit import docs, jobs, lessonbook, paths, scenarios

COURSE_DIR = paths.COURSES_DIR / "L1_lora_extraction"


@pytest.fixture(scope="module")
def lesson():
    return yaml.safe_load((COURSE_DIR / "lesson.yaml").read_text())


def test_lesson_has_no_answers(lesson):
    """公开讲义只放问题和选项：答案、解析、判定区间都留在私有仓库。"""
    text = (COURSE_DIR / "lesson.yaml").read_text()
    for key in ("answer:", "explain:", "answer_from:", "range:"):
        assert key not in text, f"lesson.yaml 不能有 {key}"
    for st in lesson["steps"]:
        for q in ([st["question"]] if st.get("question") else []) + (st.get("questions") or []):
            assert 2 <= len(q["options"]) <= 4 and all(isinstance(o, str) for o in q["options"])


def test_lesson_matches_scenarios_and_code(lesson):
    course = scenarios.load_course("L1")
    ids = {s["id"] for s in course["scenarios"]}
    used = set()
    for st in lesson["steps"]:
        for sid in (st.get("scenarios") or []) + (st.get("also") or []) + ([st["compare_to"]] if st.get("compare_to") else []):
            assert sid in ids, f"{st['id']} 引用了不存在的场景 {sid}"
            used.add(sid)
        assert st["stage"] in {s["id"] for s in lesson["stages"]}
    assert used == ids, f"这些场景没有出现在任何一步：{ids - used}"
    anchors = lessonbook._anchors(course["project"])
    for st in lesson["steps"]:
        for c in (st.get("code") or []) + [n["code"] for n in st.get("nodes") or [] if n.get("code")]:
            assert (c["file"], c["region"]) in anchors, f"代码锚点不存在：{c}"
    assert len({st["file"] for st in lesson["steps"]}) == len(lesson["steps"])


def test_step_pages_are_current_and_complete(lesson):
    """steps/ 和 README 的步骤目录是最新的，每步一页，页面里没有没替换的模板。"""
    assert docs.build_all(check=True) == []
    pages = sorted((COURSE_DIR / "steps").glob("*.md"))
    assert [p.stem for p in pages] == [st["file"] for st in lesson["steps"]]
    for p in pages:
        text = p.read_text()
        assert "{{" not in text and "../README.md" in text
    readme = (COURSE_DIR / "README.md").read_text()
    assert all(f"steps/{st['file']}.md" in readme for st in lesson["steps"])


def test_render_falls_back_with_label(lesson):
    book = lessonbook.Book(scenarios.load_course("L1"), lesson, COURSE_DIR)
    assert book.render("{{v:no_such:summary.dev.exact_all_ok}}") == lessonbook.PENDING
    if book.got["mac-m4pro"].get("baseline") and not book.got["h20x8"].get("baseline"):
        out = book.render("监督占比 {{v:baseline:meta.data.tokenized.supervised_tokens_total/meta.data.tokenized.tokens_total:pct}}")
        assert out.endswith("（数字来自 Mac 历史档）") and "%" in out


def test_fmt():
    f = lessonbook.fmt
    assert f(0.6667, pct=True) == "66.7%"
    assert f(40370176, "params") == "4,037 万" and f(1.6e8, "params") == "1.60 亿"
    assert f(1234.4, ",d") == "1,234" and f(0.5, signed=True) == "+0.5"
    assert lessonbook.fmt_agg([0.6, 0.7, 0.8], pct=True) == "70.0% ± 10.0"
    assert lessonbook.fmt_delta(-0.2, {"pct": True}) == "-20.0 个点"


def test_jobs_board(tmp_path, monkeypatch):
    jd = tmp_path / "jobs"
    for s in ("pending", "running", "done", "cancelled"):
        (jd / s).mkdir(parents=True)
    monkeypatch.setattr(paths, "JOBS_DIR", jd)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(docs, "collect", lambda course_id, profile: {})  # 和仓库里真实的 results/ 隔离
    (tmp_path / "status").mkdir()
    (tmp_path / "status" / "h20x8.json").write_text(json.dumps(
        {"profile": "h20x8", "state": "running", "job_id": "j2", "progress": "5/34", "gpu": "NVIDIA H20", "gpu_count": 8,
         "updated_at": 1793952769.8}))
    jobs.new_job(["L1"], "h20x8", only=["baseline"], seeds=[0, 1], job_id="j1")
    jobs.new_job(["L1"], "h20x8", job_id="j2")
    jobs.move("j2", "running", claimed_by="agent-x", claimed_at="2026-11-06T07:57:39Z")
    jobs.new_job(["L1"], "h20x8", only=["lora_r4"], seeds=[2], job_id="j0")
    jobs.move("j0", "done", status="partial", hours=1.2, ran_commit="abcdef1234",
              runs=[{"course": "L1", "scenario": "lora_r4", "seed": 2, "status": "failed"}])
    jobs.new_job(["L1"], "h20x8", job_id="jx")
    jobs.cancel("jx", "测试作废")
    text = lessonbook.jobs_board()
    assert "| h20x8 | 运行中 | j2 | 5/34 | 8 × NVIDIA H20 |" in text
    assert "[`j2`](running/j2.yaml)" in text and "[`j1`](pending/j1.yaml)" in text
    assert "部分成功 | 0 / 0 / 1（共 1）" in text and "`abcdef1`" in text and "测试作废" in text
    assert "## L1 · h20x8 运行矩阵" in text
    row = next(line for line in text.splitlines() if line.startswith("| `lora_r4`"))
    assert row.count("⏳") == 3  # j2 是全量作业，还在跑：失败的 s2 也重新排上了
    assert lessonbook.write_jobs_board() is True and lessonbook.write_jobs_board() is False


def test_fmt_signed_with_explicit_sign_spec():
    """讲义里差值列写成 fmt: +.3f 时，fmt_delta 不能再加一个 +（H20 作业 20261109-1317 因 '++.3f' 失败）。"""
    from labkit.lessonbook import fmt, fmt_delta

    assert fmt(0.0123, "+.3f", signed=True) == "+0.012"
    assert fmt_delta(-0.0123, {"fmt": "+.3f"}) == "-0.012"
    assert fmt_delta(0.5, {"fmt": ".2f"}) == "+0.50"
    assert fmt_delta(3, {"fmt": "d"}) == "+3"
