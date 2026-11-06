"""脱敏：替换规则、白名单导出、推送前检查（含"攻击样例"：各种藏敏感信息的写法）。"""

import json

import pytest

from labkit import paths, redact

EXTRA = ["gpu-node-17.corp.example.net", "gpu-node-17", "alice", "/mnt/secret-share"]


@pytest.mark.parametrize("text,leak", [
    ("host=gpu-node-17.corp.example.net", "gpu-node-17"),
    ("cd /home/alice/llm-rec-lab && run", "alice"),
    ("cache at /mnt/secret-share/hf", "/mnt/secret-share"),
    ("connect 10.12.3.45:22", "10.12.3.45"),
    ("ipv6 fe80:0:0:0:200:f8ff:fe21:67cf", "fe80:0:0:0:200:f8ff:fe21:67cf"),
    ("mac 00:1A:2B:3C:4D:5E", "00:1A:2B:3C:4D:5E"),
    ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "ghp_abcdefghijklmnopqrstuvwxyz0123"),
    ("HF_TOKEN=hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ("mail ops@internal-corp.cn", "ops@internal-corp.cn"),
    ("/Users/bob/Library", "bob"),
])
def test_redact_text_removes_leak(text, leak):
    out = redact.redact_text(text, EXTRA)
    assert leak not in out, out
    assert redact._scan_text(out, EXTRA) == []
    assert redact._scan_text(text, EXTRA), f"检查没有发现原文里的 {leak}"


@pytest.mark.parametrize("text", [
    "torch 2.4.1+cu124, driver 550.54.15",          # 三段版本号不是 IP
    "AUC 0.7489 LogLoss 0.5809",
    "listen on 127.0.0.1:8770",
    "commit by 123+bot@users.noreply.github.com",
    "Qwen/Qwen2.5-7B-Instruct",
])
def test_redact_keeps_harmless(text):
    assert redact.redact_text(text, EXTRA) == text
    assert redact._scan_text(text, EXTRA) == []


def _fake_run(d, extra_files=()):
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"run_id": "x", "status": "succeeded", "pid": 123,
                                            "note": "ran on gpu-node-17 by alice"}))
    (d / "env.json").write_text(json.dumps({"hostname": "gpu-node-17", "device": "cuda", "user": "alice",
                                            "ip": "10.0.0.8", "packages": {"torch": "2.4.1"}}))
    (d / "log.txt").write_text("[worker] 环境：gpu-node-17 linux\nloading /home/alice/.cache/x\n")
    (d / "metrics.jsonl").write_text("\n".join(json.dumps({"step": i, "loss": 1.0 / (i + 1)}) for i in range(5000)))
    (d / "summary.json").write_text("{}")
    (d / "artifacts").mkdir()
    (d / "artifacts" / "adapter_model.safetensors").write_bytes(b"\0" * 100)
    (d / "checkpoint_last.pt").write_bytes(b"\0" * 100)
    (d / "code" / "adextract").mkdir(parents=True)
    (d / "code" / "adextract" / "a.py").write_text("x = 1\n")
    for name, content in extra_files:
        (d / name).write_text(content)


def test_export_whitelist_and_check(tmp_path):
    src, dst = tmp_path / "batch" / "run", tmp_path / "results" / "run"
    _fake_run(src, [("secrets.txt", "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ")])
    st = redact.export_run(src, dst, EXTRA)
    names = {p.name for p in dst.rglob("*") if p.is_file()}
    assert "secrets.txt" not in names and "checkpoint_last.pt" not in names
    assert "adapter_model.safetensors" not in names
    assert "a.py" in names
    env = json.loads((dst / "env.json").read_text())
    assert "user" not in env and "ip" not in env and env["hostname"] == "<redacted>"
    meta = json.loads((dst / "run.json").read_text())
    assert "pid" not in meta and "alice" not in meta["note"]
    assert len((dst / "metrics.jsonl").read_text().splitlines()) <= redact.MAX_METRIC_LINES + 1
    assert redact.check([dst], EXTRA) == []
    assert st["bytes"] < redact.MAX_RUN


def test_check_blocks_weights_and_big_files(tmp_path):
    d = tmp_path / "results" / "r"
    d.mkdir(parents=True)
    (d / "run.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\0")
    (d / "big.txt").write_text("a" * (redact.MAX_FILE + 1))
    (d / "leak.json").write_text('{"host": "gpu-node-17"}')
    problems = {p["problem"].split(" ")[0] for p in redact.check([d], EXTRA)}
    assert "禁止的文件类型" in problems
    assert "文件" in problems
    assert "本机敏感字符串" in problems


def test_sensitive_strings_include_this_machine():
    s = redact.sensitive_strings()
    assert str(paths.ROOT) in s


def test_scan_is_linear_on_long_alnum_runs():
    """1.5 MB 的连续字母数字（像 base64）不能让正则退化成 O(n²)。"""
    import time

    text = "a" * 1_500_000 + " contact alice@corp-internal.com"
    t0 = time.perf_counter()
    hits = redact._scan_text(text, [])
    assert time.perf_counter() - t0 < 5
    assert any(h.startswith("邮箱") for h in hits)
