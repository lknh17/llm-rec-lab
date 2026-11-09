"""课程资产：公开模型。在线环境自动下载，离线环境只检查，缺什么说什么。

    python -m labkit assets                          # 列出全部资产及是否齐全
    python -m labkit assets qwen2.5-7b               # 下载/校验指定资产
    python -m labkit assets --profile h20x8          # 下载该档位所有场景需要的资产

adextract / adpref 的训练数据是仓库里的合成数据（projects/adextract/synth.py）和人工评测集（evalsets/），不需要下载；
tinylm（L01-L04）用公开数据集的单个分片；推荐课（recbase / recseq / genrec）用 MovieLens-1M（DATASETS）。
许可：Qwen2.5（阿里）Apache-2.0；中文维基 CC-BY-SA-4.0。
国内网络：设 HF_ENDPOINT=https://hf-mirror.com，或装 modelscope 后用 LABKIT_MODEL_SOURCE=modelscope。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import assets_dir

MODELS = {
    "qwen2.5-0.5b": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "size": "1 GB", "license": "Apache-2.0"},
    "qwen2.5-7b": {"repo": "Qwen/Qwen2.5-7B-Instruct", "size": "15 GB", "license": "Apache-2.0"},
}
MODEL_ALIASES = {v["repo"]: k for k, v in MODELS.items()}


def resolve_model_path(repo: str, must_exist: bool = True) -> str | None:
    """优先 LAB_ASSETS_DIR/models/<repo>，其次 HuggingFace 本地缓存。不在这里联网。"""
    local = assets_dir() / "models" / repo
    if (local / "config.json").exists():
        return str(local)
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        snaps = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots"
        for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (snap / "config.json").exists() and any(snap.glob("*.safetensors")) and (snap / "tokenizer.json").exists():
                return str(snap)
    except Exception:  # noqa: BLE001
        pass
    if must_exist:
        raise FileNotFoundError(
            f"本地找不到模型 {repo}。运行 python -m labkit assets {MODEL_ALIASES.get(repo, repo)} 下载；"
            f"或把模型目录放到 {local}（需含 config.json、tokenizer.json 和 safetensors 权重）")
    return None


def ensure_model(name: str) -> str:
    repo = MODELS[name]["repo"] if name in MODELS else name
    found = resolve_model_path(repo, must_exist=False)
    if found:
        print(f"{repo}: {found} 已存在", flush=True)
        return found
    target = assets_dir() / "models" / repo
    source = os.environ.get("LABKIT_MODEL_SOURCE", "hf")
    print(f"下载 {repo}（{MODELS.get(name, {}).get('size', '?')}）来源={source} "
          f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}", flush=True)
    if source == "modelscope":
        exe = shutil.which("modelscope")
        if not exe:
            raise RuntimeError("LABKIT_MODEL_SOURCE=modelscope 需要先 uv pip install modelscope")
        subprocess.run([exe, "download", "--model", repo, "--local_dir", str(target)], check=True)
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=str(target), allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"])
    if not (target / "config.json").exists():
        raise RuntimeError(f"{repo} 下载后缺少 config.json：{target}")
    return str(target)


# 公开数据集：只下载指定文件（一个 parquet 分片），放在 LAB_ASSETS_DIR/datasets/<名称>/ 下
DATASETS = {
    "zhwiki-2311-s2": {"repo": "wikimedia/wikipedia", "files": ["20231101.zh/train-00002-of-00006.parquet"],
                       "size": "127 MB", "license": "CC-BY-SA-4.0",
                       "note": "中文维基 2023-11-01 快照的第 3 个分片（约 13 万篇，繁简混排，原文未转换）"},
    "ml-1m": {"repo": "hazemessam/ml-1m", "files": ["ratings.csv", "users.csv", "movies.csv"], "size": "约 25 MB",
              "license": "GroupLens 使用条款（研究用途，不可再分发）",
              "note": "MovieLens-1M 的 CSV 转写（6040 用户、3883 部电影、100 万条评分）；timestamp 字段带引号且含换行"},
}


def dataset_dir(name: str) -> Path:
    return assets_dir() / "datasets" / name


def resolve_dataset_files(name: str, must_exist: bool = True) -> list[str] | None:
    """数据集文件的本地路径（全部存在才返回）。不在这里联网。"""
    spec = DATASETS[name]
    files = [dataset_dir(name) / f for f in spec["files"]]
    if all(f.exists() and f.stat().st_size > 0 for f in files):
        return [str(f) for f in files]
    if must_exist:
        raise FileNotFoundError(f"本地找不到数据集 {name}（{spec['repo']}）。运行 python -m labkit assets {name} 下载；"
                                f"或把 {spec['files']} 放到 {dataset_dir(name)}")
    return None


def ensure_dataset(name: str) -> list[str]:
    found = resolve_dataset_files(name, must_exist=False)
    if found:
        print(f"{name}: {dataset_dir(name)} 已存在", flush=True)
        return found
    spec = DATASETS[name]
    print(f"下载数据集 {spec['repo']} {spec['files']}（{spec['size']}）"
          f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}", flush=True)
    from huggingface_hub import hf_hub_download

    for f in spec["files"]:
        hf_hub_download(spec["repo"], f, repo_type="dataset", local_dir=str(dataset_dir(name)))
    return resolve_dataset_files(name)


def status() -> dict:
    out: dict = {"assets_dir": str(assets_dir())}
    for name, spec in MODELS.items():
        p = resolve_model_path(spec["repo"], must_exist=False)
        out[name] = {"kind": "model", "repo": spec["repo"], "path": p, "ok": bool(p), "size": spec["size"]}
    for name, spec in DATASETS.items():
        p = resolve_dataset_files(name, must_exist=False)
        out[name] = {"kind": "dataset", "repo": spec["repo"], "path": str(dataset_dir(name)) if p else None,
                     "ok": bool(p), "size": spec["size"]}
    return out


def ensure(names: list[str]) -> None:
    failed = []
    for n in names:
        try:
            if n in MODELS or n in MODEL_ALIASES:
                ensure_model(MODEL_ALIASES.get(n, n))
            elif n in DATASETS:
                ensure_dataset(n)
            else:
                raise KeyError(f"未知资产 {n}，可选 {list(MODELS) + list(DATASETS)}")
        except Exception as e:  # noqa: BLE001
            failed.append(n)
            print(f"[失败] {n}: {e}", file=sys.stderr, flush=True)
    if failed:
        raise SystemExit(f"以下资产没有就绪：{failed}")
