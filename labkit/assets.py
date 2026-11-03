"""课程资产：公开数据集与模型。在线环境自动下载并校验，离线环境只检查，缺什么说什么。

    python -m labkit assets                          # 列出全部资产及是否齐全
    python -m labkit assets ml-1m qwen2.5-0.5b       # 下载/校验指定资产
    python -m labkit assets --profile h20x8          # 下载该档位所有场景需要的资产

所有地址都是公开地址（2026-11 核实）。许可：
- MovieLens（GroupLens）：仅限研究、不得再分发。本仓库不分发原始数据，结果里只有聚合指标和少量样例行。
- KuaiRand-Pure（快手，CIKM'22）：CC BY-SA 4.0，引用 Gao et al. 2022。
- Qwen2.5（阿里）：Apache-2.0。
国内网络：模型设 HF_ENDPOINT=https://hf-mirror.com，或装 modelscope 后用 LABKIT_MODEL_SOURCE=modelscope。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .paths import assets_dir

DATASETS = {
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "dir": "ml-1m", "required": ["ratings.dat", "users.dat", "movies.dat"],
        "md5_file": "ratings.dat", "md5": "a89aa3591bc97d6d4e0c89459ff39362",
        "size": "6 MB", "license": "GroupLens 研究许可，不得再分发",
    },
    "ml-20m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "dir": "ml-20m", "required": ["ratings.csv", "movies.csv"],
        "archive_md5": "cd245b17a1ae2cc31bb14903e1204af3",
        "size": "190 MB 压缩包，解压后 ~830 MB", "license": "GroupLens 研究许可，不得再分发；无用户画像",
    },
    "kuairand-pure": {
        "url": "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
        "mirror_note": "国内镜像（网页手动下载）：https://rec.ustc.edu.cn/share/7f5a44c0-c196-11ed-b626-81cd6a642112",
        "dir": "KuaiRand-Pure/data",
        "required": ["log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv",
                     "user_features_pure.csv", "video_features_basic_pure.csv"],
        "archive_md5": "0820331067a3784d9691136f772b35a7",
        "size": "47 MB 压缩包，解压后 ~194 MB", "license": "CC BY-SA 4.0",
    },
}

MODELS = {
    "qwen2.5-0.5b": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "size": "1 GB", "license": "Apache-2.0"},
    "qwen2.5-7b": {"repo": "Qwen/Qwen2.5-7B-Instruct", "size": "15 GB", "license": "Apache-2.0"},
}
MODEL_ALIASES = {v["repo"]: k for k, v in MODELS.items()}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_path(name: str) -> Path:
    return assets_dir() / "datasets" / DATASETS[name]["dir"]


def dataset_ok(name: str) -> bool:
    d = dataset_path(name)
    return all((d / f).exists() for f in DATASETS[name]["required"])


def ensure_dataset(name: str) -> Path:
    spec = DATASETS[name]
    root = assets_dir() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    if not dataset_ok(name):
        archive = root / Path(spec["url"]).name
        if not archive.exists():
            print(f"下载 {spec['url']}（{spec['size']}）", flush=True)
            tmp = archive.with_suffix(archive.suffix + ".part")
            urllib.request.urlretrieve(spec["url"], tmp)
            tmp.rename(archive)
        if spec.get("archive_md5"):
            got = _md5(archive)
            if got != spec["archive_md5"]:
                archive.unlink()
                raise RuntimeError(f"{name} 压缩包 md5 不一致：{got} != {spec['archive_md5']}，已删除，请重试")
        print(f"解压 {archive.name}", flush=True)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(root)
        else:
            with tarfile.open(archive) as t:
                t.extractall(root, filter="data")
    if spec.get("md5_file"):
        got = _md5(dataset_path(name) / spec["md5_file"])
        if got != spec["md5"]:
            raise RuntimeError(f"{name}/{spec['md5_file']} md5 {got} 与课程版本 {spec['md5']} 不一致")
    print(f"{name}: {dataset_path(name)} 齐全（{spec['license']}）", flush=True)
    return dataset_path(name)


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


def status() -> dict:
    out = {"assets_dir": str(assets_dir())}
    for name, spec in DATASETS.items():
        out[name] = {"kind": "dataset", "path": str(dataset_path(name)), "ok": dataset_ok(name), "size": spec["size"]}
    for name, spec in MODELS.items():
        p = resolve_model_path(spec["repo"], must_exist=False)
        out[name] = {"kind": "model", "repo": spec["repo"], "path": p, "ok": bool(p), "size": spec["size"]}
    return out


def ensure(names: list[str]) -> None:
    failed = []
    for n in names:
        try:
            if n in DATASETS:
                ensure_dataset(n)
            elif n in MODELS or n in MODEL_ALIASES:
                ensure_model(MODEL_ALIASES.get(n, n))
            else:
                raise KeyError(f"未知资产 {n}，可选 {list(DATASETS) + list(MODELS)}")
        except Exception as e:  # noqa: BLE001
            failed.append(n)
            print(f"[失败] {n}: {e}", file=sys.stderr, flush=True)
            if n in DATASETS and DATASETS[n].get("mirror_note"):
                print("  " + DATASETS[n]["mirror_note"], file=sys.stderr)
    if failed:
        raise SystemExit(f"以下资产没有就绪：{failed}")
