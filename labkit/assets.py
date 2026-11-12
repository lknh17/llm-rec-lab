"""课程资产：公开模型。在线环境自动下载，离线环境只检查，缺什么说什么。

    python -m labkit assets                          # 列出全部资产及是否齐全
    python -m labkit assets qwen2.5-7b               # 下载/校验指定资产
    python -m labkit assets --profile h20x8          # 下载该档位所有场景需要的资产

adextract / adpref 的训练数据是仓库里的合成数据（projects/adextract/synth.py）和人工评测集（evalsets/），不需要下载；
tinylm（L01-L04）用公开数据集的单个分片；推荐课（recbase / recseq / genrec）用 MovieLens-1M（DATASETS）。
vllmserve（L09）用独立的 vLLM 环境（ENVS，和主环境的 torch 版本互不影响）。
许可：Qwen2.5（阿里）Apache-2.0；中文维基 CC-BY-SA-4.0；vLLM Apache-2.0。
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
    # L09：W4A16（AWQ）量化版，vLLM 自动走 awq_marlin kernel
    "qwen2.5-7b-awq": {"repo": "Qwen/Qwen2.5-7B-Instruct-AWQ", "size": "5.6 GB", "license": "Apache-2.0"},
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


# 独立的 Python 环境：和主环境（torch 版本由 uv.lock 锁定）依赖冲突的框架装在这里，项目用子进程调用它的解释器。
# 放在 LAB_ASSETS_DIR/venvs/<名称>--<包列表>/ 下；包列表写进目录名，改版本会新建一个环境，旧的不动。
# 用环境变量 LABKIT_<名称大写>_PYTHON 可以直接指定一个现成环境的解释器（例如 LABKIT_VLLM_PYTHON=/opt/vllm/bin/python）。
#
# vllm-env 必须同时锁 transformers：vLLM 0.11.0 只声明 transformers>=4.55.2（没有上限），不锁会装到 5.x，
# 5.x 去掉了 tokenizer.all_special_tokens_extended，vLLM 0.11 建引擎读分词器时直接报 AttributeError（2026-11-11 W1 实测）。
# probe 除了 import，还要确认 vLLM 依赖的分词器属性在，免得一个装好但用不了的环境被标成就绪。
_VLLM_PROBE = ("import vllm, torch, transformers; "
               "assert hasattr(transformers.PreTrainedTokenizerBase, 'all_special_tokens_extended'), "
               "'transformers ' + transformers.__version__ + ' 和这个 vLLM 不兼容（缺 all_special_tokens_extended）'; "
               "print(vllm.__version__, torch.__version__, torch.version.cuda, 'transformers', transformers.__version__)")
ENVS = {
    "vllm-env": {"packages": ["vllm==0.11.0", "transformers==4.57.1"], "python": "3.12",
                 "size": "约 8 GB（vLLM 自带的 torch + CUDA 运行库）", "license": "Apache-2.0", "probe": _VLLM_PROBE,
                 "override": "LABKIT_VLLM_PYTHON", "spec_override": "LABKIT_VLLM_SPEC"},
}


def uv_executable() -> str | None:
    """uv 的路径。systemd 服务用 ~/.local/bin/uv 启动 agent，但服务的 PATH 里通常没有这个目录，所以依次找：
    PATH、`uv run` 给子进程设的 UV 环境变量、默认安装位置。"""
    for c in (shutil.which("uv"), os.environ.get("UV"), str(Path.home() / ".local" / "bin" / "uv"),
              str(Path.home() / ".cargo" / "bin" / "uv")):
        if c and Path(c).is_file():
            return c
    return None


def _env_packages(name: str) -> list[str]:
    spec = ENVS[name]
    extra = os.environ.get(spec.get("spec_override", ""), "").strip()
    return extra.split() if extra else list(spec["packages"])


def env_dir(name: str) -> Path:
    tag = "-".join(p.replace("==", "-").replace("/", "_") for p in _env_packages(name))
    return assets_dir() / "venvs" / f"{name}--{tag}"


def resolve_env_python(name: str, must_exist: bool = True) -> str | None:
    """环境的解释器路径：环境变量指定的优先，其次 LAB_ASSETS_DIR/venvs 下按版本建好的。不在这里安装。"""
    spec = ENVS[name]
    override = os.environ.get(spec.get("override", ""), "").strip()
    if override:
        if Path(override).exists():
            return override
        if must_exist:
            raise FileNotFoundError(f"{spec['override']}={override} 不存在")
        return None
    py = env_dir(name) / "bin" / "python"
    if py.exists() and (env_dir(name) / ".ready").exists():
        return str(py)
    if must_exist:
        raise FileNotFoundError(f"找不到 Python 环境 {name}（{' '.join(_env_packages(name))}）。运行 python -m labkit assets {name} 安装；"
                                f"或设 {spec.get('override')} 指向一个装好的解释器")
    return None


def ensure_env(name: str) -> str:
    """建一个独立 venv 并装好依赖；装完跑一次 probe 确认能 import，成功才写 .ready。用 uv（快），没有 uv 时退回 venv + pip。"""
    found = resolve_env_python(name, must_exist=False)
    if found:
        print(f"{name}: {found} 已就绪", flush=True)
        return found
    spec, pkgs, target = ENVS[name], _env_packages(name), env_dir(name)
    print(f"安装 Python 环境 {name}：{' '.join(pkgs)}（{spec['size']}）→ {target}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    uv = uv_executable()
    py = target / "bin" / "python"
    if uv:
        subprocess.run([uv, "venv", "--python", spec["python"], "--seed", str(target)], check=True)
        subprocess.run([uv, "pip", "install", "--python", str(py), *pkgs], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(target)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", *pkgs], check=True)
    res = subprocess.run([str(py), "-c", spec["probe"]], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{name} 装好了但 import 失败：{res.stderr[-1500:]}")
    print(f"{name}: {res.stdout.strip()}", flush=True)
    (target / ".ready").write_text(res.stdout)
    return str(py)


def status() -> dict:
    out: dict = {"assets_dir": str(assets_dir())}
    for name, spec in MODELS.items():
        p = resolve_model_path(spec["repo"], must_exist=False)
        out[name] = {"kind": "model", "repo": spec["repo"], "path": p, "ok": bool(p), "size": spec["size"]}
    for name, spec in DATASETS.items():
        p = resolve_dataset_files(name, must_exist=False)
        out[name] = {"kind": "dataset", "repo": spec["repo"], "path": str(dataset_dir(name)) if p else None,
                     "ok": bool(p), "size": spec["size"]}
    for name, spec in ENVS.items():
        p = resolve_env_python(name, must_exist=False)
        out[name] = {"kind": "env", "packages": _env_packages(name), "path": p, "ok": bool(p), "size": spec["size"]}
    return out


def ensure(names: list[str]) -> None:
    failed = []
    for n in names:
        try:
            if n in MODELS or n in MODEL_ALIASES:
                ensure_model(MODEL_ALIASES.get(n, n))
            elif n in DATASETS:
                ensure_dataset(n)
            elif n in ENVS:
                ensure_env(n)
            else:
                raise KeyError(f"未知资产 {n}，可选 {list(MODELS) + list(DATASETS) + list(ENVS)}")
        except Exception as e:  # noqa: BLE001
            failed.append(n)
            print(f"[失败] {n}: {e}", file=sys.stderr, flush=True)
    if failed:
        raise SystemExit(f"以下资产没有就绪：{failed}")
