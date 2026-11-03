"""每个测试用独立的 LAB_DATA_DIR；projects/ 作为顶层包可导入（rec、adextract），和 worker 的方式一致。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "projects"))


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("LAB_DATA_DIR", str(d))
    return d
