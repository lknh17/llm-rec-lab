"""在 vllm-env 里运行：从已安装的 vLLM 源码里摘出讲义 S5.5 要核对的几处实现，打印并写 vllm_facts.md。

1. 前缀缓存的块哈希什么时候登记、投机解码的草稿 token 会不会被登记（KVCacheManager.allocate_slots 的源码）；
2. 投机解码的指标名和日志里"平均接受长度"的写法；
3. 前缀缓存命中的统计口径（prefix_cache 相关的指标）。
只读源码，不建引擎、不占显存。
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import vllm

ROOT = Path(vllm.__file__).parent
PATTERNS = {
    "投机解码日志与指标": (r"Mean acceptance length|spec_decode_num_accepted|num_accepted_tokens", ["v1/spec_decode", "v1/metrics"]),
    "前缀缓存登记（cache_blocks / num_tokens_to_cache）": (r"cache_blocks\(|cache_full_blocks\(|num_tokens_to_cache", ["v1/core"]),
    "前缀缓存命中统计": (r"prefix_cache_(queries|hits)|PrefixCacheStats", ["v1/metrics", "v1/core"]),
}


def grep(pattern: str, subdirs: list[str], limit: int = 30) -> list[str]:
    rx, out = re.compile(pattern), []
    for sub in subdirs:
        for f in sorted((ROOT / sub).rglob("*.py")):
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{f.relative_to(ROOT.parent)}:{i}: {line.strip()[:200]}")
    return out[:limit]


def main() -> None:
    lines = [f"# vLLM {vllm.__version__} 源码核对", ""]
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        src = inspect.getsource(KVCacheManager.allocate_slots).splitlines()
        lines += ["## KVCacheManager.allocate_slots", "", "```python", *src[:160], "```", ""]
    except Exception as e:  # noqa: BLE001
        lines += [f"## KVCacheManager.allocate_slots 取不到：{e!r}", ""]
    for title, (pat, subs) in PATTERNS.items():
        lines += [f"## {title}", "", "```", *grep(pat, subs), "```", ""]
    text = "\n".join(lines)
    print(text)
    Path("vllm_facts.md").write_text(text)


if __name__ == "__main__":
    main()
