"""benchmark 工具：warmup + 多次测量 + 中位数。

为什么要这样测：
- warmup：第一次运行包含 kernel 编译、内存分配、缓存冷启动，不代表稳态性能。
- 同步：GPU 异步执行，不同步就会测到错误的时间。
- 中位数：单次测量受系统抖动影响大，中位数比平均值更稳。
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from tools.device import synchronize


# --8<-- [start:bench_result]
# @dataclass 自动生成 __init__ 等样板代码，这个类只是一个"带名字的结果记录"
@dataclass
class BenchResult:
    # 中位数是主指标；p10/p90 是第 10 和第 90 百分位，两者差得多说明测量抖动大
    median_ms: float
    p10_ms: float
    p90_ms: float
    # 实际测了几次，次数太少时中位数也不可靠
    iters: int

    # print(res) 时显示的格式
    def __str__(self) -> str:
        return f"median {self.median_ms:.3f} ms (p10 {self.p10_ms:.3f}, p90 {self.p90_ms:.3f}, n={self.iters})"
# --8<-- [end:bench_result]


# --8<-- [start:bench_warmup]
# fn 是"跑一次被测计算"的无参函数（调用方常用 lambda 包一层），返回值被忽略
def benchmark(fn: Callable[[], object], device: torch.device, warmup: int = 5, iters: int = 20) -> BenchResult:
    # 预热：前几次包含 kernel 编译/选择、显存分配、缓存冷启动，不计时
    # MPS 和 CUDA 第一次跑某个形状时可能慢 10-100 倍，5 次通常足够进入稳态
    for _ in range(warmup):
        fn()
    # 等预热发出的所有 GPU 任务真正执行完，免得它们拖进第一次计时
    synchronize(device)
# --8<-- [end:bench_warmup]

# --8<-- [start:bench_measure]
    # 每次的耗时，单位毫秒
    times = []
    for _ in range(iters):
        # 计时前同步：保证 GPU 队列是空的，t0 之后只有这一次 fn 的工作
        synchronize(device)
        # perf_counter 是 Python 里精度最高的单调时钟（纳秒级），适合测短时间间隔
        t0 = time.perf_counter()
        # 对 GPU 来说这一行只是"把 kernel 放进队列"，通常几十微秒就返回
        fn()
        # 计时后同步：阻塞到 GPU 真正算完。少了这一行，测到的只是发指令的时间
        # 代价：每次同步本身有几十微秒开销，所以测极小的 kernel 时会偏大
        synchronize(device)
        # 秒 -> 毫秒
        times.append((time.perf_counter() - t0) * 1e3)
# --8<-- [end:bench_measure]

# --8<-- [start:bench_stats]
    # 排序后按下标取百分位
    times.sort()
    return BenchResult(
        # 中位数不受偶发的系统抖动（后台进程、降频）影响，比平均值稳
        median_ms=statistics.median(times),
        # iters=20 时取第 1 个和第 17 个（从 0 数），即最快的 10% 和最慢的 10% 边界
        p10_ms=times[int(0.1 * (len(times) - 1))],
        p90_ms=times[int(0.9 * (len(times) - 1))],
        iters=iters,
    )
# --8<-- [end:bench_stats]
