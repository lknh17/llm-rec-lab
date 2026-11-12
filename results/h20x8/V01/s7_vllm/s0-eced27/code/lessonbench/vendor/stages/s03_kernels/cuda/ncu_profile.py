"""用 Nsight Compute（ncu）给第 5 课的 v1-v5 各采一次关键指标，解析 CSV 成表格，并按"指标 -> 瓶颈 -> 改法"自动诊断。

GPU 上（需要 ncu 在 PATH 里，且有读取性能计数器的权限）：
    uv run python -m stages.s03_kernels.cuda.ncu_profile                    # 采集 v1-v5，写 experiments/ncu_matmul_<hw>.md
    uv run python -m stages.s03_kernels.cuda.ncu_profile --print-only       # 只打印 ncu 命令行，自己去跑
    uv run python -m stages.s03_kernels.cuda.ncu_profile --full v4          # --set full 生成 .ncu-rep，用 ncu-ui 打开看
Mac 上（只解析，不采集）：
    uv run python -m stages.s03_kernels.cuda.ncu_profile --parse stages/s03_kernels/experiments/ncu_typical_values.csv
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from tabulate import tabulate

EXP = Path(__file__).resolve().parents[1] / "experiments"

# --8<-- [start:metrics]
# 采集的指标：显示名 -> ncu 指标名（ncu 2023+ 的命名；不同版本可能改名，用 `ncu --query-metrics` 核对）
METRICS = {
    "耗时 us": "gpu__time_duration.sum",
    # SOL（Speed Of Light）：各单元吞吐占峰值的百分比，取最忙的那个单元
    # 计算 SOL：SM 内各条管线（FMA、ALU、Tensor、LSU 发射……）里最忙的那个
    "SOL 计算 %": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # 访存 SOL：L1/TEX、shared memory、L2、DRAM 里最忙的那个——它高不代表 DRAM 忙
    "SOL 访存 %": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "DRAM 吞吐 %": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "L1 命中 %": "l1tex__t_sector_hit_rate.pct",
    "L2 命中 %": "lts__t_sector_hit_rate.pct",
    # 理论 occupancy（由寄存器/smem/block 大小算出，和 tools/occupancy.py 同一套规则）vs 实际跑出来的
    "理论 occupancy %": "sm__maximum_warps_per_active_cycle_pct",
    "实际 occupancy %": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "寄存器/线程": "launch__registers_per_thread",
    # 全局读合并情况：每条 warp 级 load 平均触发几个 sector（每线程 4 字节、完美合并时是 4）
    "全局读 sector/请求": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    # 全局读效率：线程真正要的字节 / 搬运的 sector 字节（nvprof 时代的 gld_efficiency），100% 是完美合并
    "全局读效率 %": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    # shared memory 读：bank conflict 次数 / 读指令数 = 平均每条指令多排队几轮
    "smem 读 bank conflict": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem 读指令": "smsp__inst_executed_op_shared_ld.sum",
    # 管线利用率：Tensor Core（mma.sync 走 hmma 管线；Hopper 的 wgmma 看 ..._gmma_...）和 FP32 FMA
    "Tensor Core 活跃 %": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "FMA 管线 %": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
}
# warp 停顿原因：停在该原因上的 warp 数（对所有周期求和）/ 发射了指令的周期数，
# 即一个 warp 平均每发一条指令要在这个原因上等几个周期（越大越说明它是瓶颈；可以大于驻留 warp 数）
STALLS = {
    "long_scoreboard": "等 L1TEX（全局/local memory）返回",
    "barrier": "等 __syncthreads",
    "mio_throttle": "shared memory / 特殊指令队列满",
    "short_scoreboard": "等 shared memory 返回",
    "lg_throttle": "全局访存指令队列满",
    "math_pipe_throttle": "计算管线忙",
    "not_selected": "有 warp 可发射但没被选中",
    "wait": "固定延迟依赖",
}
for _reason in STALLS:
    METRICS[f"stall {_reason}"] = f"smsp__average_warps_issue_stalled_{_reason}_per_issue_active.ratio"
# --8<-- [end:metrics]


# --8<-- [start:ncu_command]
def ncu_command(version: str, size: int = 4096, full: bool = False, launch_skip: int = 2, launch_count: int = 1,
                out_dir: Path = EXP) -> list[str]:
    """拼出一条 ncu 命令：只采 matmul_<version>_* 这个 kernel 的第 launch_skip+1 次启动。"""
    cmd = [
        "ncu",
        # 被测进程如果再 fork 子进程（比如编译扩展），也要跟进去
        "--target-processes", "all",
        # -k：只采名字匹配的 kernel；regex: 前缀表示正则。kernel 名形如 matmul_v4_regtile_kernel
        "-k", f"regex:matmul_{version}_",
        # 跳过前 2 次（第一次含 JIT、冷缓存），只采 1 次：ncu 会把一次 kernel 重放几十遍来收集所有计数器
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
    ]
    if full:
        # --set full：所有 section（SOL、Memory Workload、Warp State、Source Counters……），写成 .ncu-rep
        # 用 ncu-ui 打开能看到逐行源码的停顿热点；-f 覆盖已有文件
        cmd += ["--set", "full", "-f", "-o", str(out_dir / f"ncu_matmul_{version}")]
    else:
        # 只采指定指标，CSV 打到 stdout；--page raw 是"一个 kernel 一行、一个指标一列"的格式
        # --print-units base：数值一律用基本单位（耗时是 ns），否则 ncu 会自动换成 us/ms，解析时单位不统一
        cmd += ["--metrics", ",".join(METRICS.values()), "--csv", "--page", "raw", "--print-units", "base"]
    # 被测程序：本脚本的 --run 模式，只启动 launch_skip + launch_count 次该版本 kernel
    cmd += [sys.executable, "-m", "stages.s03_kernels.cuda.ncu_profile", "--run", version, "--size", str(size),
            "--times", str(launch_skip + launch_count)]
    return cmd
# --8<-- [end:ncu_command]


def run_target(version: str, size: int, times: int) -> None:
    """被 ncu 包住运行的部分：编译扩展、准备输入、启动 kernel times 次。"""
    import torch

    from stages.s03_kernels.cuda.matmul_ext import VERSIONS, matmul

    dtype = VERSIONS[version][1]
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    for _ in range(times):
        matmul(a, b, version)
    torch.cuda.synchronize()


# --8<-- [start:parse]
def _to_number(s: str) -> float | str:
    """ncu 的数值可能带千分位逗号（"1,024"），也可能是 "n/a"；能转成数就转。"""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return s


def parse_ncu_csv(text: str) -> list[dict]:
    """把 ncu --csv 的输出解析成"每个 kernel 一个 dict（指标名 -> 值）"。支持两种格式：

    - --page raw：表头一行（ID、Kernel Name、指标名……），第二行是单位（第一格为空），之后每个 kernel 一行
    - --page details：长表，每行一个指标（列 Kernel Name、Metric Name、Metric Value……）
    stdout 里混着 "==PROF==" 日志和被测程序的输出，只保留以双引号开头的 CSV 行。"""
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    if not lines:
        return []
    rows = list(csv.reader(lines))
    header, body = rows[0], rows[1:]
    if "Metric Name" in header:
        # 长表：按 (ID, Kernel Name) 聚合成宽表
        i_id, i_name = header.index("ID"), header.index("Kernel Name")
        i_metric, i_value = header.index("Metric Name"), header.index("Metric Value")
        kernels: dict[tuple, dict] = {}
        for r in body:
            k = kernels.setdefault((r[i_id], r[i_name]), {"ID": r[i_id], "Kernel Name": r[i_name]})
            k[r[i_metric]] = _to_number(r[i_value])
        return list(kernels.values())
    # 宽表：跳过单位行（ID 那一格为空）
    out = []
    for r in body:
        if not r or r[0] == "":
            continue
        out.append({h: (_to_number(v) if h not in ("ID", "Kernel Name") else v) for h, v in zip(header, r)})
    return out
# --8<-- [end:parse]


# --8<-- [start:diagnose]
def _get(m: dict, short: str) -> float | None:
    v = m.get(METRICS[short])
    return v if isinstance(v, float) else None


def diagnose(m: dict) -> list[str]:
    """把一个 kernel 的指标翻译成"瓶颈 -> 改法"。阈值是经验值，用来给方向，不是精确判据。"""
    out = []
    sm, mem, dram = _get(m, "SOL 计算 %") or 0, _get(m, "SOL 访存 %") or 0, _get(m, "DRAM 吞吐 %") or 0
    # 第一层：SOL 判断大类
    if mem >= 60 and mem >= sm:
        if dram >= 60:
            out.append("带宽瓶颈（DRAM）：融合算子、量化（少读字节）、加大 tile 提高复用")
        else:
            out.append("片上访存瓶颈（L1/shared/L2 忙，DRAM 不忙）：查合并访存和 bank conflict，用寄存器分块减少 smem 读")
    elif sm >= 60:
        tc = _get(m, "Tensor Core 活跃 %")
        if tc is not None and tc < 10:
            out.append("计算瓶颈但没用 Tensor Core：矩阵乘改用 Tensor Core（FP16/BF16/TF32），峰值高 15 倍")
        else:
            out.append("计算瓶颈：接近上限是好事；再快只能降精度（FP8）或减少非矩阵乘指令")
    elif sm < 40 and mem < 40:
        out.append("延迟瓶颈（计算和访存都不忙）：提高 occupancy 或 ILP、加大并行度（grid 是否太小）、减少同步")
    else:
        out.append("两者中等：先看下面的停顿原因")

    # 第二层：合并访存、bank conflict、occupancy
    eff, spr = _get(m, "全局读效率 %"), _get(m, "全局读 sector/请求")
    if eff is not None and eff < 50:
        extra = f"，每请求 {spr:.1f} 个 sector" if spr else ""
        out.append(f"全局读未合并：效率 {eff:.0f}%{extra}，调整线程到元素的映射（相邻线程读相邻地址）")
    conflicts, lds = _get(m, "smem 读 bank conflict"), _get(m, "smem 读指令")
    if conflicts and lds and conflicts / lds > 0.5:
        out.append(f"shared memory bank conflict：平均每条读指令多 {conflicts / lds:.1f} 轮，用 padding 或 swizzle")
    theo, ach = _get(m, "理论 occupancy %"), _get(m, "实际 occupancy %")
    if theo and ach is not None and ach < 0.7 * theo:
        out.append(f"实际 occupancy {ach:.0f}% 远低于理论 {theo:.0f}%：block 数太少（尾波）或负载不均")

    # 第三层：最主要的 warp 停顿原因
    stalls = {r: _get(m, f"stall {r}") for r in STALLS}
    stalls = {r: v for r, v in stalls.items() if v is not None}
    if stalls:
        top = max(stalls, key=stalls.get)
        advice = {
            "long_scoreboard": "预取 / 多级流水（cp.async、TMA）、提高 occupancy 让别的 warp 顶上",
            "barrier": "减少 __syncthreads、双缓冲、warp specialization（生产者/消费者分开）",
            "mio_throttle": "减少 smem 访问次数（寄存器分块、向量化 LDS.128）、消除 bank conflict",
            "short_scoreboard": "smem 延迟没藏住：消除 bank conflict、增加每个 warp 的独立指令",
            "lg_throttle": "全局 load 指令太多：向量化（float4 / 128 位 load）、先搬进 smem 再复用",
            "math_pipe_throttle": "计算管线已满：接近算力上限；换 Tensor Core 或低精度",
            "not_selected": "可发射的 warp 富余：可以用更多寄存器换更少 warp（降 occupancy）",
            "wait": "固定延迟依赖：增加指令级并行（ILP），循环展开",
        }[top]
        out.append(f"主要停顿 {top}（{STALLS[top]}，{stalls[top]:.1f}）：{advice}")
    return out
# --8<-- [end:diagnose]


def short_name(kernel_name: str) -> str:
    """matmul_v4_regtile_kernel(int, int, ...) -> v4_regtile"""
    base = kernel_name.split("(")[0]
    return base.removeprefix("matmul_").removesuffix("_kernel")


def report_table(kernels: list[dict]) -> str:
    """每个 kernel 一列，每个指标一行（指标多、kernel 少，竖着看更方便）。"""
    names = [short_name(k["Kernel Name"]) for k in kernels]
    rows = []
    for short, full in METRICS.items():
        cells = []
        for k in kernels:
            v = k.get(full, "")
            # 耗时：ncu 默认单位 ns（raw 页的单位行会写明），这里换成 us
            if short == "耗时 us" and isinstance(v, float):
                v = v / 1e3
            cells.append(f"{v:,.1f}" if isinstance(v, float) else v)
        rows.append([short, *cells])
    table = tabulate(rows, headers=["指标", *names], tablefmt="github")
    diag = "\n".join(f"- **{n}**：" + "；".join(diagnose(k)) for n, k in zip(names, kernels))
    return f"{table}\n\n## 自动诊断\n\n{diag}\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("versions", nargs="*", default=["v1", "v2", "v3", "v4", "v5"])
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--hw", default="H100", help="只用于输出文件名")
    p.add_argument("--print-only", action="store_true", help="只打印 ncu 命令")
    p.add_argument("--full", action="store_true", help="--set full，生成 .ncu-rep")
    p.add_argument("--parse", type=Path, help="解析一份已有的 CSV（Mac 上可用）")
    p.add_argument("--run", help=argparse.SUPPRESS)
    p.add_argument("--times", type=int, default=3, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.run:
        run_target(args.run, args.size, args.times)
        return
    if args.parse:
        kernels = parse_ncu_csv(args.parse.read_text())
        report = f"# ncu 指标解析：{args.parse.name}\n\n" + report_table(kernels)
        print(report)
        out = EXP / f"{args.parse.stem}_diagnosis.md"
        out.write_text(report)
        print(f"已保存到 {out}")
        return

    cmds = [ncu_command(v, args.size, full=args.full) for v in args.versions]
    if args.print_only or shutil.which("ncu") is None:
        if not args.print_only:
            print("找不到 ncu（Nsight Compute），下面是要运行的命令：")
        for c in cmds:
            print(" ".join(c))
        return
    kernels = []
    for v, c in zip(args.versions, cmds):
        print(f"采集 {v} ...", flush=True)
        res = subprocess.run(c, capture_output=True, text=True)
        if args.full:
            print(res.stdout[-2000:], res.stderr[-2000:])
            continue
        parsed = parse_ncu_csv(res.stdout)
        if not parsed:
            print(f"{v} 没解析到结果，ncu 输出：\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
            continue
        kernels += parsed
    if kernels:
        report = f"# ncu 实测：CUDA 矩阵乘 v1-v5，M=N=K={args.size}（{args.hw}）\n\n" + report_table(kernels)
        print(report)
        (EXP / f"ncu_matmul_{args.hw}.md").write_text(report)


if __name__ == "__main__":
    main()
