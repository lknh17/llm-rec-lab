"""ncu CSV 解析和自动诊断的测试：用手写的小 CSV 样例，Mac 上就能跑。"""

from stages.s03_kernels.cuda.ncu_profile import METRICS, diagnose, ncu_command, parse_ncu_csv, short_name

T = METRICS["耗时 us"]
SM = METRICS["SOL 计算 %"]
MEM = METRICS["SOL 访存 %"]
DRAM = METRICS["DRAM 吞吐 %"]

# --page raw 格式：表头、单位行、每个 kernel 一行；前后混着 ncu 日志和程序输出
RAW = f"""==PROF== Connected to process 12345
some program output
"ID","Process ID","Kernel Name","{T}","{SM}","{MEM}","{DRAM}"
"","","","nsecond","%","%","%"
"0","12345","matmul_v1_naive_kernel(int, int, int, const float *, const float *, float *)","250,000,000","22.1","94.3","6.0"
"1","12345","matmul_v4_regtile_kernel(int, int, int, const float *, const float *, float *)","3,600,000","78","52","n/a"
==PROF== Disconnected from process 12345
"""

# --page details 格式：每行一个指标
DETAILS = f"""==PROF== Connected to process 1
"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"
"0","matmul_v5_wmma_kernel","Speed Of Light","{SM}","%","35.2"
"0","matmul_v5_wmma_kernel","Speed Of Light","{MEM}","%","78.0"
"0","matmul_v5_wmma_kernel","Memory","{DRAM}","%","28.5"
"""


def test_parse_raw_page_skips_units_and_logs():
    ks = parse_ncu_csv(RAW)
    assert len(ks) == 2
    assert ks[0][T] == 250_000_000.0  # 千分位逗号被去掉
    assert ks[0][SM] == 22.1
    assert ks[1][DRAM] == "n/a"  # 非数值原样保留
    assert short_name(ks[1]["Kernel Name"]) == "v4_regtile"


def test_parse_details_page_pivots_to_wide():
    ks = parse_ncu_csv(DETAILS)
    assert len(ks) == 1
    assert ks[0][MEM] == 78.0 and ks[0][DRAM] == 28.5


def test_parse_empty_output():
    assert parse_ncu_csv("==PROF== ERROR: no kernels profiled\n") == []


def test_diagnose_categories():
    # DRAM 和访存 SOL 都高 -> 带宽瓶颈
    assert diagnose({MEM: 90.0, SM: 20.0, DRAM: 85.0})[0].startswith("带宽瓶颈")
    # 访存 SOL 高但 DRAM 低 -> 片上访存
    assert diagnose({MEM: 90.0, SM: 20.0, DRAM: 8.0})[0].startswith("片上访存")
    # 两者都低 -> 延迟瓶颈
    assert diagnose({MEM: 20.0, SM: 15.0, DRAM: 5.0})[0].startswith("延迟瓶颈")
    # 计算 SOL 高、Tensor Core 不动 -> 建议上 Tensor Core
    assert "Tensor Core" in diagnose({MEM: 40.0, SM: 80.0, METRICS["Tensor Core 活跃 %"]: 0.0})[0]


def test_diagnose_flags_coalescing_bank_conflict_and_stall():
    m = {MEM: 90.0, SM: 30.0, DRAM: 10.0,
         METRICS["全局读效率 %"]: 25.0, METRICS["全局读 sector/请求"]: 16.5,
         METRICS["smem 读 bank conflict"]: 6e8, METRICS["smem 读指令"]: 5e8,
         METRICS["stall long_scoreboard"]: 2.0, METRICS["stall barrier"]: 7.0}
    text = " ".join(diagnose(m))
    assert "未合并" in text and "bank conflict" in text and "主要停顿 barrier" in text


def test_ncu_command_filters_kernel_and_launch():
    cmd = ncu_command("v3", size=2048)
    assert cmd[0] == "ncu"
    assert cmd[cmd.index("-k") + 1] == "regex:matmul_v3_"
    assert cmd[cmd.index("--launch-skip") + 1] == "2" and cmd[cmd.index("--launch-count") + 1] == "1"
    assert "--csv" in cmd and "--set" not in cmd
    assert cmd[-6:] == ["--run", "v3", "--size", "2048", "--times", "3"]
    full = ncu_command("v4", full=True)
    assert full[full.index("--set") + 1] == "full" and "--csv" not in full
