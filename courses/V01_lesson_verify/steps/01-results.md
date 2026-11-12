<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[课程首页](../README.md)

# 第 1 步 / 共 1 步：验证结果

> 阶段 1 / 1：**验证**

每一项对应讲义 GPU 验证清单的一条。运行目录的 trace.json 里有讲义脚本生成的报告原文（outputs），log.txt 里有完整输出；
结果回填到私有仓库的讲义折叠块和 experiments/ 里。

<details><summary>H20 · 讲义代码原样运行（--hw H20）· 每项 1 张卡、1 次</summary>

| 场景 | 类型 | 在哪一步 | 用时（秒） | 与 `baseline` 差 | 种子数 |
| - | - | - | - | - | - |
| `env_probe` | 基线 |  | 0 | — | 1 |
| `ncu_parse` | 参照 |  | 1 | — | 1 |
| `gpu_check` | 候选改动 |  | 12.3 | — | 1 |
| `decode_bench` | 候选改动 |  | 24.4 | — | 1 |
| `s3_tests` | 候选改动 |  | 122.3 | — | 1 |
| `triton_bench` | 候选改动 |  | 20.6 | — | 1 |
| `coalescing` | 候选改动 |  | 14.9 | — | 1 |
| `cuda_matmul` | 候选改动 |  | 67.9 | — | 1 |
| `ncu` | 候选改动 |  | 1 | — | 1 |
| `split_kv` | 候选改动 |  | 16.3 | — | 1 |
| `w4a16` | 候选改动 |  | 36.6 | — | 1 |
| `moe` | 候选改动 |  | 18.4 | — | 1 |
| `vllm_facts` | 候选改动 |  | 145 | — | 1 |

</details>

全部图和结果表也汇总在 [课程首页](../README.md) 的“场景与结果”“证据图”两节。

---

[课程首页](../README.md)
