<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← attention 实现](14-kernels.md) · [课程首页](../README.md) · [7B FSDP 8 卡 →](16-7b-fsdp.md)

# 第 15 步 / 共 19 步：7B 单卡全参

> 阶段 6 / 8：**回到 7B** · 场景：`qwen7b_single`

7B（76 亿参数）单卡全参训练，mbs 1。fp32 主参数 30.5 GB；AdamW 的 m、v 在第一次 optimizer.step() 时才分配。H20 有 96 GB。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20 × 8） |
| - | - | - | - | - |
| `qwen7b_fsdp8` | 参照 | 7B FSDP2 FULL_SHARD 8 卡（mbs 2） | `preset=qwen7b`, `strategy=fsdp_full` | 1 |
| `qwen7b_single` | 本步 | 7B 单卡全参训练（mbs 1） | `preset=qwen7b`, `strategy=none`, `world_size=1`, `mbs=1` | 1 |

## 先预测

**在哪一步 OOM？**

- **A.** 不会 OOM
- **B.** 建模或前向
- **C.** 反向
- **D.** 第一次 optimizer.step()

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只影响随机输入；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子）（已有 2 / 2 个运行）</summary>

| 场景 | 类型 | OOM 阶段序号 | OOM 时已分配 GB | 估算需要 GB | OOM 阶段序号：各种子 | 断言 |
| - | - | - | - | - | - | - |
| `qwen7b_fsdp8` | 候选改动 | 4 | — | — | s0 4 | 通过 |
| `qwen7b_single` | 候选改动 | 3 | 97.8 | 130 | s0 3 | 通过 |

![峰值显存构成（实测峰值，分项为估算）](figures/h20x8-p_7b_oom-memory.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `qwen7b_fsdp8` [s0](../../../results/h20x8/L08/qwen7b_fsdp8/s0-89b424/)
- `qwen7b_single` [s0](../../../results/h20x8/L08/qwen7b_single/s0-4c327e/)

</details>

---

[← attention 实现](14-kernels.md) · [课程首页](../README.md) · [7B FSDP 8 卡 →](16-7b-fsdp.md)
