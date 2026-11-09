<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [单卡显存构成 →](07-memory.md)

# 第 6 步 / 共 19 步：单卡 MFU

> 阶段 3 / 8：**单卡** · 场景：`single`

1.5B 单卡（不并行），seq 2048，mbs 2，bf16 autocast，SDPA，fused AdamW，随机 token（数据成本为 0）。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20 × 8） |
| - | - | - | - | - |
| `single` | 本步 | 单卡（不并行） | `strategy=none`, `world_size=1` | 3 |

## 先预测

**单卡的 MFU 大约多少？**

- **A.** 不到 30%
- **B.** 30% - 45%
- **C.** 45% - 60%
- **D.** 60% 以上

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只影响随机输入；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子）（已有 3 / 3 个运行）</summary>

| 场景 | 类型 | MFU | 对实测峰值 | 每卡 tokens/s | 优化器 ms | MFU：各种子 | 断言 |
| - | - | - | - | - | - | - | - |
| `single` | 参照 | 61.4% ± 0.2% | 99.8% ± 0.5% | 9,277 ± 37 | 14 ± 0 | s0 61.3% / s1 61.2% / s2 61.6% | 通过 |

![MFU（均值，误差线为种子间最小到最大）](figures/h20x8-p_single-metrics.svg)

![峰值显存构成（实测峰值，分项为估算）](figures/h20x8-p_single-memory.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `single` [s0](../../../results/h20x8/L08/single/s0-2a864f/) · [s1](../../../results/h20x8/L08/single/s1-b8026d/) · [s2](../../../results/h20x8/L08/single/s2-81f014/)

</details>

---

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [单卡显存构成 →](07-memory.md)
