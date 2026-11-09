<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 单卡 MFU](06-single.md) · [课程首页](../README.md) · [DDP 8 卡 →](08-ddp.md)

# 第 7 步 / 共 19 步：单卡显存构成

> 阶段 3 / 8：**单卡** · 场景：`single`

同一个单卡运行：峰值显存里，参数、梯度、优化器状态之外，"激活与其他"占多少？（激活随 mbs × T 线性增长；还包括 [2, 2048, 151936] 的 logits。）

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20 × 8） |
| - | - | - | - | - |
| `single` | 本步 | 单卡（不并行） | `strategy=none`, `world_size=1` | 3 |

## 先预测

**激活与其他占峰值显存的多少？**

- **A.** 不到 25%
- **B.** 25% - 50%
- **C.** 50% - 75%
- **D.** 75% 以上

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只影响随机输入；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子）（已有 3 / 3 个运行）</summary>

| 场景 | 类型 | 峰值 GB | 激活与其他 GB | 静态估算 GB | 峰值 GB：各种子 | 断言 |
| - | - | - | - | - | - | - |
| `single` | 参照 | 48.9 ± 0.0 | 24.1 ± 0.0 | 24.7 ± 0.0 | s0 48.9 / s1 48.9 / s2 48.9 | 通过 |

![峰值显存构成（实测峰值，分项为估算）](figures/h20x8-p_memory-memory.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `single` [s0](../../../results/h20x8/L08/single/s0-2a864f/) · [s1](../../../results/h20x8/L08/single/s1-b8026d/) · [s2](../../../results/h20x8/L08/single/s2-81f014/)

</details>

---

[← 单卡 MFU](06-single.md) · [课程首页](../README.md) · [DDP 8 卡 →](08-ddp.md)
