<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 激活重计算](11-ckpt.md) · [课程首页](../README.md) · [全 fp32 →](13-precision.md)

# 第 12 步 / 共 19 步：mbs 2 → 1

> 阶段 5 / 8：**算得划不划算** · 场景：`mbs1`（对照 `baseline`）

每卡 micro batch 从 2 条降到 1 条序列。矩阵乘的 M 维 = mbs × T：从 4096 降到 2048。另有 mbs 4 的对照。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20 × 8） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：1.5B，DDP 8 卡，seq 2048，mbs 2，bf16，SDPA，fused AdamW | — | 3 |
| `mbs4` | 参照 | mbs 4 | `mbs=4` | 3 |
| `mbs1` | 本步 | mbs 1 | `mbs=1` | 3 |

## 先预测

**mbs 1 的 MFU 是 mbs 2 的多少？**

- **A.** 不到 80%
- **B.** 80% - 95%
- **C.** 95% 以上

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只影响随机输入；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子）（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 激活重计算](11-ckpt.md) · [课程首页](../README.md) · [全 fp32 →](13-precision.md)
