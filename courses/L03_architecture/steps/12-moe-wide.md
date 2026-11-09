<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← MoE：同样的激活算力](11-moe.md) · [课程首页](../README.md) · [去掉负载均衡损失 →](13-fault.md)

# 第 12 步 / 共 15 步：MoE 对同总参数的宽 Dense

> 阶段 6 / 7：**MoE** · 场景：`moe`（对照 `dense_wide`）

另一个对照：宽 Dense，中间维 4096，FFN 参数和 MoE 的 8 个专家加起来一样多，但每个 token 都要过全部参数，每 token 算力约 2.2 倍。它是 MoE 的"上限"：同样的参数全都用上。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `dense_wide` | 对照 | 宽 Dense：中间维 4096（FFN 总参数和 MoE 相同，每 token 算力约 2.2 倍） | `d_ff=4096` | 3 |
| `baseline` | 参照 | 基线：MHA 6 头、RoPE、Pre-Norm、Dense SwiGLU（= L01 基线） | — | 3 |
| `moe` | 本步 | MoE：8 个专家（中间维 512）top-2，激活参数和 dense 相同，负载均衡系数 0.01 | `ffn=moe` | 3 |

## 先预测

**MoE 比同总参数的宽 Dense 差多少？**

- **A.** 差不到 0.05：几乎追平
- **B.** 差 0.05 - 0.15
- **C.** 差 0.15 以上

> 先选定一个（记下来），再展开下面的实测。bpb 差异小于 0.05（H20 上 3 种子均值之差的噪声量级）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 小 GPT（6 层 d=384）从零训练 100 MB 中文维基 · 每个场景只换一处结构 · 3 种子（已有 9 / 9 个运行）</summary>

| 场景 | 类型 | 留出集 bpb | 总参数 | 每 token 训练 FLOPs | 训练秒数 | 留出集 bpb：与 `dense_wide` 差 | 留出集 bpb：各种子 | 断言 |
| - | - | - | - | - | - | - | - | - |
| `dense_wide` | 候选改动 | 1.739 ± 0.088 | 38,146,944.0 ± 0.0 | 235,959,552.0 ± 0.0 | 131 ± 1 | — | s0 1.643 / s1 1.815 / s2 1.758 | 通过 |
| `baseline` | 基线 | 1.727 ± 0.048 | 16,913,280.0 ± 0.0 | 108,557,568.0 ± 0.0 | 79 ± 0 | -0.011 | s0 1.725 / s1 1.776 / s2 1.680 |  |
| `moe` | 候选改动 | 1.697 ± 0.034 | 38,165,376.0 ± 0.0 | 108,668,160.0 ± 0.0 | 128 ± 1 | -0.042 | s0 1.730 / s1 1.662 / s2 1.699 | 通过 |

![留出集 bpb（均值，误差线为种子间最小到最大）](figures/h20x8-p_moe_wide-metrics.svg)

![可训练参数量](figures/h20x8-p_moe_wide-params.svg)

![验证集 bits-per-byte（均值，阴影为种子间范围）](figures/h20x8-p_moe_wide-loss.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `dense_wide` [s0](../../../results/h20x8/L03/dense_wide/s0-7df068/) · [s1](../../../results/h20x8/L03/dense_wide/s1-429f0a/) · [s2](../../../results/h20x8/L03/dense_wide/s2-bbd85c/)
- `baseline` [s0](../../../results/h20x8/L03/baseline/s0-39279f/) · [s1](../../../results/h20x8/L03/baseline/s1-cbec53/) · [s2](../../../results/h20x8/L03/baseline/s2-618736/)
- `moe` [s0](../../../results/h20x8/L03/moe/s0-d26637/) · [s1](../../../results/h20x8/L03/moe/s1-c21a54/) · [s2](../../../results/h20x8/L03/moe/s2-728fe8/)

</details>

## 看完实测再想一步

三条验证曲线的种子带有多宽？和曲线之间的距离比，哪个大？

---

[← MoE：同样的激活算力](11-moe.md) · [课程首页](../README.md) · [去掉负载均衡损失 →](13-fault.md)
