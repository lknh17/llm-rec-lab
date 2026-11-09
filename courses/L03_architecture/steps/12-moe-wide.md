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

- **A.** 差不到 0.02：几乎追平
- **B.** 差 0.02 - 0.06
- **C.** 差 0.06 以上

> 先选定一个（记下来），再展开下面的实测。bpb 差异小于 0.02 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 小 GPT（6 层 d=384）从零训练 100 MB 中文维基 · 每个场景只换一处结构 · 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

按"每 token FLOPs"横轴摆这三个点，MoE 在不在两个 Dense 连线的下方？

---

[← MoE：同样的激活算力](11-moe.md) · [课程首页](../README.md) · [去掉负载均衡损失 →](13-fault.md)
