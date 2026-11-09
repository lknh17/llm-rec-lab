<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 12 层 Post-Norm](10-post.md) · [课程首页](../README.md) · [MoE 对同总参数的宽 Dense →](12-moe-wide.md)

# 第 11 步 / 共 15 步：MoE：同样的激活算力

> 阶段 6 / 7：**MoE** · 场景：`moe`（对照 `baseline`）

FFN 换成 8 个专家（每个中间维 512）、每 token 选 top-2，负载均衡系数 0.01。每 token 的算力和 Dense 相同，总参数约 2.3 倍。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：MHA 6 头、RoPE、Pre-Norm、Dense SwiGLU（= L01 基线） | — | 3 |
| `moe` | 本步 | MoE：8 个专家（中间维 512）top-2，激活参数和 dense 相同，负载均衡系数 0.01 | `ffn=moe` | 3 |

## 先预测

**和 Dense 基线比，MoE 的留出集 bpb 怎样？**

- **A.** 更好（低 0.02 以上）
- **B.** 差不多（±0.02）
- **C.** 更差（高 0.02 以上）

> 先选定一个（记下来），再展开下面的实测。bpb 差异小于 0.02 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 小 GPT（6 层 d=384）从零训练 100 MB 中文维基 · 每个场景只换一处结构 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

训练早期 MoE 比 Dense 慢还是快？最后呢？

---

[← 12 层 Post-Norm](10-post.md) · [课程首页](../README.md) · [MoE 对同总参数的宽 Dense →](12-moe-wide.md)
