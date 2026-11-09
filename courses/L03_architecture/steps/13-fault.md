<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← MoE 对同总参数的宽 Dense](12-moe-wide.md) · [课程首页](../README.md) · [诊断：去掉负载均衡的代价在哪 →](14-diagnose.md)

# 第 13 步 / 共 15 步：去掉负载均衡损失

> 阶段 6 / 7：**MoE** · 场景：`fault_moe_noaux`（对照 `moe`）

同事觉得负载均衡损失"和语言建模无关、会拖累质量"，把系数设成 0 重训了一遍。其余和 MoE 场景完全相同。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `moe` | 对照 | MoE：8 个专家（中间维 512）top-2，激活参数和 dense 相同，负载均衡系数 0.01 | `ffn=moe` | 3 |
| `baseline` | 参照 | 基线：MHA 6 头、RoPE、Pre-Norm、Dense SwiGLU（= L01 基线） | — | 3 |
| `fault_moe_noaux` | 本步 | MoE 去掉负载均衡损失（moe_aux_coef = 0） | `ffn=moe`, `moe_aux_coef=0.0` | 3 |

## 先预测

**去掉负载均衡后，最忙的专家分到的 token 是平均的几倍？（top-2 / 8 专家时最多 4 倍）**

- **A.** 不到 1.5 倍：路由自己会均衡
- **B.** 1.5 - 2.5 倍
- **C.** 2.5 倍以上

> 先选定一个（记下来），再展开下面的实测。bpb 差异小于 0.02 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 小 GPT（6 层 d=384）从零训练 100 MB 中文维基 · 每个场景只换一处结构 · 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

moe_max_load 曲线上，失衡是在训练早期就出现了，还是慢慢变严重？

---

[← MoE 对同总参数的宽 Dense](12-moe-wide.md) · [课程首页](../README.md) · [诊断：去掉负载均衡的代价在哪 →](14-diagnose.md)
