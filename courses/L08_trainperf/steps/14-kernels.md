<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 全 fp32](13-precision.md) · [课程首页](../README.md) · [7B 单卡全参 →](15-7b-oom.md)

# 第 14 步 / 共 19 步：attention 实现

> 阶段 5 / 8：**算得划不划算** · 场景：`naive_attn` `compile` `adam_forloop`（对照 `baseline`）

attention 从 SDPA（flash）换成手写：显式算 [mbs, 12, 2048, 2048] 的打分和 softmax，每层存下来给反向用。另有 torch.compile 和"AdamW 逐参数循环"两个对照。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20 × 8） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：1.5B，DDP 8 卡，seq 2048，mbs 2，bf16，SDPA，fused AdamW | — | 3 |
| `naive_attn` | 本步 | 手写 attention（显式 [B, H, T, T]） | `attn=naive` | 3 |
| `compile` | 本步 | torch.compile 每个层 | `compile=True` | 1 |
| `adam_forloop` | 本步 | AdamW 逐参数循环实现（不 fused、不 foreach） | `adam=forloop` | 1 |

## 先预测

**手写 attention 的峰值显存比 SDPA 多多少 GB？**

- **A.** 不到 2 GB
- **B.** 2 - 10 GB
- **C.** 10 GB 以上

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只影响随机输入；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子）（已有 0 / 8 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 全 fp32](13-precision.md) · [课程首页](../README.md) · [7B 单卡全参 →](15-7b-oom.md)
