<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← DDP 8 卡](08-ddp.md) · [课程首页](../README.md) · [梯度累积不用 no_sync →](10-gas.md)

# 第 9 步 / 共 19 步：FSDP FULL_SHARD

> 阶段 4 / 8：**8 卡并行** · 场景：`fsdp_full`（对照 `baseline`）

从 DDP 换成 FSDP2 FULL_SHARD：参数、梯度、优化器状态都切成 8 份，每层用之前 all-gather、用完释放。另有一个"前向后不释放"（≈ ZeRO-2）的对照。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20 × 8） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：1.5B，DDP 8 卡，seq 2048，mbs 2，bf16，SDPA，fused AdamW | — | 3 |
| `fsdp_grad_op` | 参照 | FSDP2 前向后不释放整份参数（≈ ZeRO-2） | `strategy=fsdp_grad_op` | 3 |
| `fsdp_full` | 本步 | FSDP2 FULL_SHARD（≈ ZeRO-3） | `strategy=fsdp_full` | 3 |

## 先预测

**FULL_SHARD 的每卡峰值显存比 DDP 少多少 GB？**

- **A.** 少 30 GB 以上
- **B.** 少 15 - 30 GB
- **C.** 少 3 - 15 GB
- **D.** 少不到 3 GB

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只影响随机输入；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子）（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← DDP 8 卡](08-ddp.md) · [课程首页](../README.md) · [梯度累积不用 no_sync →](10-gas.md)
