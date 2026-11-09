<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 7B 单卡全参](15-7b-oom.md) · [课程首页](../README.md) · [换成在线分词 →](17-fault.md)

# 第 16 步 / 共 19 步：7B FSDP 8 卡

> 阶段 6 / 8：**回到 7B** · 场景：`qwen7b_fsdp8`（对照 `baseline`）

7B 用 FSDP2 FULL_SHARD 8 卡，mbs 2，seq 2048。每层前向、反向前都要 all-gather 这一层的整份参数（bf16，约 0.45 GB），通信量是 DDP 的约 1.5 倍。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20 × 8） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：1.5B，DDP 8 卡，seq 2048，mbs 2，bf16，SDPA，fused AdamW | — | 3 |
| `qwen7b_fsdp8` | 本步 | 7B FSDP2 FULL_SHARD 8 卡（mbs 2） | `preset=qwen7b`, `strategy=fsdp_full` | 1 |

## 先预测

**7B FSDP 8 卡的 MFU 大约多少？**

- **A.** 不到 20%
- **B.** 20% - 35%
- **C.** 35% - 50%
- **D.** 50% 以上

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只影响随机输入；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子）（已有 0 / 4 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 7B 单卡全参](15-7b-oom.md) · [课程首页](../README.md) · [换成在线分词 →](17-fault.md)
