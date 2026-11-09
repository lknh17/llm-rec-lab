<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← attention 实现](14-kernels.md) · [课程首页](../README.md) · [7B FSDP 8 卡 →](16-7b-fsdp.md)

# 第 15 步 / 共 19 步：7B 单卡全参

> 阶段 6 / 8：**回到 7B** · 场景：`qwen7b_single`

7B（76 亿参数）单卡全参训练，mbs 1。fp32 主参数 30.5 GB；AdamW 的 m、v 在第一次 optimizer.step() 时才分配。H20 有 96 GB。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20 × 8） |
| - | - | - | - | - |
| `qwen7b_fsdp8` | 参照 | 7B FSDP2 FULL_SHARD 8 卡（mbs 2） | `preset=qwen7b`, `strategy=fsdp_full` | 1 |
| `qwen7b_single` | 本步 | 7B 单卡全参训练（mbs 1） | `preset=qwen7b`, `strategy=none`, `world_size=1`, `mbs=1` | 1 |

## 先预测

**在哪一步 OOM？**

- **A.** 不会 OOM
- **B.** 建模或前向
- **C.** 反向
- **D.** 第一次 optimizer.step()

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只影响随机输入；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子）（已有 0 / 2 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← attention 实现](14-kernels.md) · [课程首页](../README.md) · [7B FSDP 8 卡 →](16-7b-fsdp.md)
