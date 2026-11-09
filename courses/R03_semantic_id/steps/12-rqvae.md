<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 1000 万物品](11-scale.md) · [课程首页](../README.md) · [同事简化了量化代码 →](13-fault.md)

# 第 12 步 / 共 15 步：RQ-VAE

> 阶段 5 / 7：**RQ-VAE** · 场景：`rqvae`（对照 `synth`）

TIGER 原文用 RQ-VAE：编码器把物品向量压到 64 维，3 层 × 256 的残差码本和编解码器一起训练（straight-through + commitment loss）。码本随机初始化、**不**重置死码。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `synth` | 对照 | 合成 100 万物品，4 层 × 256 码（前 d 层即 d 层的结果） | `items=synth`, `sid_levels=4`, `sid_k=256` | 3 |
| `rqvae` | 本步 | 合成 100 万物品，RQ-VAE 3 层 × 256 码（端到端训练，不重置死码） | `items=synth`, `quantizer=rqvae`, `sid_k=256` | 3 |

## 先预测

**RQ-VAE 第 1 层实际用到的码字占多少？**

- **A.** 不到 50%
- **B.** 50% - 90%
- **C.** 90% 以上

> 先选定一个（记下来），再展开下面的实测。量化只受种子影响（k-means 初始化），比例类指标的种子间波动通常在 1 个百分点以内；差值小于种子间波动按“差不多”算。

## 实测

<details><summary>展开实测：H20 · ML-1M 3706 部电影 + 合成 100 万 / 1000 万物品 · RQ-KMeans / RQ-VAE · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 1000 万物品](11-scale.md) · [课程首页](../README.md) · [同事简化了量化代码 →](13-fault.md)
