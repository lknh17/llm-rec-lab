<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← embedding 维度 32 → 128](09-emb.md) · [课程首页](../README.md) · [8 个 epoch + 128 维 →](11-epochs-emb.md)

# 第 10 步 / 共 13 步：训 8 个 epoch

> 阶段 4 / 5：**容量与 epoch** · 场景：`ep8`（对照 `baseline`）

同一个模型训 8 遍，每遍结束在验证和留出上各评估一次。曲线是"验证 AUC 相对第 1 个 epoch 的变化（百分点）"。工业 CTR 模型通常只训 1 个 epoch。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：全局时间切分（训练 < t80 ≤ 验证 < t90 ≤ 留出），只用 ID，1 个 epoch | — | 3 |
| `ep8` | 本步 | 训 8 个 epoch（每个 epoch 末在验证 / 留出上各评估一次） | `epochs=8` | 3 |

## 先预测

**验证 AUC 最高出现在第几个 epoch？（3 种子平均）**

- **A.** 第 1 个
- **B.** 第 2 - 3 个
- **C.** 第 4 个以后

> 先选定一个（记下来），再展开下面的实测。AUC 差异小于 0.3 个百分点或小于种子间波动，都按“差不多”算。不同切分的验证 AUC 不能直接比，要看固定留出集。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M 全量 · 用户 ID + 电影 ID 的 Embedding + MLP · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

切到 valid_logloss 曲线：它的最低点在第几个 epoch？

---

[← embedding 维度 32 → 128](09-emb.md) · [课程首页](../README.md) · [8 个 epoch + 128 维 →](11-epochs-emb.md)
