<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 不针对错例：泛泛加难样本](08-generic.md) · [课程首页](../README.md) · [离线好好的，上线全错 →](10-fault.md)

# 第 9 步 / 共 13 步：多训两轮：loss 和指标同步吗

> 阶段 5 / 8：**loss 与指标** · 场景：`epochs_4`（对照 `baseline`）

训练 4 轮（基线 2 轮），每轮结束都在 dev 上算一次 loss，并做一次生成评估。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：和 L05 的 SFT 基线同配置（LoRA r=16 挂 7 类线性层，assistant-only，2 轮），5 个训练种子量噪声 | — | 5 |
| `epochs_4` | 本步 | 训练 4 轮（基线 2 轮），每轮都评估：dev loss 和 dev 全对率是否同步 | `epochs=4` | 3 |

## 先预测

**第 2 轮到第 4 轮，dev loss 和 dev 全对率分别怎么走？**

- **A.** dev loss 继续降，全对率继续升（同步）
- **B.** dev loss 上升，全对率基本不变或略降（不同步）
- **C.** 两者都明显变差

> 先选定一个（记下来），再展开下面的实测。差异小于 6.7 个点（dev 两条样本）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · Qwen2.5-7B · 基线和主改动 5 种子，其余 3 种子（已有 0 / 8 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

loss 曲线最后两轮 train loss 几乎为 0，dev loss 往哪走？逐轮全对率在运行目录的 metrics.jsonl（dev_exact_all_ok）里。

---

[← 不针对错例：泛泛加难样本](08-generic.md) · [课程首页](../README.md) · [离线好好的，上线全错 →](10-fault.md)
