<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 诊断：fp16 为什么静默变差](14-diagnose.md) · [课程首页](../README.md) · [回顾 →](16-keycard.md)

# 第 15 步 / 共 16 步：fp16 + GradScaler

> 阶段 6 / 7：**精度** · 场景：`fp16_scaler`（对照 `baseline`）

修复：fp16 加上动态 loss scaling（初始 scale 65536，连续 2000 步没有 inf 就翻倍，遇到 inf 就减半并跳过这一步）。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：lr 2e-3、warmup 2%、batch 64 × 512、bf16（= L01 基线） | — | 3 |
| `fault_fp16` | 参照 | 换成 fp16 autocast，没有 loss scaling | `precision=fp16` | 3 |
| `fp16_scaler` | 本步 | fp16 + GradScaler（动态 loss scaling） | `precision=fp16`, `grad_scaler=True` | 3 |

## 先预测

**加上 GradScaler 后，fp16 和 bf16 基线比怎样？**

- **A.** 比 bf16 更好（低 0.02 以上）
- **B.** 和 bf16 差不多（±0.02）
- **C.** 仍然更差（高 0.02 以上）

> 先选定一个（记下来），再展开下面的实测。bpb 差异小于 0.02 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 小 GPT（1060 万主体参数）从零训练 100 MB 中文维基 · 每个场景只改一个训练参数 · 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

三条 loss 曲线里，fp16 + GradScaler 和 bf16 重合吗？

---

[← 诊断：fp16 为什么静默变差](14-diagnose.md) · [课程首页](../README.md) · [回顾 →](16-keycard.md)
