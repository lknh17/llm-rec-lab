<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← packing 加文档 mask](12-docmask.md) · [课程首页](../README.md) · [诊断：fp16 为什么静默变差 →](14-diagnose.md)

# 第 13 步 / 共 16 步：换成 fp16

> 阶段 6 / 7：**精度** · 场景：`fault_fp16`（对照 `baseline`）

为了兼容一批只支持 fp16 的老卡，同事把 autocast 从 bf16 换成了 fp16，其余不变。训练跑完了，没有 NaN，没有报错。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：lr 2e-3、warmup 2%、batch 64 × 512、bf16（= L01 基线） | — | 3 |
| `fault_fp16` | 本步 | 换成 fp16 autocast，没有 loss scaling | `precision=fp16` | 3 |

## 先预测

**和 bf16 基线比，留出集 bpb 怎样？**

- **A.** 差不多（差 0.02 以内）
- **B.** 差 0.02 - 0.1
- **C.** 差 0.1 以上

> 先选定一个（记下来），再展开下面的实测。bpb 差异小于 0.02 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 小 GPT（1060 万主体参数）从零训练 100 MB 中文维基 · 每个场景只改一个训练参数 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

fp16 的 loss 曲线是一开始就慢，还是中途才变慢？

---

[← packing 加文档 mask](12-docmask.md) · [课程首页](../README.md) · [诊断：fp16 为什么静默变差 →](14-diagnose.md)
