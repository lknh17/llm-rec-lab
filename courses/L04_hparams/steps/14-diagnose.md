<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 换成 fp16](13-fault.md) · [课程首页](../README.md) · [fp16 + GradScaler →](15-fix.md)

# 第 14 步 / 共 16 步：诊断：fp16 为什么静默变差

> 阶段 6 / 7：**精度** · 场景：`baseline`（对照 `fault_fp16`）

只把 bf16 换成了 fp16，没有 NaN、没有报错，效果明显变差。按需逐级点开线索，然后点选根因。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `fault_fp16` | 对照 | 换成 fp16 autocast，没有 loss scaling | `precision=fp16` | 3 |
| `baseline` | 本步 | 基线：lr 2e-3、warmup 2%、batch 64 × 512、bf16（= L01 基线） | — | 3 |

## 线索

逐条展开，看到能下判断为止。

<details><summary>线索 1</summary>

看 loss 曲线和梯度范数：梯度范数中位数 bf16 〔待 H20 生成〕，fp16 〔待 H20 生成〕。没有出现 NaN。

</details>

<details><summary>线索 2</summary>

统计 logits 梯度里恰好为 0 的比例：bf16 〔待 H20 生成〕，fp16 〔待 H20 生成〕。交叉熵的梯度要除以 32768 个 token。

</details>

<details><summary>线索 3</summary>

加上 GradScaler 之后：bpb 〔待 H20 生成〕（bf16 〔待 H20 生成〕），零梯度比例 〔待 H20 生成〕，最终 scale 〔待 H20 生成〕。

</details>

## 先预测

**fp16 训练静默变差的根因是什么？**

- **A.** fp16 的尾数只有 10 位，前向的 loss 算不准
- **B.** 没有 loss scaling：交叉熵对 logits 的小梯度在 fp16 里下溢成 0，大部分词收不到梯度
- **C.** 梯度上溢成 inf，这些步被跳过了
- **D.** fp16 要配更小的学习率

> 先选定一个（记下来），再展开下面的实测。bpb 差异小于 0.02 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 小 GPT（1060 万主体参数）从零训练 100 MB 中文维基 · 每个场景只改一个训练参数 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 换成 fp16](13-fault.md) · [课程首页](../README.md) · [fp16 + GradScaler →](15-fix.md)
