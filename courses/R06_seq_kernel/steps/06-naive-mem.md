<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [SDPA →](07-sdpa.md)

# 第 6 步 / 共 15 步：naive 的显存

> 阶段 3 / 6：**四种实现** · 场景：`baseline`

naive 实现，逐点打分，L 从 512 到 8192（16 倍）。看增量峰值显存。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 本步 | 基线：naive（PyTorch 逐步，物化 L × L） | — | 3 |

## 先预测

**L = 8192 时 naive 逐点 attention 的增量峰值显存大约多少？**

- **A.** 不到 1 GB
- **B.** 1 - 10 GB
- **C.** 10 GB 以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · B=4、H=8、L 512-8192、D 64/128、bf16 · 3 种子（已有 0 / 3 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [SDPA →](07-sdpa.md)
