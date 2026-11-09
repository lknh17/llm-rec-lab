<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← torch.compile](08-compile.md) · [课程首页](../README.md) · [块大小 →](10-block.md)

# 第 9 步 / 共 15 步：手写 Triton

> 阶段 3 / 6：**四种实现** · 场景：`triton`（对照 `baseline`）

手写的 Triton 融合 kernel（块 64 × 64、4 warp、2 级流水），逐点和 softmax 两种打分。每个结果都和 fp32 参照比相对误差。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：naive（PyTorch 逐步，物化 L × L） | — | 3 |
| `compiled` | 参照 | torch.compile(naive) | `attn_impl=compiled` | 3 |
| `triton` | 本步 | 手写 Triton 融合 kernel（块 64 × 64） | `attn_impl=triton` | 3 |

## 先预测

**L = 8192 逐点打分，Triton 比 naive 快几倍？**

- **A.** 不到 3 倍
- **B.** 3 - 10 倍
- **C.** 10 倍以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · B=4、H=8、L 512-8192、D 64/128、bf16 · 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← torch.compile](08-compile.md) · [课程首页](../README.md) · [块大小 →](10-block.md)
