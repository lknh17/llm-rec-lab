<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 块大小](10-block.md) · [课程首页](../README.md) · [attention 在整层里占多少 →](12-layer.md)

# 第 11 步 / 共 15 步：离 roofline 多远

> 阶段 4 / 6：**块大小与 roofline** · 场景：`triton`

Triton 逐点 kernel 在 L = 8192、D = 64 时，算术强度约 2048 FLOP/B，远在拐点右边：上限是 148 TFLOPS。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `sdpa` | 参照 | PyTorch SDPA（flash，只有 softmax） | `attn_impl=sdpa`, `attn_kinds=softmax` | 3 |
| `triton` | 本步 | 手写 Triton 融合 kernel（块 64 × 64） | `attn_impl=triton` | 3 |

## 先预测

**它达到了 roofline 上限的多少？**

- **A.** 不到 20%
- **B.** 20% - 50%
- **C.** 50% 以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · B=4、H=8、L 512-8192、D 64/128、bf16 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 块大小](10-block.md) · [课程首页](../README.md) · [attention 在整层里占多少 →](12-layer.md)
