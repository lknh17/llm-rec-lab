<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← naive 的显存](06-naive-mem.md) · [课程首页](../README.md) · [torch.compile →](08-compile.md)

# 第 7 步 / 共 15 步：SDPA

> 阶段 3 / 6：**四种实现** · 场景：`sdpa`（对照 `baseline`）

softmax 打分换成 PyTorch SDPA（is_causal=True，走 flash 后端）。SDPA 不支持逐点打分，这个场景只测 softmax。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：naive（PyTorch 逐步，物化 L × L） | — | 3 |
| `sdpa` | 本步 | PyTorch SDPA（flash，只有 softmax） | `attn_impl=sdpa`, `attn_kinds=softmax` | 3 |

## 先预测

**L = 8192 时 SDPA 比 naive softmax 快几倍？**

- **A.** 不到 3 倍
- **B.** 3 - 10 倍
- **C.** 10 倍以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · B=4、H=8、L 512-8192、D 64/128、bf16 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← naive 的显存](06-naive-mem.md) · [课程首页](../README.md) · [torch.compile →](08-compile.md)
