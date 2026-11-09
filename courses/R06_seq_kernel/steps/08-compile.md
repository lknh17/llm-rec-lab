<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← SDPA](07-sdpa.md) · [课程首页](../README.md) · [手写 Triton →](09-triton.md)

# 第 8 步 / 共 15 步：torch.compile

> 阶段 3 / 6：**四种实现** · 场景：`compiled`（对照 `baseline`）

不改代码，把 naive 交给 torch.compile。编译器（Inductor）会把逐元素操作融合起来，但 QKᵀ 和 AV 两个矩阵乘之间的 L × L 张量能不能消掉，要看它能不能识别出整个 attention 模式。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：naive（PyTorch 逐步，物化 L × L） | — | 3 |
| `compiled` | 本步 | torch.compile(naive) | `attn_impl=compiled` | 3 |

## 先预测

**L = 8192 逐点打分，torch.compile 的增量显存比 naive 少多少？**

- **A.** 少不到一半
- **B.** 少一半到 90%
- **C.** 少 90% 以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · B=4、H=8、L 512-8192、D 64/128、bf16 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← SDPA](07-sdpa.md) · [课程首页](../README.md) · [手写 Triton →](09-triton.md)
