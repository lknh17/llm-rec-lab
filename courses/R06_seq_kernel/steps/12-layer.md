<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 离 roofline 多远](11-roofline.md) · [课程首页](../README.md) · [同事改了 kernel →](13-fault.md)

# 第 12 步 / 共 15 步：attention 在整层里占多少

> 阶段 4 / 6：**块大小与 roofline** · 场景：`triton`（对照 `baseline`）

一个 HSTU 层（d_model = 512、8 头、L = 2048）：f1 投影 → 逐点 attention → 门控 → f2 投影，前向。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：naive（PyTorch 逐步，物化 L × L） | — | 3 |
| `triton` | 本步 | 手写 Triton 融合 kernel（块 64 × 64） | `attn_impl=triton` | 3 |

## 先预测

**用 Triton kernel 时，attention 占整层前向耗时的多少？**

- **A.** 不到 30%
- **B.** 30% - 70%
- **C.** 70% 以上

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · B=4、H=8、L 512-8192、D 64/128、bf16 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 离 roofline 多远](11-roofline.md) · [课程首页](../README.md) · [同事改了 kernel →](13-fault.md)
