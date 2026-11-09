<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 同事改了 kernel](13-fault.md) · [课程首页](../README.md) · [回顾 →](15-keycard.md)

# 第 14 步 / 共 15 步：诊断：哪里看到了未来

> 阶段 5 / 6：**故障** · 场景：`fault_diag`（对照 `triton`）

只删了块内的因果 mask。按需逐级点开线索，然后点选根因。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `triton` | 对照 | 手写 Triton 融合 kernel（块 64 × 64） | `attn_impl=triton` | 3 |
| `fault_diag` | 本步 | Triton 的对角块内部没有做因果 mask | `attn_impl=triton`, `diag_causal=False` | 3 |

## 线索

逐条展开，看到能下判断为止。

<details><summary>线索 1</summary>

误差的量级：正确的 kernel 〔待 H20 生成〕（bf16 噪声），故障 〔待 H20 生成〕。

</details>

<details><summary>线索 2</summary>

是不是只在尾块出错：L = 2000（不是 64 的倍数）〔待 H20 生成〕，L = 2048 〔待 H20 生成〕。两种打分：逐点 〔待 H20 生成〕，softmax 〔待 H20 生成〕。

</details>

<details><summary>线索 3</summary>

第 m 个 query 块循环到哪一块为止？最后那一块里，第 i 行和第 j 列（j > i）是什么关系？

</details>

## 先预测

**根因是什么？**

- **A.** bf16 精度不够
- **B.** 尾块越界读了垃圾数据
- **C.** 对角块里 query 看到了同一块里更靠后的 key（未来）
- **D.** online softmax 的缩放写错了

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · B=4、H=8、L 512-8192、D 64/128、bf16 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 同事改了 kernel](13-fault.md) · [课程首页](../README.md) · [回顾 →](15-keycard.md)
