<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 同事改了特征处理](14-fault.md) · [课程首页](../README.md) · [加 worker 能救吗 →](16-fix.md)

# 第 15 步 / 共 17 步：诊断：GPU 在等什么

> 阶段 6 / 7：**故障** · 场景：`fault_pyfeat`（对照 `baseline`）

模型和优化器都没变。按需逐级点开线索，然后点选根因。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：查表微基准（1000 万 × 64 fp32，bag 1 / 10 / 50，均匀与 Zipf）、SparseAdam 与 Adam、数据管线、端到端 DLRM | — | 3 |
| `fault_pyfeat` | 本步 | 特征处理改成逐样本、逐特征的 Python 循环 | `parts=loader,e2e`, `feature_proc=python` | 3 |

## 线索

逐条展开，看到能下判断为止。

<details><summary>线索 1</summary>

逐阶段耗时：等数据 〔待 H20 生成〕 ms（基线 〔待 H20 生成〕），前向反向 〔待 H20 生成〕 ms（基线 〔待 H20 生成〕）。

</details>

<details><summary>线索 2</summary>

只取数不训练：管线每秒 〔待 H20 生成〕 个样本（基线 〔待 H20 生成〕）；处理一批 〔待 H20 生成〕 ms（基线 〔待 H20 生成〕）。

</details>

<details><summary>线索 3</summary>

worker 数 〔待 H20 生成〕：每批平均要多久才能产出？和 GPU 一步比呢？

</details>

## 先预测

**根因是什么？**

- **A.** PCIe 带宽不够
- **B.** GPU 被别的进程占用
- **C.** CPU 上逐样本的 Python 特征处理太慢，worker 跟不上，GPU 在等数据
- **D.** Embedding 表太大

> 先选定一个（记下来），再展开下面的实测。微基准的种子只影响随机输入；耗时差小于 10% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 微基准 + 28 张表的 DLRM · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 同事改了特征处理](14-fault.md) · [课程首页](../README.md) · [加 worker 能救吗 →](16-fix.md)
