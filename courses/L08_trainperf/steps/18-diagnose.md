<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 换成在线分词](17-fault.md) · [课程首页](../README.md) · [回顾 →](19-keycard.md)

# 第 18 步 / 共 19 步：诊断：MFU 掉在哪

> 阶段 7 / 8：**故障** · 场景：`fault_dataloader`（对照 `baseline`）

模型、并行方式都没变，只换了数据来源。按需逐级点开线索，然后点选根因。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20 × 8） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：1.5B，DDP 8 卡，seq 2048，mbs 2，bf16，SDPA，fused AdamW | — | 3 |
| `fix_dataloader` | 参照 | 同样的 CPU 预处理，4 个 worker + 锁页内存 + 预取 | `data=cpu_pipeline`, `num_workers=4`, `pin_memory=True` | 3 |
| `fault_dataloader` | 本步 | 数据在每个 rank 的主进程里做 CPU 规范化 + 分词（num_workers 0） | `data=cpu_pipeline`, `num_workers=0` | 3 |

## 线索

逐条展开，看到能下判断为止。

<details><summary>线索 1</summary>

步时分解：数据 〔待 H20 生成〕 ms（基线 〔待 H20 生成〕），前向 〔待 H20 生成〕、反向 〔待 H20 生成〕 ms（基线 〔待 H20 生成〕、〔待 H20 生成〕）。

</details>

<details><summary>线索 2</summary>

profiler：每步 kernel 计算 〔待 H20 生成〕 ms（基线 〔待 H20 生成〕），GPU 空闲占比 〔待 H20 生成〕（基线 〔待 H20 生成〕）。

</details>

<details><summary>线索 3</summary>

配置：num_workers = 0；每个 rank 每步要处理 mbs 条样本；机器 CPU 核数 〔待 H20 生成〕。

</details>

## 先预测

**根因是什么？**

- **A.** 梯度通信没有和计算重叠
- **B.** 数据在训练主进程里同步做 CPU 预处理，GPU 空等
- **C.** 显存碎片导致分配器反复整理
- **D.** mbs 太小，矩阵乘吃不饱

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只影响随机输入；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 × 8 · 类 Qwen2.5-1.5B / 7B · seq 2048 · bf16 · 3 种子（部分 1 种子）（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 换成在线分词](17-fault.md) · [课程首页](../README.md) · [回顾 →](19-keycard.md)
