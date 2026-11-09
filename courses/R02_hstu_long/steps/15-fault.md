<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 补齐 vs jagged](14-jagged.md) · [课程首页](../README.md) · [诊断：短用户为什么掉得多 →](16-diagnose.md)

# 第 15 步 / 共 17 步：同事简化了 mask

> 阶段 5 / 6：**padding** · 场景：`fault_padmask`（对照 `baseline`）

同事觉得 HSTU 的 mask"三个条件太啰嗦，因果就够了"，只保留了因果 mask。序列是左补齐的。训练正常结束。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：HSTU，最近 10240 个行为（按 query 分块计算注意力） | — | 3 |
| `fault_padmask` | 本步 | HSTU 的 mask 只做因果（左补齐的 padding 没挡住） | `mask=causal_only` | 3 |

## 先预测

**短用户（总长度 < 2048）的测试 Recall@10 会怎样？**

- **A.** 降不到 0.01
- **B.** 降 0.01 - 0.05
- **C.** 降 0.05 以上

> 先选定一个（记下来），再展开下面的实测。Recall@10 在 2048 个测试用户上的抽样标准误约 0.01，长程题（约 800 个）约 0.017；均值差小于 0.03（长程题小于 0.05）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 合成长序列 1K-10K（回访簇作长程信号）· HSTU d=64、2 层 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

训练 loss 曲线看得出问题吗？

---

[← 补齐 vs jagged](14-jagged.md) · [课程首页](../README.md) · [诊断：短用户为什么掉得多 →](16-diagnose.md)
