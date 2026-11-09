<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 长度的成本](09-cost-len.md) · [课程首页](../README.md) · [手写注意力对 SDPA →](11-manual.md)

# 第 10 步 / 共 17 步：SASRec（softmax）对 HSTU

> 阶段 4 / 6：**结构与实现** · 场景：`sasrec_10k`（对照 `baseline`）

同样 10240 的窗口、同样的训练预算，换成 SASRec：softmax 注意力 + MLP，绝对位置，没有时间偏置，SDPA（flash）实现。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：HSTU，最近 10240 个行为（按 query 分块计算注意力） | — | 3 |
| `sasrec_10k` | 本步 | SASRec（softmax 注意力，SDPA），L = 10240 | `model=sasrec` | 3 |

## 先预测

**长程题的 Recall@10，SASRec 和 HSTU 比怎样？**

- **A.** SASRec 低 0.05 以上
- **B.** 差不多（±0.05）
- **C.** SASRec 高 0.05 以上

> 先选定一个（记下来），再展开下面的实测。Recall@10 在 2048 个测试用户上的抽样标准误约 0.01，长程题（约 800 个）约 0.017；均值差小于 0.03（长程题小于 0.05）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 合成长序列 1K-10K（回访簇作长程信号）· HSTU d=64、2 层 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

三类题里，两种结构差得最多的是哪一类？

---

[← 长度的成本](09-cost-len.md) · [课程首页](../README.md) · [手写注意力对 SDPA →](11-manual.md)
