<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 不加去重位](12-nodedup.md) · [课程首页](../README.md) · [同事改了 beam search →](14-fault.md)

# 第 13 步 / 共 16 步：和 SASRec 比

> 阶段 6 / 8：**和 SASRec 比** · 场景：`sasrec`（对照 `baseline`）

参照：SASRec（和 R01 基线同一套代码）对全部 3706 部电影打分取 Top-10。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：3 × 64 + 去重位，trie 约束，beam 10 | — | 3 |
| `sasrec` | 本步 | SASRec 对全部物品打分（和 R01 基线同一套代码） | `method=sasrec` | 3 |

## 先预测

**SASRec 和生成式召回比，Recall@10 怎样？**

- **A.** SASRec 高 0.02 以上
- **B.** 差不多（±0.02）
- **C.** 生成式召回高 0.02 以上

> 先选定一个（记下来），再展开下面的实测。Recall@10 在 6040 个测试用户上的抽样标准误约 0.005；均值差小于 0.01 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · ML-1M · 语义 ID 3 × 64 + 去重位 · encoder-decoder（4 + 2 层）· 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 不加去重位](12-nodedup.md) · [课程首页](../README.md) · [同事改了 beam search →](14-fault.md)
