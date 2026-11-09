<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 同事改了 beam search](14-fault.md) · [课程首页](../README.md) · [回顾 →](16-keycard.md)

# 第 15 步 / 共 16 步：诊断：模型没变，召回为什么掉了

> 阶段 7 / 8：**故障** · 场景：`fault_beamdup`（对照 `baseline`）

模型是同一个（训练缓存命中），只改了 beam search 的初始化。按需逐级点开线索，然后点选根因。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：3 × 64 + 去重位，trie 约束，beam 10 | — | 3 |
| `fault_beamdup` | 本步 | beam 的初始分数全设成 0 | `beam_init=all_zero` | 3 |

## 线索

逐条展开，看到能下判断为止。

<details><summary>线索 1</summary>

beam 扫描：W = 1 / 10 / 20 时 Recall 〔待 H20 生成〕 / 〔待 H20 生成〕 / 〔待 H20 生成〕；基线 〔待 H20 生成〕 / 〔待 H20 生成〕 / 〔待 H20 生成〕。

</details>

<details><summary>线索 2</summary>

每用户不重复物品数：故障 〔待 H20 生成〕，基线 〔待 H20 生成〕；非法元组 〔待 H20 生成〕。

</details>

<details><summary>线索 3</summary>

看追踪里同一个用户的 10 条 beam：token 元组和分数是否完全相同。

</details>

## 先预测

**根因是什么？**

- **A.** trie 掩码错了
- **B.** beam 初始分数全为 0，同一个前缀被扩展了 W 次，W 条 beam 完全相同
- **C.** 模型坍缩到热门物品
- **D.** 去重位让碰撞的物品无法区分

> 先选定一个（记下来），再展开下面的实测。Recall@10 在 6040 个测试用户上的抽样标准误约 0.005；均值差小于 0.01 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · ML-1M · 语义 ID 3 × 64 + 去重位 · encoder-decoder（4 + 2 层）· 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 同事改了 beam search](14-fault.md) · [课程首页](../README.md) · [回顾 →](16-keycard.md)
