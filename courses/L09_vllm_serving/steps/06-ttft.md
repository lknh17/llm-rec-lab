<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [单条 decode →](07-tpot.md)

# 第 6 步 / 共 18 步：单条 prefill

> 阶段 3 / 7：**单条请求** · 场景：`seqs_1`

一次只跑一条（max_num_seqs 1，16 条依次处理）。这时 TTFT 就是单条 4096 token 的 prefill（加上一点调度开销），屋顶线下限 414 ms。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `seqs_1` | 本步 | 一次只跑一条（max_num_seqs 1，16 条）：TTFT 就是单条 prefill，TPOT 就是 batch=1 的 decode 一步 | `max_num_seqs=1`, `num_prompts=16` | 3 |

## 先预测

**实测 TTFT 是屋顶线下限的几倍？**

- **A.** 不到 1.2 倍
- **B.** 1.2 - 1.6 倍
- **C.** 1.6 - 2.5 倍
- **D.** 2.5 倍以上

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只换内容文本和到达时间；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · vLLM 0.11 · Qwen2.5-7B-Instruct bf16 · 输入 4096 / 输出 50 token · 3 种子（每个运行 1 张卡）（已有 0 / 3 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [单条 decode →](07-tpot.md)
