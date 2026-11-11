<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← AWQ INT4](14-awq.md) · [课程首页](../README.md) · [规范挪到内容后面 →](16-fault.md)

# 第 15 步 / 共 18 步：投机解码

> 阶段 5 / 7：**省 FLOPs 还是省字节** · 场景：`spec_ngram_c1`（对照 `seqs_1`）

n-gram 投机（不需要草稿模型）：用最近生成的几个 token 去输入里找相同的片段，把它后面的 4 个 token 当草稿，目标模型一步验证 5 个位置。打标输出的 JSON 大量原文摘录内容里的卖点和优惠，草稿经常能对上。先看一次只跑一条（16 条）。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `seqs_1` | 对照 | 一次只跑一条（max_num_seqs 1，16 条）：TTFT 就是单条 prefill，TPOT 就是 batch=1 的 decode 一步 | `max_num_seqs=1`, `num_prompts=16` | 3 |
| `spec_draft_c1` | 参照 | Qwen2.5-0.5B 草稿模型（k=4），一次只跑一条（16 条） | `spec=draft`, `max_num_seqs=1`, `num_prompts=16` | 3 |
| `spec_ngram_c32` | 参照 | n-gram 投机（k=4），max_num_seqs 32（256 条） | `spec=ngram` | 3 |
| `baseline` | 参照 | 基线：7B bf16，256 条同时到达，max_num_seqs 32，每步 token 预算 8192，前缀缓存开（没有共享前缀） | — | 3 |
| `spec_ngram_c1` | 本步 | n-gram 投机（k=4），一次只跑一条（16 条） | `spec=ngram`, `max_num_seqs=1`, `num_prompts=16` | 3 |

## 先预测

**TPOT 变成不投机时的多少？**

- **A.** 不到 60%（快很多）
- **B.** 60% - 90%
- **C.** 90% - 110%（差不多）
- **D.** 110% 以上（变慢）

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只换内容文本和到达时间；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · vLLM 0.11 · Qwen2.5-7B-Instruct bf16 · 输入 4096 / 输出 50 token · 3 种子（每个运行 1 张卡）（已有 0 / 15 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← AWQ INT4](14-awq.md) · [课程首页](../README.md) · [规范挪到内容后面 →](16-fault.md)
