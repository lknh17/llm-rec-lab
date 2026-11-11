<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 单条 decode](07-tpot.md) · [课程首页](../README.md) · [并发的代价 →](09-latency.md)

# 第 8 步 / 共 18 步：max_num_seqs 的拐点

> 阶段 4 / 7：**并发** · 场景：`seqs_256` `seqs_1` `seqs_8` `baseline` `seqs_128`（对照 `seqs_8`）

256 条请求同时到达，只改 max_num_seqs（同时在跑的请求上限）：1、8、32（基线）、128、256。decode 的权重读取会被同一步里的所有请求分摊，batch 越大 decode 越划算。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `seqs_8` | 对照 | max_num_seqs 8 | `max_num_seqs=8` | 3 |
| `seqs_256` | 本步 | max_num_seqs 256 | `max_num_seqs=256` | 3 |
| `seqs_1` | 本步 | 一次只跑一条（max_num_seqs 1，16 条）：TTFT 就是单条 prefill，TPOT 就是 batch=1 的 decode 一步 | `max_num_seqs=1`, `num_prompts=16` | 3 |
| `seqs_8` | 本步 | max_num_seqs 8 | `max_num_seqs=8` | 3 |
| `baseline` | 本步 | 基线：7B bf16，256 条同时到达，max_num_seqs 32，每步 token 预算 8192，前缀缓存开（没有共享前缀） | — | 3 |
| `seqs_128` | 本步 | max_num_seqs 128 | `max_num_seqs=128` | 3 |

## 先预测

**max_num_seqs 从 8 加到 256，吞吐（条/s）变成几倍？**

- **A.** 不到 1.2 倍（基本不变）
- **B.** 1.2 - 2 倍
- **C.** 2 倍以上

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只换内容文本和到达时间；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · vLLM 0.11 · Qwen2.5-7B-Instruct bf16 · 输入 4096 / 输出 50 token · 3 种子（每个运行 1 张卡）（已有 0 / 15 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 单条 decode](07-tpot.md) · [课程首页](../README.md) · [并发的代价 →](09-latency.md)
