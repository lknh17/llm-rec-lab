<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← max_num_seqs 的拐点](08-batch.md) · [课程首页](../README.md) · [KV 显存与并发上限 →](10-kv.md)

# 第 9 步 / 共 18 步：并发的代价

> 阶段 4 / 7：**并发** · 场景：`seqs_256` `seqs_8` `baseline` `seqs_128`（对照 `baseline`）

同一组运行，看延迟。max_num_seqs 越大，同时在 decode 的请求越多；而每一步里只要还有请求在等 prefill，就会塞进一块 prefill（每步 token 预算 8192）。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：7B bf16，256 条同时到达，max_num_seqs 32，每步 token 预算 8192，前缀缓存开（没有共享前缀） | — | 3 |
| `seqs_256` | 本步 | max_num_seqs 256 | `max_num_seqs=256` | 3 |
| `seqs_8` | 本步 | max_num_seqs 8 | `max_num_seqs=8` | 3 |
| `baseline` | 本步 | 基线：7B bf16，256 条同时到达，max_num_seqs 32，每步 token 预算 8192，前缀缓存开（没有共享前缀） | — | 3 |
| `seqs_128` | 本步 | max_num_seqs 128 | `max_num_seqs=128` | 3 |

## 先预测

**max_num_seqs 256 的 TPOT p50 是基线（32）的几倍？**

- **A.** 不到 1.5 倍
- **B.** 1.5 - 4 倍
- **C.** 4 倍以上

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只换内容文本和到达时间；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · vLLM 0.11 · Qwen2.5-7B-Instruct bf16 · 输入 4096 / 输出 50 token · 3 种子（每个运行 1 张卡）（已有 0 / 12 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← max_num_seqs 的拐点](08-batch.md) · [课程首页](../README.md) · [KV 显存与并发上限 →](10-kv.md)
