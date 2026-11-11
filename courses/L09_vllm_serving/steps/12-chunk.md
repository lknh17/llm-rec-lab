<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 前缀缓存](11-prefix.md) · [课程首页](../README.md) · [FP8 →](13-fp8.md)

# 第 12 步 / 共 18 步：每步 token 预算（chunked prefill）

> 阶段 5 / 7：**省 FLOPs 还是省字节** · 场景：`chunk_2k`（对照 `chunk_16k`）

换成在线场景：请求按每秒 1 条随机到达（200 条），看用户体验。每步 token 预算 16384 时，一条 4096 的 prefill 一步做完，这一步正在 decode 的请求全都要等；预算 2048 时，prefill 被切成 2048 一块，和 decode 拼在一起分几步做。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `chunk_16k` | 对照 | 在线到达 1 条/s（200 条），每步 token 预算 16384：一条 4096 的 prefill 一步做完，正在 decode 的请求跟着等 | `request_rate=1.0`, `num_prompts=200`, `max_num_batched_tokens=16384` | 3 |
| `chunk_2k` | 本步 | 在线到达 1 条/s（200 条），每步 token 预算 2048：prefill 切成 2048 的块和 decode 拼在一起 | `request_rate=1.0`, `num_prompts=200`, `max_num_batched_tokens=2048` | 3 |

## 先预测

**预算从 16384 降到 2048，ITL p99（相邻两个输出之间最长的那批间隔）变成多少？**

- **A.** 不到原来的 40%
- **B.** 原来的 40% - 80%
- **C.** 80% 以上（差不多或更差）

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只换内容文本和到达时间；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · vLLM 0.11 · Qwen2.5-7B-Instruct bf16 · 输入 4096 / 输出 50 token · 3 种子（每个运行 1 张卡）（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 前缀缓存](11-prefix.md) · [课程首页](../README.md) · [FP8 →](13-fp8.md)
