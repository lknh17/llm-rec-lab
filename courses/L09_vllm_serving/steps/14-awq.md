<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← FP8](13-fp8.md) · [课程首页](../README.md) · [投机解码 →](15-spec.md)

# 第 14 步 / 共 18 步：AWQ INT4

> 阶段 5 / 7：**省 FLOPs 还是省字节** · 场景：`awq`（对照 `baseline`）

换成官方的 AWQ 版（权重 INT4，激活仍是 16 位，W4A16），vLLM 用 awq_marlin kernel：矩阵乘时把 INT4 权重在片上解包成 16 位再算。权重从 15.2 GB 降到约 4 GB，但计算仍是 16 位 Tensor Core。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：7B bf16，256 条同时到达，max_num_seqs 32，每步 token 预算 8192，前缀缓存开（没有共享前缀） | — | 3 |
| `awq_c1` | 参照 | AWQ INT4，一次只跑一条（max_num_seqs 1，16 条） | `model=qwen2.5-7b-awq`, `max_num_seqs=1`, `num_prompts=16` | 3 |
| `seqs_1` | 参照 | 一次只跑一条（max_num_seqs 1，16 条）：TTFT 就是单条 prefill，TPOT 就是 batch=1 的 decode 一步 | `max_num_seqs=1`, `num_prompts=16` | 3 |
| `awq` | 本步 | AWQ INT4 权重（W4A16，awq_marlin） | `model=qwen2.5-7b-awq` | 3 |

## 先预测

**同样 256 条、max_num_seqs 32，吞吐是 bf16 基线的几倍？**

- **A.** 不到 0.9 倍（变慢）
- **B.** 0.9 - 1.15 倍（差不多）
- **C.** 1.15 倍以上（变快）

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只换内容文本和到达时间；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · vLLM 0.11 · Qwen2.5-7B-Instruct bf16 · 输入 4096 / 输出 50 token · 3 种子（每个运行 1 张卡）（已有 0 / 12 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← FP8](13-fp8.md) · [课程首页](../README.md) · [投机解码 →](15-spec.md)
