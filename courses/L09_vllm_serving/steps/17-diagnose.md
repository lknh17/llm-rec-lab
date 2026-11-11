<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 规范挪到内容后面](16-fault.md) · [课程首页](../README.md) · [回顾 →](18-keycard.md)

# 第 17 步 / 共 18 步：诊断：缓存为什么没命中

> 阶段 6 / 7：**故障** · 场景：`fault_prefix_tail`（对照 `baseline`）

引擎配置没变，只改了 prompt 的拼法。按需逐级点开线索，然后点选根因。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：7B bf16，256 条同时到达，max_num_seqs 32，每步 token 预算 8192，前缀缓存开（没有共享前缀） | — | 3 |
| `fix_prefix_head` | 参照 | 修复：同样 3072 token 的规范放回开头 | `shared_prefix=3072`, `prefix_position=head` | 3 |
| `fault_prefix_tail` | 本步 | 故障：3072 token 的打标规范放在内容后面（前缀缓存开着，但每条请求的开头都不同） | `shared_prefix=3072`, `prefix_position=tail` | 3 |

## 线索

逐条展开，看到能下判断为止。

<details><summary>线索 1</summary>

前缀命中率：〔待 H20 生成〕（引擎配置里 enable_prefix_caching = true）；TTFT p50 〔待 H20 生成〕 ms（基线 〔待 H20 生成〕 ms）。

</details>

<details><summary>线索 2</summary>

追踪：前两条请求开头相同的 token 数 〔待 H20 生成〕；KV 块大小 16 token。

</details>

<details><summary>线索 3</summary>

KV 容量 〔待 H20 生成〕 token；max_num_seqs 〔待 H20 生成〕，和基线相同。

</details>

## 先预测

**根因是什么？**

- **A.** 前缀缓存没有开
- **B.** 共享的规范放在内容后面，每条请求的开头都不同，按块哈希的前缀匹配不上
- **C.** KV cache 太小，缓存的块很快被挤掉
- **D.** max_num_seqs 太小，请求之间没有重叠

> 先选定一个（记下来），再展开下面的实测。性能测量的种子只换内容文本和到达时间；吞吐差小于 5% 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · vLLM 0.11 · Qwen2.5-7B-Instruct bf16 · 输入 4096 / 输出 50 token · 3 种子（每个运行 1 张卡）（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

---

[← 规范挪到内容后面](16-fault.md) · [课程首页](../README.md) · [回顾 →](18-keycard.md)
