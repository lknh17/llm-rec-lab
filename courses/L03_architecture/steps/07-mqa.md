<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← GQA：6 个查询头共享 2 组 KV](06-gqa.md) · [课程首页](../README.md) · [同样的 KV 字节，MLA 对 MQA →](08-mla.md)

# 第 7 步 / 共 15 步：MQA：只剩 1 组 KV

> 阶段 3 / 7：**注意力变体** · 场景：`mqa`（对照 `baseline`）

再极端一点：6 个查询头共享同一组 k、v。KV cache 只剩 MHA 的 1/6。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：MHA 6 头、RoPE、Pre-Norm、Dense SwiGLU（= L01 基线） | — | 3 |
| `gqa2` | 参照 | GQA：6 个查询头共享 2 组 KV | `n_kv_head=2` | 3 |
| `mqa` | 本步 | MQA：6 个查询头共享 1 组 KV | `n_kv_head=1` | 3 |

## 先预测

**和 MHA 比，MQA 的留出集 bpb 怎样？**

- **A.** 更好（低 0.02 以上）
- **B.** 差不多（±0.02）
- **C.** 更差（高 0.02 以上）

> 先选定一个（记下来），再展开下面的实测。bpb 差异小于 0.02 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 小 GPT（6 层 d=384）从零训练 100 MB 中文维基 · 每个场景只换一处结构 · 3 种子（已有 0 / 9 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

MHA → GQA → MQA，bpb 是单调变差的吗？

---

[← GQA：6 个查询头共享 2 组 KV](06-gqa.md) · [课程首页](../README.md) · [同样的 KV 字节，MLA 对 MQA →](08-mla.md)
