<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 同样的 KV 字节，MLA 对 MQA](08-mla.md) · [课程首页](../README.md) · [12 层 Post-Norm →](10-post.md)

# 第 9 步 / 共 15 步：去掉位置编码

> 阶段 4 / 7：**位置编码** · 场景：`nope`（对照 `baseline`）

不加任何位置编码（NoPE）。模型只能靠因果 mask 隐式知道位置：第 t 个位置能看到 t 个 token，注意力的分母随位置变化。另外在 2 倍训练长度（1024 token）的窗口上评估，看外推。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 对照 | 基线：MHA 6 头、RoPE、Pre-Norm、Dense SwiGLU（= L01 基线） | — | 3 |
| `nope` | 本步 | 不加位置编码（NoPE） | `pos_emb=none` | 3 |

## 先预测

**去掉 RoPE，留出集 bpb 会差多少？**

- **A.** 差不到 0.02
- **B.** 差 0.02 - 0.1
- **C.** 差 0.1 以上

> 先选定一个（记下来），再展开下面的实测。bpb 差异小于 0.02 或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · 小 GPT（6 层 d=384）从零训练 100 MB 中文维基 · 每个场景只换一处结构 · 3 种子（已有 0 / 6 个运行）</summary>

_还没有结果：H20 正在跑或还在排队，进度见 [作业看板](../../../jobs/README.md)。_

</details>

## 看完实测再想一步

两条验证曲线在训练早期差得多，还是后期差得多？

---

[← 同样的 KV 字节，MLA 对 MQA](08-mla.md) · [课程首页](../README.md) · [12 层 Post-Norm →](10-post.md)
