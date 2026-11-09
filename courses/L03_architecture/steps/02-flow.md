<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一层注意力 →](03-trace.md)

# 第 2 步 / 共 15 步：流程结构

> 阶段 1 / 7：**理解任务**

训练循环和 L02 完全一样，只是 Block 里的几个部件可以替换。点击每个部件看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. embedding | [B, T] → [B, T, C] | 和输出层共享。learned 位置编码时还要加一张 [T, C] 的位置表。 | [`tinylm/model.py#L201-L243`](../../../projects/tinylm/model.py#L201-L243) |
| 2. 注意力变体 | q [B, H, T, D]；k、v [B, Hkv, T, D] | MHA：Hkv = H；GQA：H 个查询头分成 Hkv 组；MQA：Hkv = 1。 | [`tinylm/model.py#L143-L158`](../../../projects/tinylm/model.py#L143-L158) |
| 3. MLA | x → c [B, T, R] + k_rope [B, T, Dr] → 各头 k、v | 缓存 c 和 k_rope；RoPE 只加在解耦出来的 Dr 维上。 | [`tinylm/model.py#L162-L181`](../../../projects/tinylm/model.py#L162-L181) |
| 4. Norm 与残差 | Pre：x + f(norm(x))；Post：norm(x + f(x)) | Pre-Norm 的残差流不经过 norm，梯度有一条恒等路径。 | [`tinylm/model.py#L191-L196`](../../../projects/tinylm/model.py#L191-L196) |
| 5. FFN 或 MoE | [N, C] → 路由 [N, E] → top-k → 专家 → 加权和 | MoE 每个专家是中间维 512 的 SwiGLU，top-2 时激活参数和 1024 的 Dense 相同。 | [`tinylm/model.py#L89-L109`](../../../projects/tinylm/model.py#L89-L109) |
| 6. KV cache 统计 | 每层每 token 缓存的元素数 × 层数 × 2 字节 | 不训练也能算：只取决于结构。 | [`tinylm/model.py#L297-L304`](../../../projects/tinylm/model.py#L297-L304) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一层注意力 →](03-trace.md)
