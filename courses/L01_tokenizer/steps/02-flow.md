<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一段文本 →](03-trace.md)

# 第 2 步 / 共 15 步：流程结构

> 阶段 1 / 6：**理解任务**

语料准备的每一步都有缓存：同一份语料和分词器只建一次，3 个种子、只换模型参数的场景都复用。点击每个环节看它的输入输出和对应代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 清洗 | 23 万篇 → 去空括号、多余空行，丢掉 < 50 字的 | 维基转纯文本后，被删掉的模板会留下"（）"之类的痕迹。 | [`tinylm/corpus.py#L45-L49`](../../../projects/tinylm/corpus.py#L45-L49) |
| 2. 底稿去重 | MinHash 去掉维基自带的模板化近重复条目 | 市镇、物种等条目由模板生成，彼此高度相似（约占 8% 字节）。先去掉，干净留出集和验证集才同分布。 | [`tinylm/corpus.py#L112-L165`](../../../projects/tinylm/corpus.py#L112-L165) |
| 3. 模拟转载 + 去重 + 切分 | 20% 文章 × 1-40 份副本 → 去重 → 按 URL 哈希切分 | 留出集在注入副本之前拿走。去重只看文本；原文优先保留；验证集按（来源, 副本号）哈希切分，各场景验证集里的原文是同一批。 | [`tinylm/corpus.py#L220-L281`](../../../projects/tinylm/corpus.py#L220-L281) |
| 4. 训练 BPE | 训练集前 20 MB → 词表 V | HF tokenizers 的字节级 BPE，预切分正则与 Qwen2 相同（数字、标点、汉字串分开）。 | [`tinylm/tokenizer.py#L86-L100`](../../../projects/tinylm/tokenizer.py#L86-L100) |
| 5. token 流 | 文档首尾相接 + 分隔符 → [N_tokens] | 训练时随机切成 [B, T] = [64, 512] 的块，目标是右移一位的同一块。 |  |
| 6. 小 GPT | [B, T] → [B, T, V] | Pre-RMSNorm、RoPE、SwiGLU；输入 embedding 和输出层共享同一个 V × d 矩阵。 | [`tinylm/model.py#L201-L243`](../../../projects/tinylm/model.py#L201-L243) |
| 7. 训练 | 100 MB 文本过一遍 | AdamW（只对矩阵权重做衰减），warmup 后余弦降到 10%，梯度裁剪 1.0。 | [`tinylm/train.py#L163-L295`](../../../projects/tinylm/train.py#L163-L295) |
| 8. bits-per-byte | Σ NLL / ln 2 / 字节数 | 评估所有子集：验证集、留出集、广告、验证集按泄漏拆的组。 | [`tinylm/train.py#L81-L118`](../../../projects/tinylm/train.py#L81-L118) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一段文本 →](03-trace.md)
