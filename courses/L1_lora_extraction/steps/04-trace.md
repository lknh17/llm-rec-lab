<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 先估算](03-estimate.md) · [课程首页](../README.md) · [代码导读 →](05-code.md)

# 第 4 步 / 共 15 步：追踪一条样本

> 阶段 3 / 6：**看清内部**

下面是基线运行第 1 个 batch 的第 1 条样本（真实 tokenizer 输出）。看三件事：

1. `<|im_start|>assistant\n` 本身被监督了吗？`<|im_end|>` 呢？
2. 第 1 个 batch 的 `input_ids` 是 `[8, T]`，为什么 `logits` 的第二维只有几十？
3. LoRA B 的范数和基础权重指纹，训练前后各是多少？

来源：Mac 历史档 · Qwen2.5-0.5B · 单种子，`baseline` s0 的 [`trace.json`](../../../results/mac-m4pro/L1/baseline/s0-f8f01d/trace.json)。

### 样本

- 文案：这款行李箱万向轮、20寸登机、TSA海关锁。满300减50，详情见店铺首页
- 标注：`{"category": "行李箱", "selling_points": ["万向轮", "20寸登机", "TSA海关锁"], "promotions": [{"offer": "满300减50", "condition": null}]}`

| 角色 | 字符数 | 内容（开头） |
| - | - | - |
| system | 429 | 你是广告文案结构化抽取器。从用户给出的广告文案中抽取商业元素，只输出一个 JSON 对象，不要输出任何其他文字。 字段：… |
| user | 37 | 这款行李箱万向轮、20寸登机、TSA海关锁。满300减50，详情见店铺首页 |
| assistant | 112 | {"category":"行李箱","selling_points":["万向轮","20寸登机","TSA海关锁"],… |

### token 与监督 mask

prompt 286 个 token，全长 330，**参与 loss 的 44 个**（13.3%），截断：False。

监督边界附近的 token（✅ = 参与 loss；中间省略的都是 ✅）：

| 位置 | token id | token | 区域 | 监督 |
| - | - | - | - | - |
| 280 | 74876 | `首页` | prompt |  |
| 281 | 151645 | `<\|im_end\|>` | prompt |  |
| 282 | 198 | `\n` | prompt |  |
| 283 | 151644 | `<\|im_start\|>` | prompt |  |
| 284 | 77091 | `assistant` | prompt |  |
| 285 | 198 | `\n` | prompt |  |
| 286 | 4913 | `{\"` | answer | ✅ |
| 287 | 5471 | `category` | answer | ✅ |
| 288 | 3252 | `\":\"` | answer | ✅ |
| 289 | 107956 | `行李` | answer | ✅ |
| 290 | 73296 | `箱` | answer | ✅ |
| 291 | 2198 | `\",\"` | answer | ✅ |
| … | | | | |
| 326 | 92 | `}` | answer | ✅ |
| 327 | 13989 | `]}` | answer | ✅ |
| 328 | 151645 | `<\|im_end\|>` | answer | ✅ |
| 329 | 198 | `\n` | answer | ✅ |

被监督的文本（loss 只算这一段）：

```text
{"category":"行李箱","selling_points":["万向轮","20寸登机","TSA海关锁"],"promotions":[{"offer":"满300减50","condition":null}]}<|im_end|>⏎
```

### 第 1 个 batch 的张量

| 张量 | 形状 | dtype | 备注 |
| - | - | - | - |
| `input_ids` | `[8, 330]` | int64 |  |
| `attention_mask` | `[8, 330]` | int64 |  |
| `labels` | `[8, 330]` | int64 |  |
| `logits` | `[8, 64, 151936]` | torch.float32 | 从第 266 个位置开始算；不切片时是 `[8, 330, 151936]` |

- `per_sample_len`：[330, 297, 305, 309, 302, 323, 311, 295]
- `per_sample_supervised`：[44, 30, 32, 36, 26, 39, 33, 28]
- `pad_tokens`：168
- `batch_loss`：0.47274

### 参数有没有真的更新

| 检查 | 训练前 | 训练后 |
| - | - | - |
| LoRA B 的范数（初始化为 0） | 0.0 | 2.617 |
| 基础权重指纹（必须不变） | -13.570786476135254 | -13.570786476135254 |

### 保存 → 重新加载核对

重新加载基础模型 + 适配器后，抽查 4 条，输出完全一致 4 条。

---

[← 先估算](03-estimate.md) · [课程首页](../README.md) · [代码导读 →](05-code.md)
