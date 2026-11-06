<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [先估算 →](03-estimate.md)

# 第 2 步 / 共 13 步：迭代闭环

> 阶段 1 / 8：**理解任务**

一次"围绕错误"的迭代分 6 个环节。点击每个环节看它的输入输出，以及对应的代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 逐条评分与错误归类 | 预测 JSON + 标注 → errors: [漏条件, 多卖点, …] | 每条样本按字段判对错并给出错误类型；按样本标签（排除型条件、多优惠……）分组统计。 | [`adextract/schema.py#L83-L133`](../../../projects/adextract/schema.py#L83-L133) |
| 2. 假设 | 一类错误 → 一个原因 | 排除型条件全错，而训练集里没有这种句式：数据覆盖问题。 |  |
| 3. 只改一个变量 | exclusion_ratio：替换 round(n × ratio) 条 | 用独立的随机数流挑位置替换，其余样本和基线逐条相同；词表避开 dev/test 原句。 | [`adextract/synth.py#L135-L163`](../../../projects/adextract/synth.py#L135-L163) |
| 4. 多种子训练 | 同一配置 × 5 个训练种子 | 种子影响 LoRA 初始化和打乱顺序；看均值和种子间波动。 | [`adextract/train.py#L73-L134`](../../../projects/adextract/train.py#L73-L134) |
| 5. 分组 + 逐样本对照 | 同种子配对：修好 / 弄坏 / 都对 / 都错 | 整体均值之外，看目标类别是否变好、被修好的是不是同一批样本、有没有弄坏别的。 | [`adextract/schema.py#L138-L163`](../../../projects/adextract/schema.py#L138-L163) |
| 6. 保留集检验 | test 30 条，只看一次 | 迭代时只看 dev；决定方案之后才看 test，回答提升能否泛化。推理必须和训练共用同一个模板函数。 | [`adextract/infer.py#L14-L36`](../../../projects/adextract/infer.py#L14-L36) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [先估算 →](03-estimate.md)
