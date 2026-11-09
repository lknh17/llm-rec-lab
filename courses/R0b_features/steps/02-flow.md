<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一条样本的四种口径 →](03-trace.md)

# 第 2 步 / 共 19 步：流程结构

> 阶段 1 / 7：**理解任务**

和 R0a 相同的训练链路，多了特征组和统计口径。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. R0a 基线 | 用户 ID + 电影 ID → Embedding + MLP | 全局时间切分，固定未来留出集。 | [`recbase/data.py#L78-L103`](../../../projects/recbase/data.py#L78-L103) |
| 2. 历史统计 | (key, 时间, 标签) → 每行的此前次数、此前正例数 | 按 (key, 时间) 排序后做分组前缀和；四种口径只差"统计到哪一刻为止"。 | [`recbase/features.py#L40-L58`](../../../projects/recbase/features.py#L40-L58) |
| 3. 训练口径与线上口径 | 每行两套特征：场景口径 / 线上口径 | 训练、验证用场景口径；留出集和 valid_serving 永远用线上口径。另做口径体检：多少行数进了未来。 | [`recbase/experiment.py#L135-L162`](../../../projects/recbase/experiment.py#L135-L162) |
| 4. 特征组进模型 | [B, F, d] 拼接 + dense [B, n] → MLP | 画像、年代、次数分桶走 embedding；题材是多值字段求平均；喜欢率和 log 次数是 dense。 | [`recbase/model.py#L36-L96`](../../../projects/recbase/model.py#L36-L96) |
| 5. 分组评估 | 每组 {n, 正例率, AUC, GAUC} | 新用户 / 老用户、此前评分数、长尾 / 头部电影。 | [`recbase/metrics.py#L114-L137`](../../../projects/recbase/metrics.py#L114-L137) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪一条样本的四种口径 →](03-trace.md)
