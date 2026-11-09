<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 19 步：追踪一条样本的四种口径

> 阶段 1 / 7：**理解任务**

训练第 1 个 batch 的第 1 条样本，在四种口径下的用户 / 电影统计并排：此前次数 cnt、此前正例数 pos、平滑喜欢率 rate。对照着看：
- include_self 比 point_in_time 多了 1 次、多了本行的标签；
- full 的次数是这个用户 / 这部电影的全部评分，包括这条样本之后的；
- daily 只到样本前一天。

〔待 H20 生成〕
---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
