<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)

# 第 3 步 / 共 19 步：追踪一条样本的四种口径

> 阶段 1 / 7：**理解任务**

训练第 1 个 batch 的第 1 条样本，在四种口径下的用户 / 电影统计并排：此前次数 cnt、此前正例数 pos、平滑喜欢率 rate。对照着看：
- include_self 比 point_in_time 多了 1 次、多了本行的标签；
- full 的次数是这个用户 / 这部电影的全部评分，包括这条样本之后的；
- daily 只到样本前一天。

来源：H20 · MovieLens-1M 全量 · 在 R0a 基线上逐组加特征、换统计口径 · 3-5 种子，`baseline` s0 的 [`trace.json`](../../../results/h20x8/R0b/baseline/s0-461376/trace.json)。

### 样本

- 文案：
- 标注：`null`

### 第 1 个 batch 的张量

| 张量 | 形状 | dtype | 备注 |
| - | - | - | - |
| `mlp_input` | `[2048, 64]` | float32 |  |


### 保存 → 重新加载核对

重新加载基础模型 + 适配器后，抽查 None 条，输出完全一致 None 条。

---

[← 流程结构](02-flow.md) · [课程首页](../README.md) · [代码导读 →](04-code.md)
