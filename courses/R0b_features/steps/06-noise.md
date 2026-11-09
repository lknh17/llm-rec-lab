<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [加用户画像 →](07-profile.md)

# 第 6 步 / 共 19 步：只换种子

> 阶段 3 / 7：**噪声线** · 场景：`baseline`

基线配置完全不变，只换训练种子（初始化和样本顺序），跑 5 次。验证集、特征、超参数都一样。

## 这一步跑了什么

| 场景 | 作用 | 说明 | 相对基线改了什么 | 种子（H20） |
| - | - | - | - | - |
| `baseline` | 本步 | 基线：只用用户 ID + 电影 ID（= R0a 基线），5 个种子 | — | 5 |

## 先预测

**5 个种子的验证 AUC，最高 − 最低约多少？**

- **A.** 不到 0.1 个点
- **B.** 0.1 - 0.4 个点
- **C.** 0.4 个点以上

> 先选定一个（记下来），再展开下面的实测。AUC 差异小于 0.3 个百分点（GAUC 小于 0.5）或小于种子间波动，都按“差不多”算。

## 实测

<details><summary>展开实测：H20 · MovieLens-1M 全量 · 在 R0a 基线上逐组加特征、换统计口径 · 3-5 种子（已有 5 / 5 个运行）</summary>

| 场景 | 类型 | 验证 AUC | 验证 GAUC | bootstrap 标准差 | 验证 AUC：各种子 | 断言 |
| - | - | - | - | - | - | - |
| `baseline` | 基线 | 0.7449 ± 0.0015 | 0.7304 ± 0.0004 | 0.0055 ± 0.0001 | s0 0.7447 / s1 0.7437 / s2 0.7458 / s3 0.7469 / s4 0.7433 | 通过 |

![验证 AUC（均值，误差线为种子间最小到最大）](figures/h20x8-p_noise-metrics.svg)

运行目录（每个目录里有 `summary.json` 指标、`evidence.json` 证据摘要、`predictions.jsonl` 逐条预测、`log.txt` 日志）：

- `baseline` [s0](../../../results/h20x8/R0b/baseline/s0-461376/) · [s1](../../../results/h20x8/R0b/baseline/s1-71c585/) · [s2](../../../results/h20x8/R0b/baseline/s2-570ea7/) · [s3](../../../results/h20x8/R0b/baseline/s3-a5952b/) · [s4](../../../results/h20x8/R0b/baseline/s4-e7e664/)

</details>

## 看完实测再想一步

把种子切换成单个种子，各种子的柱子差多少？

---

[← 先估算](05-estimate.md) · [课程首页](../README.md) · [加用户画像 →](07-profile.md)
