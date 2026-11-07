<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [代码导读 →](03-code.md)

# 第 2 步 / 共 17 步：流程结构

> 阶段 1 / 7：**理解任务**

先做 SFT（和 L05 基线同配置），把 SFT 的 LoRA 合并进基础权重，再挂一个新 LoRA 做偏好阶段。**关掉新 LoRA 就是参考模型**，不需要第二份 7B 权重。点击每个环节看它的输入输出和对应代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. SFT | 4000 条 → LoRA r=16，2 轮 | 与 L05 基线同一套代码和参数；保存 sft_adapter。 | `adextract/train.py` |
| 2. 合并 + 新 LoRA | W ← W + (α/r)·BA；再挂 B=0 的新 LoRA | merge_and_unload 后 get_peft_model。新 LoRA 的 B=0：训练开始时策略 = 参考模型。 |  |
| 3. 偏好对 | (prompt, chosen, rejected) × 512 | rule：标准答案 vs 删条件 / 塞无依据卖点；onpolicy：标准答案 vs SFT 采样 G 条里最差的一条（全对的 prompt 不产生偏好对）。 | [`adpref/pairs.py#L71-L83`](../../../projects/adpref/pairs.py#L71-L83) |
| 4. 序列对数概率 | [N, L] 逐 token → 求和 [N] | 只在回答位置算 lm_head；交叉熵取负就是 log p(真实 token)。 | [`adpref/optim.py#L24-L35`](../../../projects/adpref/optim.py#L24-L35) |
| 5. DPO loss | −log σ(β·(Δc − Δr)) | 参考模型对数概率训练前一次算好（关掉新 LoRA 前向一遍）。 | [`adpref/optim.py#L46-L119`](../../../projects/adpref/optim.py#L46-L119) |
| 6. 在线采样 | 8 prompt × G=4 → 32 条回答 | 温度 1、关掉 top_k/top_p/重复惩罚：采样分布 = 模型分布。生成的 token 原样拿去算梯度。 | [`adpref/pairs.py#L27-L60`](../../../projects/adpref/pairs.py#L27-L60) |
| 7. 可验证奖励 | 回答 + 标准答案 → [0, 1] | field：品类、卖点 F1、优惠 F1、条件保留，无依据扣分；recall_only：只看召回。 | [`adpref/reward.py#L29-L43`](../../../projects/adpref/reward.py#L29-L43) |
| 8. GRPO loss | A = (r − 组均值)/组标准差 | −A·mean log π + β_KL·k3(π_ref‖π)；每批只更新一次，重要性比恒为 1。 | [`adpref/optim.py#L124-L200`](../../../projects/adpref/optim.py#L124-L200) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [代码导读 →](03-code.md)
