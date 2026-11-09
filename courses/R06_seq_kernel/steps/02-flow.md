<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪 →](03-trace.md)

# 第 2 步 / 共 15 步：流程结构

> 阶段 1 / 6：**理解任务**

一次 attention 调用在四种实现里各走什么路径。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. naive | QKᵀ [B, H, L, L] → mask → softmax / SiLU → @ V | 每一步一个 kernel，中间张量都写回 HBM。 | [`recops/attn.py#L25-L37`](../../../projects/recops/attn.py#L25-L37) |
| 2. Triton 融合 | grid = (⌈L / BLOCK_M⌉, B·H)，每个 program 一块 Q | 循环读 K / V 块，打分、mask、激活、累加都在片上。 | [`recops/attn.py#L59-L104`](../../../projects/recops/attn.py#L59-L104) |
| 3. CPU 分块镜像 | 和 kernel 同样的循环，PyTorch 写 | 在没有 GPU 的地方验证分块算法和故障。 | [`recops/attn.py#L124-L165`](../../../projects/recops/attn.py#L124-L165) |
| 4. FLOPs 与字节 | 2·B·H·L²·D FLOPs；融合 4·B·H·L·D 个元素 | naive 还要加上 L × L 中间张量的读写。 | [`recops/attn.py#L170-L179`](../../../projects/recops/attn.py#L170-L179) |
| 5. roofline | AI = FLOPs / 字节 → min(峰值, AI × 带宽) | H20：148 TFLOPS、4.0 TB/s，拐点 37。 | [`recops/bench.py#L109-L114`](../../../projects/recops/bench.py#L109-L114) |
| 6. 计时 | 热身 → L2 冲刷 → CUDA event → 中位数 |  | [`recops/bench.py#L31-L71`](../../../projects/recops/bench.py#L31-L71) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪 →](03-trace.md)
