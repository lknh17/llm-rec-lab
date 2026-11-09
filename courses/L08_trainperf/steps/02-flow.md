<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪 →](03-trace.md)

# 第 2 步 / 共 19 步：流程结构

> 阶段 1 / 8：**理解任务**

一步训练和测量方式。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 启动 8 个 rank | rank0 在 labkit worker 里，rank 1-7 用 spawn | NCCL，超时 3 分钟；结束时 gather 各 rank 统计。 | [`trainperf/train.py#L264-L294`](../../../projects/trainperf/train.py#L264-L294) |
| 2. 包装 | none / DDP / FSDP2；重计算；compile | DDP 用 gradient_as_bucket_view 省一份梯度。 | [`trainperf/train.py#L38-L59`](../../../projects/trainperf/train.py#L38-L59) |
| 3. 取数据 | [mbs, T + 1] token | 默认直接在 GPU 上生成；故障场景走 CPU 分词。 | [`trainperf/data.py#L15-L17`](../../../projects/trainperf/data.py#L15-L17) |
| 4. 前向 → 反向 → 优化器 | logits [mbs, T, 151936] | 吞吐窗口不插同步；逐阶段窗口每段前后同步。 | [`trainperf/train.py#L142-L187`](../../../projects/trainperf/train.py#L142-L187) |
| 5. 显存快照 | 建模后 / 首个反向后 / 首次 step 后 / 峰值 | 差值就是参数、梯度、优化器状态、激活与其他。 | [`trainperf/measure.py#L13-L39`](../../../projects/trainperf/measure.py#L13-L39) |
| 6. MFU | 每卡 tokens/s × 每 token FLOPs ÷ 峰值 |  | [`trainperf/train.py#L299-L307`](../../../projects/trainperf/train.py#L299-L307) |
| 7. profiler | CUDA kernel 区间求并集 | GPU 忙闲、NCCL 时间、暴露的通信（NCCL 区间减去计算区间）。 | [`trainperf/measure.py#L44-L90`](../../../projects/trainperf/measure.py#L44-L90) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪 →](03-trace.md)
