<!-- 本页由 python -m labkit docs 从 ../lesson.yaml 和 results/ 生成，不要手改 -->

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪 →](03-trace.md)

# 第 2 步 / 共 18 步：流程结构

> 阶段 1 / 7：**理解任务**

一次测量的流程。点击每个节点看代码。

| 环节 | 输入输出 / 形状 | 说明 | 代码 |
| - | - | - | - |
| 1. 构造请求 | 256 条 × 4096 token（规范 + 内容 + 指令） | 直接给 token ID，长度精确；共享规范放开头或结尾。 | [`vllmserve/prompts.py#L83-L98`](../../../projects/vllmserve/prompts.py#L83-L98) |
| 2. 屋顶线估算 | prefill 61 TFLOPs / decode 15.5 GB | 和讲义 S2 用同一个 roofline.py。 | [`vllmserve/experiment.py#L39-L57`](../../../projects/vllmserve/experiment.py#L39-L57) |
| 3. 独立环境起测量进程 | vllm-env 的解释器 + JSON 行协议 | 每个配置一个进程，取消和超时杀整个进程组。 | [`vllmserve/experiment.py#L100-L149`](../../../projects/vllmserve/experiment.py#L100-L149) |
| 4. 建引擎 | AsyncLLMEngine：max_num_seqs、token 预算、前缀缓存、量化、投机 | 引擎日志里报 KV cache 能放多少 token。 | [`vllmserve/bench_worker.py#L40-L58`](../../../projects/vllmserve/bench_worker.py#L40-L58) |
| 5. 按到达时刻发请求 | 流式输出：每次收到新 token 记一个时刻 | 贪心、固定输出 50 token。 | [`vllmserve/bench_worker.py#L83-L130`](../../../projects/vllmserve/bench_worker.py#L83-L130) |
| 6. TTFT / TPOT / ITL | 每条请求一个 TTFT、一个 TPOT；ITL 是相邻两次输出的间隔 |  | [`vllmserve/stats.py#L31-L57`](../../../projects/vllmserve/stats.py#L31-L57) |
| 7. 命中率、投机接受长度 | 每条请求的 num_cached_tokens；Prometheus 计数器 |  | [`vllmserve/stats.py#L94-L115`](../../../projects/vllmserve/stats.py#L94-L115) |

---

[← 场景与痛点](01-pain.md) · [课程首页](../README.md) · [追踪 →](03-trace.md)
