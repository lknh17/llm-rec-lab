# V01 讲义验证：讲义 GPU 验证清单在 H20 上的实测

私有仓库的讲义（S2-S7）里有一批"需 GPU"的实验，代码写好了、在 Mac 上验证过方法，但数字还是"预期 / 典型值"。
这里把它们原样搬到 H20 上跑一遍：[`projects/lessonbench/vendor/`](../../projects/lessonbench/vendor/) 是讲义代码的逐字节拷贝，
[`items.py`](../../projects/lessonbench/items.py) 列出每一项跑什么命令。推理服务相关的项（S5 vLLM 四组 benchmark）由
[L09](../L09_vllm_serving/) 覆盖，FSDP 由 [L08](../L08_trainperf/) 覆盖，不在这里重复。

## 怎么看

每个场景是一项验证。`summary.ok` 表示命令全部成功；没成功时 `summary.missing` 写着缺什么环境（nvcc、ncu、vLLM），
`log.txt` 里有完整输出。讲义脚本生成的报告（md / csv）全文在运行目录 `trace.json` 的 `outputs` 里。

## 步骤

<!-- labkit:path -->
| # | 阶段 | 步骤 | 这一步回答的问题 | 场景 | H20 结果 |
| - | - | - | - | - | - |
| 1 | 验证 | [验证结果](steps/01-results.md) |  | — | — |
<!-- /labkit:path -->

## 场景与结果

<!-- labkit:results -->
<details><summary>展开结果（先自己预测，再看）</summary>

**h20x8**：8 × NVIDIA H20，每项独占 1 张卡；讲义脚本原样运行（vendor/ 是私有仓库讲义代码的逐字节拷贝），--hw H20。主环境叠加 tabulate / matplotlib / ninja（uv run --with，不改 uv.lock），vLLM 相关的项用 vllm-env（已有 0 / 14 个运行）

_还没有结果：等待批量运行。_

</details>
<!-- /labkit:results -->

## 证据图

<!-- labkit:figures -->
<details><summary>展开证据图（先自己预测，再看）</summary>

- **h20x8**：还没有结果，图在结果推回后自动生成。

</details>
<!-- /labkit:figures -->

## 代码

<!-- labkit:code -->
| 代码段 | 位置 |
| - | - |
| `probe` | [projects/lessonbench/experiment.py#L43](../../projects/lessonbench/experiment.py#L43) |
| `run_step` | [projects/lessonbench/experiment.py#L115](../../projects/lessonbench/experiment.py#L115) |
| `items` | [projects/lessonbench/items.py#L14](../../projects/lessonbench/items.py#L14) |
| `naive` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L17](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L17) |
| `naive_scores` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L18](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L18) |
| `naive_mask` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L36](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L36) |
| `naive_softmax` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L50](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L50) |
| `sdpa` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L64](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L64) |
| `module` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L75](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L75) |
| `module_init` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L76](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L76) |
| `module_split` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L107](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L107) |
| `module_gqa` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L120](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L120) |
| `module_out` | [projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L133](../../projects/lessonbench/vendor/stages/s01_transformer/code/attention.py#L133) |
| `rmsnorm` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L63](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L63) |
| `rope` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L89](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L89) |
| `rope_table` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L90](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L90) |
| `rope_apply` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L111](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L111) |
| `causal_attention` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L130](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L130) |
| `causal_attn_fast` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L131](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L131) |
| `causal_attn_mask` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L152](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L152) |
| `kv_cache` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L166](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L166) |
| `kv_cache_alloc` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L167](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L167) |
| `kv_cache_update` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L187](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L187) |
| `swiglu_block` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L246](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L246) |
| `swiglu` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L247](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L247) |
| `block` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L271](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L271) |
| `gpt_forward` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L314](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L314) |
| `gpt_forward_pos` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L315](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L315) |
| `gpt_forward_mask` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L340](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L340) |
| `gpt_forward_layers` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L354](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L354) |
| `count_params` | [projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L380](../../projects/lessonbench/vendor/stages/s01_transformer/code/model.py#L380) |
| `sample_next` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L13](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L13) |
| `sample_temp_topk` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L14](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L14) |
| `sample_topp` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L38](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L38) |
| `sample_draw` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L57](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L57) |
| `generate` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L67](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L67) |
| `generate_prefill` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L68](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L68) |
| `generate_decode` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L94](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L94) |
| `beam_search` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L115](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L115) |
| `beam_signature` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L116](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L116) |
| `beam_prefill` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L135](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L135) |
| `beam_expand` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L156](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L156) |
| `beam_reorder` | [projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L176](../../projects/lessonbench/vendor/stages/s01_transformer/code/sampling.py#L176) |
| `data` | [projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L34](../../projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L34) |
| `data_load` | [projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L35](../../projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L35) |
| `data_batch` | [projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L62](../../projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L62) |
| `flops` | [projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L78](../../projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L78) |
| `train_loop` | [projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L117](../../projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L117) |
| `train_setup` | [projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L118](../../projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L118) |
| `train_eval` | [projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L141](../../projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L141) |
| `train_step` | [projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L155](../../projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L155) |
| `train_timing` | [projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L178](../../projects/lessonbench/vendor/stages/s01_transformer/code/train.py#L178) |
| `build` | [projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L33](../../projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L33) |
| `prefill_loop` | [projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L67](../../projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L67) |
| `decode_loop` | [projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L87](../../projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L87) |
| `decode_setup` | [projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L88](../../projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L88) |
| `decode_step` | [projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L105](../../projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L105) |
| `decode_metrics` | [projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L118](../../projects/lessonbench/vendor/stages/s02_perf/code/decode_bench.py#L118) |
| `ref` | [projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L18](../../projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L18) |
| `ref_outer` | [projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L19](../../projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L19) |
| `ref_scores` | [projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L46](../../projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L46) |
| `ref_update` | [projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L64](../../projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L64) |
| `hbm_model` | [projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L86](../../projects/lessonbench/vendor/stages/s03_kernels/code/flash_attention_ref.py#L86) |
| `version_spec` | [projects/lessonbench/vendor/stages/s03_kernels/code/matmul_model.py#L26](../../projects/lessonbench/vendor/stages/s03_kernels/code/matmul_model.py#L26) |
| `traffic_model` | [projects/lessonbench/vendor/stages/s03_kernels/code/matmul_model.py#L66](../../projects/lessonbench/vendor/stages/s03_kernels/code/matmul_model.py#L66) |
| `warp_sectors` | [projects/lessonbench/vendor/stages/s03_kernels/code/memory_access.py#L23](../../projects/lessonbench/vendor/stages/s03_kernels/code/memory_access.py#L23) |
| `strided_traffic` | [projects/lessonbench/vendor/stages/s03_kernels/code/memory_access.py#L41](../../projects/lessonbench/vendor/stages/s03_kernels/code/memory_access.py#L41) |
| `bank_conflict` | [projects/lessonbench/vendor/stages/s03_kernels/code/memory_access.py#L53](../../projects/lessonbench/vendor/stages/s03_kernels/code/memory_access.py#L53) |
| `mac_experiment` | [projects/lessonbench/vendor/stages/s03_kernels/code/memory_access.py#L85](../../projects/lessonbench/vendor/stages/s03_kernels/code/memory_access.py#L85) |
| `route` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L27](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L27) |
| `expert_ffn` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L42](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L42) |
| `naive` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L54](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L54) |
| `permute` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L67](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L67) |
| `grouped` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L85](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L85) |
| `align` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L109](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L109) |
| `fused_ref` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L138](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L138) |
| `padded_bmm` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L176](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L176) |
| `waste` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L198](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L198) |
| `a2a` | [projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L227](../../projects/lessonbench/vendor/stages/s03_kernels/code/moe_ref.py#L227) |
| `sdpa_decode` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L26](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L26) |
| `split_kv` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L40](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L40) |
| `split_kv_partial` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L41](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L41) |
| `split_kv_local` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L71](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L71) |
| `merge` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L91](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L91) |
| `paged` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L107](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L107) |
| `paged_build` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L108](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L108) |
| `paged_decode` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L141](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L141) |
| `gather` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L193](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L193) |
| `model` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L209](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L209) |
| `model_hw` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L210](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L210) |
| `model_body` | [projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L234](../../projects/lessonbench/vendor/stages/s03_kernels/code/split_kv_decode.py#L234) |
| `pack` | [projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L29](../../projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L29) |
| `ref_gemm` | [projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L62](../../projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L62) |
| `blocked_gemm` | [projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L80](../../projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L80) |
| `model` | [projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L107](../../projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L107) |
| `fp8` | [projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L133](../../projects/lessonbench/vendor/stages/s03_kernels/code/w4a16_ref.py#L133) |
| `bench` | [projects/lessonbench/vendor/stages/s03_kernels/cuda/bench_matmul.py#L24](../../projects/lessonbench/vendor/stages/s03_kernels/cuda/bench_matmul.py#L24) |
| `load_inline` | [projects/lessonbench/vendor/stages/s03_kernels/cuda/matmul_ext.py#L18](../../projects/lessonbench/vendor/stages/s03_kernels/cuda/matmul_ext.py#L18) |
| `metrics` | [projects/lessonbench/vendor/stages/s03_kernels/cuda/ncu_profile.py#L22](../../projects/lessonbench/vendor/stages/s03_kernels/cuda/ncu_profile.py#L22) |
| `ncu_command` | [projects/lessonbench/vendor/stages/s03_kernels/cuda/ncu_profile.py#L66](../../projects/lessonbench/vendor/stages/s03_kernels/cuda/ncu_profile.py#L66) |
| `parse` | [projects/lessonbench/vendor/stages/s03_kernels/cuda/ncu_profile.py#L109](../../projects/lessonbench/vendor/stages/s03_kernels/cuda/ncu_profile.py#L109) |
| `diagnose` | [projects/lessonbench/vendor/stages/s03_kernels/cuda/ncu_profile.py#L148](../../projects/lessonbench/vendor/stages/s03_kernels/cuda/ncu_profile.py#L148) |
| `verify` | [projects/lessonbench/vendor/stages/s03_kernels/tests/test_flash_ref.py#L8](../../projects/lessonbench/vendor/stages/s03_kernels/tests/test_flash_ref.py#L8) |
| `verify` | [projects/lessonbench/vendor/stages/s03_kernels/tests/test_moe.py#L28](../../projects/lessonbench/vendor/stages/s03_kernels/tests/test_moe.py#L28) |
| `verify` | [projects/lessonbench/vendor/stages/s03_kernels/tests/test_split_kv.py#L23](../../projects/lessonbench/vendor/stages/s03_kernels/tests/test_split_kv.py#L23) |
| `verify` | [projects/lessonbench/vendor/stages/s03_kernels/tests/test_w4a16.py#L18](../../projects/lessonbench/vendor/stages/s03_kernels/tests/test_w4a16.py#L18) |
| `strided_kernel` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/coalescing.py#L28](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/coalescing.py#L28) |
| `gather_kernel` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/coalescing.py#L57](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/coalescing.py#L57) |
| `flash_signature` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L13](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L13) |
| `kernel_body` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L30](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L30) |
| `flash_load_q` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L31](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L31) |
| `flash_init` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L53](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L53) |
| `flash_scores` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L68](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L68) |
| `flash_update` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L91](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L91) |
| `flash_store` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L109](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L109) |
| `flash_launch` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L119](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/flash_attn.py#L119) |
| `kernel` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/fused_softmax.py#L11](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/fused_softmax.py#L11) |
| `softmax_load` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/fused_softmax.py#L19](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/fused_softmax.py#L19) |
| `softmax_compute` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/fused_softmax.py#L36](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/fused_softmax.py#L36) |
| `launch` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/fused_softmax.py#L53](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/fused_softmax.py#L53) |
| `autotune` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L14](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L14) |
| `matmul_configs` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L15](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L15) |
| `matmul_group` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L39](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L39) |
| `kloop` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L61](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L61) |
| `matmul_ptrs` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L62](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L62) |
| `matmul_loop` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L77](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L77) |
| `matmul_store` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L97](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L97) |
| `matmul_launch` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L109](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/matmul.py#L109) |
| `pid` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L27](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L27) |
| `gather` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L40](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L40) |
| `kloop` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L58](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L58) |
| `epilogue` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L70](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L70) |
| `launch` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L82](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/moe_grouped_gemm.py#L82) |
| `partial_setup` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L30](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L30) |
| `partial_loop` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L64](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L64) |
| `partial_store` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L93](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L93) |
| `reduce` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L114](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L114) |
| `launch` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L134](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/split_kv_decode.py#L134) |
| `kernel` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/vector_add.py#L13](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/vector_add.py#L13) |
| `add_kernel` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/vector_add.py#L14](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/vector_add.py#L14) |
| `add_launch` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/vector_add.py#L41](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/vector_add.py#L41) |
| `autotune` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/w4a16_gemm.py#L19](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/w4a16_gemm.py#L19) |
| `setup` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/w4a16_gemm.py#L38](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/w4a16_gemm.py#L38) |
| `mainloop` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/w4a16_gemm.py#L55](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/w4a16_gemm.py#L55) |
| `launch` | [projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/w4a16_gemm.py#L82](../../projects/lessonbench/vendor/stages/s03_kernels/triton_kernels/w4a16_gemm.py#L82) |
| `alloc_release` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L35](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L35) |
| `alloc` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L36](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L36) |
| `release` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L57](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L57) |
| `prefix_cache` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L72](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L72) |
| `prefix_hash` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L73](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L73) |
| `prefix_lookup` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L82](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L82) |
| `prefix_hit` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L103](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L103) |
| `prefix_register` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L123](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L123) |
| `ensure_capacity` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L148](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L148) |
| `capacity_check` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L149](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L149) |
| `capacity_cow` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L170](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L170) |
| `fork_free` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L194](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L194) |
| `fork_share` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L195](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L195) |
| `free_release` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L210](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/block_manager.py#L210) |
| `slots` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L29](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L29) |
| `slots_map` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L30](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L30) |
| `slots_copy` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L40](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L40) |
| `run_inputs` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L53](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L53) |
| `inputs_gather` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L54](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L54) |
| `inputs_tensors` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L77](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L77) |
| `run_layers` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L93](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L93) |
| `layers_kv_write` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L94](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L94) |
| `layers_attn` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L105](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L105) |
| `layers_logits` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L127](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/model_runner.py#L127) |
| `sequence` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/sequence.py#L6](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/sequence.py#L6) |
| `sequence_fields` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/sequence.py#L7](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/sequence.py#L7) |
| `sequence_props` | [projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/sequence.py#L35](../../projects/lessonbench/vendor/stages/s05_inference/code/mini_engine/sequence.py#L35) |
| `quantize` | [projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L24](../../projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L24) |
| `quantize_fn` | [projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L25](../../projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L25) |
| `dequantize_fn` | [projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L55](../../projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L55) |
| `quant_linear` | [projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L67](../../projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L67) |
| `quant_linear_init` | [projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L68](../../projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L68) |
| `quant_linear_forward` | [projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L84](../../projects/lessonbench/vendor/stages/s05_inference/code/quant.py#L84) |
| `item_emb` | [projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L72](../../projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L72) |
| `item_emb_cooc` | [projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L73](../../projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L73) |
| `item_emb_ppmi` | [projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L95](../../projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L95) |
| `item_emb_svd` | [projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L109](../../projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L109) |
| `make_batch` | [projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L128](../../projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L128) |
| `make_batch_window` | [projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L129](../../projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L129) |
| `make_batch_tensor` | [projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L152](../../projects/lessonbench/vendor/stages/s06_recsys/code/data.py#L152) |
| `hstu_layer` | [projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L22](../../projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L22) |
| `hstu_init` | [projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L23](../../projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L23) |
| `hstu_proj` | [projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L42](../../projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L42) |
| `hstu_rab` | [projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L54](../../projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L54) |
| `hstu_attn` | [projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L65](../../projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L65) |
| `hstu_out` | [projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L80](../../projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L80) |
| `block_params` | [projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L92](../../projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L92) |
| `hstu_rec` | [projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L93](../../projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L93) |
| `block_params_fn` | [projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L121](../../projects/lessonbench/vendor/stages/s06_recsys/code/hstu.py#L121) |
| `kmeans` | [projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L20](../../projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L20) |
| `kmeans_init` | [projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L21](../../projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L21) |
| `kmeans_lloyd` | [projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L39](../../projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L39) |
| `rq_kmeans` | [projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L82](../../projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L82) |
| `rq_levels` | [projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L83](../../projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L83) |
| `rq_dedup` | [projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L106](../../projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L106) |
| `tokenizer` | [projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L124](../../projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L124) |
| `tokenizer_vocab` | [projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L125](../../projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L125) |
| `tokenizer_encode` | [projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L147](../../projects/lessonbench/vendor/stages/s06_recsys/code/rq_kmeans.py#L147) |
| `eval` | [projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L51](../../projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L51) |
| `eval_id` | [projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L52](../../projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L52) |
| `eval_tiger` | [projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L81](../../projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L81) |
| `tiger_batch` | [projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L129](../../projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L129) |
| `train_tiger` | [projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L152](../../projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L152) |
| `train_tiger_setup` | [projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L153](../../projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L153) |
| `train_tiger_loop` | [projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L176](../../projects/lessonbench/vendor/stages/s06_recsys/code/train_genrec.py#L176) |
| `trie` | [projects/lessonbench/vendor/stages/s06_recsys/code/trie.py#L12](../../projects/lessonbench/vendor/stages/s06_recsys/code/trie.py#L12) |
| `trie_build` | [projects/lessonbench/vendor/stages/s06_recsys/code/trie.py#L13](../../projects/lessonbench/vendor/stages/s06_recsys/code/trie.py#L13) |
| `trie_allowed` | [projects/lessonbench/vendor/stages/s06_recsys/code/trie.py#L39](../../projects/lessonbench/vendor/stages/s06_recsys/code/trie.py#L39) |
| `decode_slots` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L79](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L79) |
| `decode_cow` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L80](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L80) |
| `decode_write_slots` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L105](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L105) |
| `decode_read_slots` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L117](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L117) |
| `decode_layers` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L142](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L142) |
| `decode_qkv` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L143](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L143) |
| `decode_cascade` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L162](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L162) |
| `decode_out` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L189](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L189) |
| `prefill_fork` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L201](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L201) |
| `prefill_once` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L202](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L202) |
| `fork_first` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L229](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L229) |
| `beam_step` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L251](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L251) |
| `beam_select` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L252](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L252) |
| `beam_fork` | [projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L268](../../projects/lessonbench/vendor/stages/s07_capstone/code/beam_engine.py#L268) |
| `cascade` | [projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L21](../../projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L21) |
| `cascade_attend` | [projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L22](../../projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L22) |
| `cascade_private` | [projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L39](../../projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L39) |
| `cascade_shared` | [projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L71](../../projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L71) |
| `cascade_merge` | [projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L87](../../projects/lessonbench/vendor/stages/s07_capstone/code/cascade_attention.py#L87) |
| `build` | [projects/lessonbench/vendor/stages/s07_capstone/code/trie_gpu.py#L25](../../projects/lessonbench/vendor/stages/s07_capstone/code/trie_gpu.py#L25) |
| `ops` | [projects/lessonbench/vendor/stages/s07_capstone/code/trie_gpu.py#L53](../../projects/lessonbench/vendor/stages/s07_capstone/code/trie_gpu.py#L53) |
| `frontend_beam` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/constrained_beam.py#L28](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/constrained_beam.py#L28) |
| `frontend_reqs` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/constrained_beam.py#L29](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/constrained_beam.py#L29) |
| `frontend_expand` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/constrained_beam.py#L49](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/constrained_beam.py#L49) |
| `llm_init` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/constrained_beam.py#L98](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/constrained_beam.py#L98) |
| `rename` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/export_hf.py#L26](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/export_hf.py#L26) |
| `rename_layers` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/export_hf.py#L27](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/export_hf.py#L27) |
| `rename_tie` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/export_hf.py#L49](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/export_hf.py#L49) |
| `update_state` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L52](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L52) |
| `update_added` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L53](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L53) |
| `update_moved` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L76](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L76) |
| `apply` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L93](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L93) |
| `apply_nodes` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L94](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L94) |
| `apply_mask` | [projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L110](../../projects/lessonbench/vendor/stages/s07_capstone/code/vllm_genrec/trie_logits_processor.py#L110) |
| `bench_result` | [projects/lessonbench/vendor/tools/bench.py#L19](../../projects/lessonbench/vendor/tools/bench.py#L19) |
| `bench_warmup` | [projects/lessonbench/vendor/tools/bench.py#L36](../../projects/lessonbench/vendor/tools/bench.py#L36) |
| `bench_measure` | [projects/lessonbench/vendor/tools/bench.py#L47](../../projects/lessonbench/vendor/tools/bench.py#L47) |
| `bench_stats` | [projects/lessonbench/vendor/tools/bench.py#L64](../../projects/lessonbench/vendor/tools/bench.py#L64) |
| `calibrate` | [projects/lessonbench/vendor/tools/gpu_check.py#L40](../../projects/lessonbench/vendor/tools/gpu_check.py#L40) |
| `calibrate_matmul` | [projects/lessonbench/vendor/tools/gpu_check.py#L41](../../projects/lessonbench/vendor/tools/gpu_check.py#L41) |
| `calibrate_bandwidth` | [projects/lessonbench/vendor/tools/gpu_check.py#L56](../../projects/lessonbench/vendor/tools/gpu_check.py#L56) |
| `gpu_spec` | [projects/lessonbench/vendor/tools/occupancy.py#L20](../../projects/lessonbench/vendor/tools/occupancy.py#L20) |
| `occupancy` | [projects/lessonbench/vendor/tools/occupancy.py#L91](../../projects/lessonbench/vendor/tools/occupancy.py#L91) |
| `occupancy_regs` | [projects/lessonbench/vendor/tools/occupancy.py#L109](../../projects/lessonbench/vendor/tools/occupancy.py#L109) |
| `occupancy_smem` | [projects/lessonbench/vendor/tools/occupancy.py#L123](../../projects/lessonbench/vendor/tools/occupancy.py#L123) |
| `hardware` | [projects/lessonbench/vendor/tools/roofline.py#L16](../../projects/lessonbench/vendor/tools/roofline.py#L16) |
| `hardware_class` | [projects/lessonbench/vendor/tools/roofline.py#L17](../../projects/lessonbench/vendor/tools/roofline.py#L17) |
| `hardware_table` | [projects/lessonbench/vendor/tools/roofline.py#L42](../../projects/lessonbench/vendor/tools/roofline.py#L42) |
| `estimate` | [projects/lessonbench/vendor/tools/roofline.py#L66](../../projects/lessonbench/vendor/tools/roofline.py#L66) |
| `estimate_inputs` | [projects/lessonbench/vendor/tools/roofline.py#L67](../../projects/lessonbench/vendor/tools/roofline.py#L67) |
| `estimate_times` | [projects/lessonbench/vendor/tools/roofline.py#L84](../../projects/lessonbench/vendor/tools/roofline.py#L84) |
| `matmul` | [projects/lessonbench/vendor/tools/roofline.py#L120](../../projects/lessonbench/vendor/tools/roofline.py#L120) |
| `model_shape` | [projects/lessonbench/vendor/tools/roofline.py#L162](../../projects/lessonbench/vendor/tools/roofline.py#L162) |
| `prefill_decode` | [projects/lessonbench/vendor/tools/roofline.py#L197](../../projects/lessonbench/vendor/tools/roofline.py#L197) |
| `prefill_fn` | [projects/lessonbench/vendor/tools/roofline.py#L198](../../projects/lessonbench/vendor/tools/roofline.py#L198) |
| `decode_fn` | [projects/lessonbench/vendor/tools/roofline.py#L218](../../projects/lessonbench/vendor/tools/roofline.py#L218) |
| `max_batch` | [projects/lessonbench/vendor/tools/roofline.py#L234](../../projects/lessonbench/vendor/tools/roofline.py#L234) |
<!-- /labkit:code -->
