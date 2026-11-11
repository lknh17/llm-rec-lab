"""在分页 KV 上执行一个批次。对应 vLLM 的 GPUModelRunner + PagedAttention kernel。

一个批次里混着不同序列的不同数量的新 token（有的在 prefill 一个 chunk，有的在 decode 1 个 token）：
- 所有新 token 被拼成一个"扁平"的 [1, N] 输入（varlen）：线性层（QKV、O、MLP、lm_head）对 N 个 token 一次算完，
  权重只读一遍 —— 这就是 continuous batching 提升吞吐的来源。
- attention 每个序列各算各的：按 block_table 从物理块里取出它的历史 K/V。
  这里用 Python 循环 + gather 写清楚逻辑；真实引擎用一个 paged attention kernel 直接按块表读取，没有 gather 拷贝。
"""

import torch

from stages.s01_transformer.code.model import GPT, causal_attention
from stages.s05_inference.code.mini_engine.sequence import Sequence


class ModelRunner:
    def __init__(self, model: GPT, num_blocks: int, block_size: int):
        p = next(model.parameters())
        cfg = model.cfg
        self.model, self.block_size = model, block_size
        # 物理 KV 池：[L, 2(K/V), num_blocks·block_size 个槽位, Hkv, D]
        self.kv = torch.zeros(cfg.n_layer, 2, num_blocks * block_size, cfg.n_kv_head, cfg.head_dim,
                              device=p.device, dtype=p.dtype)
        self.device = p.device

    def kv_bytes(self) -> int:
        return self.kv.numel() * self.kv.element_size()

    # --8<-- [start:slots]
    # --8<-- [start:slots_map]
    def _slots(self, seq: Sequence, start: int, end: int) -> list[int]:
        """逻辑位置 [start, end) -> 物理槽位：block_table[pos // bs] * bs + pos % bs"""
        bs = self.block_size
        # p // bs 是逻辑块号，查块表得到物理块号；乘 bs 得到这个物理块的首槽位，再加块内偏移 p % bs
        # 例：bs=16、block_table=[7, 2]，p=20 -> 逻辑块 1 -> 物理块 2 -> 槽位 2·16 + 4 = 36
        # 返回长度 end - start 的 Python 列表；纯 CPU 循环，10K token 就是 10K 次
        return [seq.block_table[p // bs] * bs + p % bs for p in range(start, end)]
    # --8<-- [end:slots_map]

    # --8<-- [start:slots_copy]
    # @torch.no_grad()：函数内不记录计算图、不为反向传播保存中间结果，推理时省显存也省时间
    @torch.no_grad()
    def apply_copies(self, copies: list[tuple[int, int]]) -> None:
        """执行写时复制：把物理块 src 的 KV 拷到 dst。"""
        bs = self.block_size
        for src, dst in copies:
            # 物理块 b 占槽位 [b·bs, (b+1)·bs)；前两维写 ":" 表示所有层、K 和 V 一起拷
            # 左边是池子的切片视图，赋值即原地写入；一次搬 [L, 2, bs, Hkv, D]
            self.kv[:, :, dst * bs:(dst + 1) * bs] = self.kv[:, :, src * bs:(src + 1) * bs]
    # --8<-- [end:slots_copy]
    # --8<-- [end:slots]

    # --8<-- [start:run_inputs]
    # --8<-- [start:inputs_gather]
    @torch.no_grad()
    def run(self, batch: list[tuple[Sequence, int]]) -> list[torch.Tensor | None]:
        """batch: [(seq, 本步计算的 token 数)]。返回每个序列最后一个位置的 logits [V]（prefill 未完成的返回 None）。"""
        m = self.model
        # 逐序列拼接的五个 Python 列表：tokens、positions、write_slots 最终长度都是 N（本步新 token 总数）
        # ctx_slots 每个序列一个张量，spans 记每个序列本步的 n，后面按它把扁平的 N 切回各序列
        tokens, positions, write_slots, ctx_slots, spans = [], [], [], [], []
        for seq, n in batch:
            # 本步从第 num_computed 个 token 开始算（prefix 命中的、前几个 chunk 都已经算过）
            start = seq.num_computed
            # 本步要喂的 n 个 token：prefill 是一段 prompt，decode 是刚采样的 1 个
            tokens += seq.tokens[start:start + n]
            # 绝对位置 start ... start+n-1，给 RoPE 用；从 0 开始数的话第 2 个 chunk 会被当成句首
            positions += range(start, start + n)
            # 新 token 的 KV 写到哪些物理槽位（vLLM 叫 slot_mapping），长度 n
            write_slots += self._slots(seq, start, start + n)
            # 这个序列做 attention 要读的全部槽位：历史 + 本步新 token，长度 S = start + n
            # 每个序列建一个设备张量，batch=16 就是 16 次小的 CPU -> 设备拷贝
            ctx_slots.append(torch.tensor(self._slots(seq, 0, start + n), device=self.device))
            spans.append(n)
    # --8<-- [end:inputs_gather]

    # --8<-- [start:inputs_tensors]
        # 所有序列的新 token 首尾相接、没有 padding；[None] 在最前面加一维 batch=1
        idx = torch.tensor(tokens, device=self.device)[None]  # [1, N]
        # 每个新 token 的绝对位置
        pos = torch.tensor(positions, device=self.device)  # [N]
        # 本步 KV 的写入槽位
        w_slots = torch.tensor(write_slots, device=self.device)  # [N]
        # 用位置当下标从预计算的 RoPE 表取行：rope_cos 是 [S_max, D]，取出 [N, D]
        # 每个 token 用自己的绝对位置：同一行里挨着的两个 token 可能分属两个序列，位置互不相干
        cos, sin = m.rope_cos[pos], m.rope_sin[pos]  # [N, D]
        # 词嵌入查表：[V, C] 的表按 idx 取行
        x = m.embed(idx)  # [1, N, C]
        n_total = idx.size(1)
    # --8<-- [end:inputs_tensors]
        # --8<-- [end:run_inputs]

        # --8<-- [start:run_layers]
        # --8<-- [start:layers_kv_write]
        # 逐层前向：每层是 attention 子层 + MLP 子层，都是 pre-norm 残差
        for li, block in enumerate(m.blocks):
            # 复用 S1 模型的投影 + 拆头 + RoPE：N 个 token 一次矩阵乘，权重只读一遍
            # q [1, H, N, D]，k/v [1, Hkv, N, D]
            q, k, v = block.attn.qkv(block.attn_norm(x), cos, sin)
            # 写入分页缓存：k[0] 去掉 batch 维得 [Hkv, N, D]，transpose 成 [N, Hkv, D] 和池子的槽位维对齐
            # self.kv[li, 0, w_slots] 用整数张量做下标 = 散写（scatter）：第 i 个 token 写到槽位 w_slots[i]
            self.kv[li, 0, w_slots] = k[0].transpose(0, 1)  # [N, Hkv, D]
            self.kv[li, 1, w_slots] = v[0].transpose(0, 1)  # [N, Hkv, D]
        # --8<-- [end:layers_kv_write]
            # --8<-- [start:layers_attn]
            # attention 只能逐序列算：每个序列只看自己的历史，不能像线性层那样 N 个 token 一起算
            # off 是当前序列的新 token 在扁平 N 维里的起点
            outs, off = [], 0
            for n, slots in zip(spans, ctx_slots):
                # 按块表 gather 历史：用槽位下标从池子取出 [S, Hkv, D]，转成 [Hkv, S, D] 再加 batch 维
                # gather 会把历史 KV 拷一份：Llama-3-8B、S=10K 时每层每序列 K+V 约 40 MB
                kk = self.kv[li, 0, slots].transpose(0, 1)[None]  # [1, Hkv, S, D]
                vv = self.kv[li, 1, slots].transpose(0, 1)[None]  # [1, Hkv, S, D]
                # q 只取本序列的 n 个新 token [1, H, n, D]，k/v 有 S 个
                # causal_attention 用右下对齐的 mask：第 j 个新 token（从 0 数）能看到前 S - n + j + 1 个位置
                # GQA（H != Hkv）由它内部的 SDPA enable_gqa 处理，不需要复制 K/V
                outs.append(causal_attention(q[:, :, off:off + n], kk, vv))  # [1, H, n, D]
                off += n
            # 沿 token 维（dim=2）拼回 N，再合并多头：[1, H, N, D] -> [1, N, H, D] -> [1, N, C]
            # reshape 遇到 transpose 后不连续的内存会自动复制（view 会报错）
            y = torch.cat(outs, dim=2).transpose(1, 2).reshape(1, n_total, -1)  # [1, N, C]
            # 输出投影 + 残差；MLP 子层同样对 N 个 token 一起算
            x = x + block.attn.wo(y)  # [1, N, C]
            x = x + block.mlp(block.mlp_norm(x))  # [1, N, C]
            # --8<-- [end:layers_attn]

        # --8<-- [start:layers_logits]
        # 只对"本步之后 prompt 已全部算完"的序列取最后一个位置算 logits（lm_head 很大，不要对所有 token 都算）
        # last：要取 logits 的扁平位置；owners：它们分别属于 batch 里的第几个序列
        last, owners, off = [], [], 0
        for i, (seq, n) in enumerate(batch):
            # 累加后 off 指向本序列最后一个新 token 的下一个位置
            off += n
            # 算完这 n 个就追上了全部 token：prefill 的最后一个 chunk，或者 decode
            # 没追上（chunked prefill 的中间 chunk）就不算 logits，engine 收到 None
            if seq.num_computed + n == seq.num_tokens:
                last.append(off - 1)
                owners.append(i)
        # 默认全是 None，只有 owners 对应的位置会被填上
        result: list[torch.Tensor | None] = [None] * len(batch)
        if last:
            # x[0, last] 用下标列表一次取出 n_last 行 [n_last, C]，只对它们过 norm 和 lm_head
            # 省了多少：prefill 512 token、V=128K、C=4096 时全算要 2·512·4096·128K ≈ 0.54 TFLOP，这里只算 1 行
            logits = m.lm_head(m.norm(x[0, last]))  # [n_last, V]
            # 按行拆回各序列，每个 lg 是 [V]
            for i, lg in zip(owners, logits):
                result[i] = lg
        return result
        # --8<-- [end:layers_logits]
        # --8<-- [end:run_layers]
