"""生成式推荐专用的分页 beam search 引擎（毕业项目核心）。

在 S5 mini engine 的组件上改造：
1. 用户历史只 prefill 一次（S1 的 beam_search 把 prompt 复制 W 份再算，计算量 ×W）；
2. W 个 beam 通过块引用计数共享历史 KV：fork 只复制块表，写时复制只发生在没写满的最后一块；
3. decode 用 cascade attention：共享前缀的 KV 每个请求只读一次，beam 私有部分各读各的；
4. trie 约束完全在 GPU 上（TensorTrie 稠密转移表）；
5. 跨请求的历史缓存：同一用户的下一次请求（历史只追加了几个行为）通过 prefix caching 命中已有的块，只 prefill 增量。

对照 S1 `beam_search`（连续 KV）：每个 beam 一份完整历史，每步按来源整体重排 —— 拷贝 W·P 个 token 的 KV。

    engine = PagedBeamSearch(model, trie, num_blocks=4096, block_size=16)
    results = engine.search([history_tokens_user1, history_tokens_user2], beam_width=20, steps=4)
"""

from dataclasses import dataclass, field

import torch

from stages.s05_inference.code.mini_engine.block_manager import BlockManager
from stages.s05_inference.code.mini_engine.model_runner import ModelRunner
from stages.s05_inference.code.mini_engine.sequence import Sequence
from stages.s07_capstone.code.cascade_attention import cascade_attention
from stages.s07_capstone.code.trie_gpu import TensorTrie


@dataclass
class BeamGroup:
    """一个请求（一个用户）的 W 个 beam。"""
    prompt_len: int
    beams: list[Sequence]
    scores: torch.Tensor  # [W]
    nodes: torch.Tensor  # [W] 每个 beam 当前所在的 trie 节点
    shared_len: int = 0  # 所有 beam 共享的 token 数 = 历史中写满的块；没写满的最后一块会被写时复制，成为各 beam 私有
    shared_slots: torch.Tensor | None = None  # [shared_len] 共享区间的物理槽位（整个搜索过程中不变）


@dataclass
class SearchStats:
    prefill_tokens: int = 0
    prefix_hit_tokens: int = 0
    cow_copies: int = 0
    peak_blocks: int = 0
    kv_read_tokens: int = 0  # decode 阶段 attention 读取的 KV token 数（衡量带宽）
    extra: dict = field(default_factory=dict)


class PagedBeamSearch:
    def __init__(self, model, trie: TensorTrie | None, num_blocks: int = 4096, block_size: int = 16,
                 enable_prefix_cache: bool = True, renormalize: bool = False):
        """renormalize=False：得分 = 全词表 log_softmax 后再屏蔽非法 token（与 S1 beam_search 一致）；
        True：先屏蔽再 log_softmax，即"在合法物品中的条件概率"（与 vLLM 的 processed_logprobs 一致）。"""
        self.model = model.eval()
        self.trie = trie
        self.renormalize = renormalize
        self.bm = BlockManager(num_blocks, block_size, enable_prefix_cache)
        self.runner = ModelRunner(model, num_blocks, block_size)
        self.bs = block_size
        self._next_id = 0
        self.device = next(model.parameters()).device

    def _new_seq(self, prompt: list[int]) -> Sequence:
        self._next_id += 1
        return Sequence(self._next_id, list(prompt), max_new_tokens=1 << 30)

    def _fork(self, parent: Sequence, token: int) -> Sequence:
        child = self._new_seq(parent.prompt)
        child.output = parent.output + [token]
        self.bm.fork(parent, child)
        return child

    def _mask(self, logp: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        if self.trie is None:
            return logp
        logp = logp.masked_fill(~self.trie.allowed(nodes), float("-inf"))
        return logp.log_softmax(-1) if self.renormalize else logp

    # ---------------------------------------------------------------- decode：一次前向所有请求的所有 beam
    # --8<-- [start:decode_slots]
    # --8<-- [start:decode_cow]
    # @torch.no_grad()：函数里的运算不记录计算图，推理时省掉为反向传播保存的中间激活
    @torch.no_grad()
    def _decode(self, groups: list[BeamGroup], stats: SearchStats) -> torch.Tensor:
        """每个 beam 计算 1 个 token（它最后选中的 token），返回 logits [R, W, V]。"""
        # kv 是 S5 ModelRunner 的整个 KV 池：[L, 2, num_blocks·bs, Hkv, D]，第 1 维 0 是 K、1 是 V
        # 第 2 维叫"槽位"（slot）：物理块 b 里的第 i 个 token 存在槽位 b·bs + i
        m, bs, kv = self.model, self.bs, self.runner.kv
        # R 个请求 × W 个 beam 拍平成一个列表，N = R·W，后面一起前向
        beams = [b for g in groups for b in g.beams]
        copies = []
        # 先保证每个 beam 本步要写的位置（第 num_computed 个 token）落在一个私有块里
        # ensure_capacity 处理两种情况：当前块写满了 -> 追加新块；要写的块 ref>1（fork 来的）-> 写时复制
        # 它只改块表，返回要执行的 [(src, dst)] 复制列表，还没有搬任何 KV
        for b in beams:
            c = self.bm.ensure_capacity(b, b.num_computed + 1)
            if c is None:
                raise RuntimeError("KV 块不够：增大 num_blocks")
            copies += c
        # 集中执行块复制：每一对把 L 层 × K/V × bs 个 token 从 src 块拷到 dst 块
        # 一块的大小：Mac 小模型 16 × 2 KB = 32 KB；genrec_1b 16 × 192 KB = 3 MB
        self.runner.apply_copies(copies)
        stats.cow_copies += len(copies)
    # --8<-- [end:decode_cow]

    # --8<-- [start:decode_write_slots]
        # 逻辑位置 p -> 物理槽位：查块表得到第 p // bs 块的物理块号，乘 bs 再加块内偏移 p % bs
        # vLLM 的 slot_mapping 用的是同一个公式，只是在 GPU 张量上批量算
        slot = lambda s, p: s.block_table[p // bs] * bs + p % bs
        # 每个 beam 本步的输入 token：位置 num_computed 上的 token，就是它上一步刚选中的那个
        tokens = torch.tensor([b.tokens[b.num_computed] for b in beams], device=self.device)  # [N]
        # 新 token 的位置，用来查 RoPE 的 cos/sin
        pos = torch.tensor([b.num_computed for b in beams], device=self.device)  # [N]
        # 新 token 的 K/V 写到哪个槽位；上面刚做完写时复制，这个槽位一定在私有块里
        w_slots = torch.tensor([slot(b, b.num_computed) for b in beams], device=self.device)  # [N]
    # --8<-- [end:decode_write_slots]

    # --8<-- [start:decode_read_slots]
        # 每个请求：共享前缀的槽位 [P_shared]，每个 beam 私有部分的槽位 [W, S]
        shared_slots, priv_slots = [], []
        for g in groups:
            # 共享区间在所有 beam 的块表里是同一组物理块，看第 0 个 beam 的块表就够了
            first = g.beams[0]
            # 只在第一次 decode 时算：共享块只读、永远不会被写，所以槽位在整个搜索里不变
            if g.shared_slots is None:
                # 前 shared_len // bs 个物理块号；10K 历史、bs=16 时是 625 个
                blocks = torch.tensor(first.block_table[: g.shared_len // bs], device=self.device)  # [nb]
                # blocks[:, None] 是 [nb, 1]，arange(bs) 是 [bs]，广播相加得到 [nb, bs]：每块展开成 bs 个连续槽位
                # flatten 拍平成 [nb·bs] = [P_shared]；10K 历史时是 1 万个 int64，80 KB
                g.shared_slots = (blocks[:, None] * bs + torch.arange(bs, device=self.device)).flatten()
            shared_slots.append(g.shared_slots)
            # 本步算完后的序列长度；同一请求的 W 个 beam 走的步数相同，长度一致
            end = first.num_computed + 1
            # 私有区间 [shared_len, end)：历史最后没写满的那段 + 已生成的 token + 当前 token
            # 双重列表推导在 Python 里逐 beam、逐 token 算槽位，共 W·S 次：这是教学版的主要 CPU 开销
            priv_slots.append(torch.tensor([[slot(b, p) for p in range(g.shared_len, end)] for b in g.beams],
                                           device=self.device))  # [W, S]
            # 读 KV 的计数：共享段每个请求只读 1 次（P），私有段每个 beam 各读 S 个，合计 P + W·S
            stats.kv_read_tokens += g.shared_len + len(g.beams) * (end - g.shared_len)
    # --8<-- [end:decode_read_slots]
        # --8<-- [end:decode_slots]

        # --8<-- [start:decode_layers]
        # --8<-- [start:decode_qkv]
        # tokens[None]：[N] -> [1, N]，把 N 个 beam 当成"一条长度为 N 的序列"送进模型
        # 好处：线性层对 N 行做一次矩阵乘，权重每步只读一遍，被所有用户、所有 beam 摊薄
        x = m.embed(tokens[None])  # [1, N, C]
        # rope_cos 是 register_buffer 存的 [S_max, D] 表；用 pos 做行索引，每个 beam 取自己位置那一行
        cos, sin = m.rope_cos[pos], m.rope_sin[pos]  # [N, D]
        n = len(beams)
        for li, block in enumerate(m.blocks):
            # 复用 S1 Attention.qkv：RMSNorm 后投影、拆头、加 RoPE，只算新 token 的 Q/K/V
            # 返回 q [1, H, N, D]，k/v [1, Hkv, N, D]
            q, k, v = block.attn.qkv(block.attn_norm(x), cos, sin)
            # k[0] 去掉批维得到 [Hkv, N, D]，transpose 成 [N, Hkv, D]，和池里每个槽位的 [Hkv, D] 对齐
            # 用槽位列表做下标赋值 = scatter：N 个新 token 各写进自己的物理槽位
            kv[li, 0, w_slots] = k[0].transpose(0, 1)
            kv[li, 1, w_slots] = v[0].transpose(0, 1)
            # 去掉批维、把 beam 维挪到最前：第 i 行就是第 i 个 beam 的 [H, D] query
            q = q[0].transpose(0, 1)  # [N, H, D]
        # --8<-- [end:decode_qkv]

            # --8<-- [start:decode_cascade]
            # outs 收集每个请求的 attention 输出；off 是这个请求的 beam 在 N 维上的起始行
            outs, off = [], 0
            # 按请求循环：不同请求的共享前缀是不同的物理块，没法塞进同一次 cascade 调用
            for g, ss, ps in zip(groups, shared_slots, priv_slots):
                w = len(g.beams)
                # kv[li, 0, ss]：用槽位列表做高级索引（gather），结果是一份新拷贝 [P, Hkv, D]
                # transpose 成 [Hkv, P, D]；每个请求只 gather 一次，W 个 beam 共用
                # Mac 小模型 P=8192 时这份拷贝每层 2 MB；真实 paged kernel 直接按块表读池，不物化
                # 历史短于一个块时 shared_len=0，没有共享段，传 None
                k_sh = kv[li, 0, ss].transpose(0, 1) if g.shared_len else None  # [Hkv, P, D]
                v_sh = kv[li, 1, ss].transpose(0, 1) if g.shared_len else None
                # ps 是 [W, S] 的槽位表，kv[li, 0, ps] 得到 [W, S, Hkv, D]
                # permute(0, 2, 1, 3) 换成 cascade 要的 [W, Hkv, S, D]
                k_pr = kv[li, 0, ps].permute(0, 2, 1, 3)  # [W, Hkv, S, D]
                v_pr = kv[li, 1, ps].permute(0, 2, 1, 3)
                # q[off:off + w] 取出这个请求的 W 行 query
                outs.append(cascade_attention(q[off:off + w], k_sh, v_sh, k_pr, v_pr))  # [W, H, D]
                off += w

            # 按 beam 顺序拼回 [N, H, D]，reshape 把 H、D 合成 C 并补回批维
            y = torch.cat(outs).reshape(1, n, -1)  # [1, N, C]
            # 输出投影 + 残差，再 MLP + 残差：和 S1 的 Block 完全一样，只是 attention 换成了 cascade
            x = x + block.attn.wo(y)
            x = x + block.mlp(block.mlp_norm(x))
            # --8<-- [end:decode_cascade]

        # --8<-- [start:decode_out]
        # 所有层的 KV 都写进去了，每个 beam 已计算的 token 数 +1
        for b in beams:
            b.num_computed += 1
        # 最终 RMSNorm + lm_head：[N, C] -> [N, V]
        logits = m.lm_head(m.norm(x[0]))  # [N, V]
        # view 成 [R, W, V]；成立的前提是每个请求的 beam 数都是 W
        return logits.view(len(groups), -1, logits.size(-1))  # [R, W, V]
        # --8<-- [end:decode_out]
        # --8<-- [end:decode_layers]

    # ---------------------------------------------------------------- 主流程
    # --8<-- [start:prefill_fork]
    # --8<-- [start:prefill_once]
    @torch.no_grad()
    def search(self, prompts: list[list[int]], beam_width: int, steps: int) -> tuple[list[list[tuple[list[int], float]]], SearchStats]:
        stats = SearchStats()
        w = beam_width
        # 1. prefill：每个请求的历史只算一次（命中跨请求缓存的部分不算）
        # 每个用户历史建一个 Sequence，此时只有 prompt、没有 output
        seqs = [self._new_seq(p) for p in prompts]
        for s in seqs:
            # 按链式哈希查跨请求缓存：同一用户上次请求写满的块直接引用（ref+1），返回命中的 token 数
            # 命中后 s.num_computed 已经等于命中数，剩下的才需要算
            stats.prefix_hit_tokens += self.bm.allocate_prefix(s)
            # 为没命中的部分分配新块；10K 历史、bs=16、不命中时是 625 块
            if self.bm.ensure_capacity(s, s.num_tokens) is None:
                raise RuntimeError("KV 块不够：增大 num_blocks")
        # num_pending = 还没算的 token 数：不命中时是整段历史，命中后只剩新追加的几十个
        batch = [(s, s.num_pending) for s in seqs]
        # prefill 计算量 = Σ 历史长度（S1 是 W × Σ 历史长度）；genrec_1b 10K 在 H800 上约 70 ms
        stats.prefill_tokens += sum(n for _, n in batch)
        # S5 ModelRunner：所有请求拼成一个变长 batch 前向，KV 按块表写进池，返回每个请求最后位置的 logits [V]
        logits = self.runner.run(batch)
        for s, n in batch:
            s.num_computed += n
            # 把新写满的块登记进哈希表，同一用户的下一次请求就能命中
            self.bm.register_full_blocks(s)
    # --8<-- [end:prefill_once]

    # --8<-- [start:fork_first]
        # 2. 第 1 个 token：从单个 prompt 扩展出 W 个 beam（fork，零拷贝）
        groups = []
        for s, lg in zip(seqs, logits):
            # lg [V] 是历史最后位置的 logits；.float() 转 fp32，log_softmax(-1) 沿词表维变成 log 概率
            # [None] 补成 [1, V] 配合 root(1)（1 个在根节点的状态），_mask 屏蔽第 1 层不合法的码，[0] 再去掉这一维
            logp = self._mask(lg.float().log_softmax(-1)[None], self.trie.root(1) if self.trie else None)[0]  # [V]
            # 只有 1 个父 beam，直接在 V 个 token 里取前 W 个：得分 top [W]，token tok [W]
            top, tok = logp.topk(w)
            # 每个 token fork 一个子序列：复制块表（10K 历史时 625 个整数），每块 ref += 1，不碰任何 KV
            beams = [self._fork(s, t) for t in tok.tolist()]
            # W 个 beam 都从根节点沿自己的 token 走一步；没有 trie 时用全 0 占位
            nodes = self.trie.step(self.trie.root(w), tok) if self.trie else torch.zeros(w, dtype=torch.long)  # [W]
            # prompt 序列的引用交给了 beam：释放后历史块的 ref 从 W+1 降到正好 W
            self.bm.free_seq(s)
            # 最后一个参数是 shared_len = 历史里写满的块对应的 token 数，例如 P=21、bs=8 时是 16
            groups.append(BeamGroup(len(s.prompt), beams, top, nodes, (len(s.prompt) // self.bs) * self.bs))
        # 此刻的 KV 占用 = 每个用户一份历史，还没有任何私有块
        stats.peak_blocks = max(stats.peak_blocks, self.bm.num_used)
    # --8<-- [end:fork_first]
        # --8<-- [end:prefill_fork]

        # --8<-- [start:beam_step]
        # --8<-- [start:beam_select]
        # 3. 后续每一步：前向所有 beam -> 每个请求在 W·V 个候选中选前 W 个 -> fork 新 beam、释放旧 beam
        for _ in range(1, steps):
            # 一次前向算完所有请求的所有 beam
            logits = self._decode(groups, stats)  # [R, W, V]
            for g, lg in zip(groups, logits):
                # 每个 beam 对 V 个 token 的 log 概率，当前前缀下的非法 token 被 trie 屏蔽成 -inf
                logp = self._mask(lg.float().log_softmax(-1), g.nodes)  # [W, V]
                # 累计得分 = 父 beam 得分 [W, 1] + 本步 log 概率 [W, V]，广播成 W·V 个候选
                cand = g.scores[:, None] + logp  # [W, V]
                # view(-1) 拍平成 [W·V]，在全部候选里取前 W 个；W=20、V=8192 时是 16 万选 20
                top, flat = cand.view(-1).topk(w)  # [W]
                # 拍平下标还原成 (父 beam, token)：行号 = flat // V，列号 = flat % V
                origin, tok = flat // logp.size(1), flat % logp.size(1)  # [W]
        # --8<-- [end:beam_select]

        # --8<-- [start:beam_fork]
                # 先 fork 新 beam：子 beam 复制父 beam 的块表，所有块 ref+1
                new = [self._fork(g.beams[o], t) for o, t in zip(origin.tolist(), tok.tolist())]
                # 再释放旧 beam。顺序反了的话，父 beam 独占的私有块会先归零回收，子 beam 就 fork 不到了
                for b in g.beams:
                    self.bm.free_seq(b)
                g.beams, g.scores = new, top
                # trie 状态跟着来源走：先按 origin 取父 beam 的节点 [W]，再沿选中的 token 走一步
                if self.trie is not None:
                    g.nodes = self.trie.step(g.nodes[origin], tok)  # [W]
                # --8<-- [end:beam_fork]
            stats.peak_blocks = max(stats.peak_blocks, self.bm.num_used)
        # --8<-- [end:beam_step]

        results = []
        for g in groups:
            res = [(b.output, s) for b, s in zip(g.beams, g.scores.tolist()) if s > float("-inf")]
            results.append(sorted(res, key=lambda x: -x[1]))
            for b in g.beams:
                self.bm.free_seq(b)
        stats.extra["kv_bytes_per_token"] = self.runner.kv[:, :, 0].numel() * self.runner.kv.element_size()
        return results, stats
