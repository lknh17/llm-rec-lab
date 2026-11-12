"""分页 KV 的物理块管理。对应 vLLM 的 BlockSpaceManager / KVCacheManager。

KV 不再按请求预留一整段连续显存，而是切成固定大小（block_size 个 token）的物理块，按需分配：
- 内部碎片只存在于每个请求的最后一块（< block_size 个 token），显存利用率从 20-40% 提升到 90%+。
- 块可以被多个序列共享（引用计数）：
  * prefix caching：内容相同的前缀块（按 token 内容哈希）跨请求复用，免掉重复 prefill；
  * fork：beam search / 并行采样的子序列共享父序列的块，写入时才复制（copy-on-write）。
"""

from collections import OrderedDict

from stages.s05_inference.code.mini_engine.sequence import Sequence


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.free: list[int] = list(range(num_blocks - 1, -1, -1))  # 空闲块（栈）
        self.ref = [0] * num_blocks  # 每个物理块被多少个序列引用
        self.hash_to_block: dict[int, int] = {}  # 内容哈希 -> 物理块（只登记写满的块）
        self.block_to_hash: dict[int, int] = {}
        self.evictable: OrderedDict[int, None] = OrderedDict()  # ref=0 但内容仍有效的缓存块（LRU）
        self.stats = {"prefix_hit_tokens": 0, "cow_copies": 0, "peak_used": 0}

    # ---------------------------------------------------------------- 基础分配
    @property
    def num_available(self) -> int:
        return len(self.free) + len(self.evictable)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_available

    # --8<-- [start:alloc_release]
    # --8<-- [start:alloc]
    def _alloc(self) -> int:
        # 优先拿真正空闲的块；list.pop() 取栈顶，O(1)
        # free 初始化成 [n-1, ..., 0]，所以最先分出去的是 0、1、2……，调试时块表好读
        if self.free:
            b = self.free.pop()
        else:
            # 没有空闲块时，淘汰最久未用的缓存块（它的内容作废）
            # popitem(last=False) 弹出 OrderedDict 里最早插入的项，也就是 LRU 的队头
            # 调用方已用 num_available（free + evictable）检查过额度，这里不会弹空
            b, _ = self.evictable.popitem(last=False)
            # 先从 block_to_hash 删掉 b、拿到它的哈希，再用哈希删 hash_to_block 的登记
            # 两张表必须一起删：否则后来的请求会"命中"一个已被覆盖的块，输出悄悄出错
            del self.hash_to_block[self.block_to_hash.pop(b)]
        # 新分配的块只有调用者一个引用者
        self.ref[b] = 1
        # 记录峰值占用块数；第 5 步的"分页 KV 峰值"= 峰值块数 × bs × 每 token KV 字节
        self.stats["peak_used"] = max(self.stats["peak_used"], self.num_used)
        return b
    # --8<-- [end:alloc]

    # --8<-- [start:release]
    def _release(self, b: int) -> None:
        # 一个引用者放弃块 b；共享块（fork、prefix 命中）要等最后一个引用者走了才回收
        self.ref[b] -= 1
        if self.ref[b] == 0:
            # 登记过哈希 = 写满且内容可复用：放进 LRU 队尾，而不是立刻作废
            if b in self.block_to_hash:
                # 内容保留，等待被复用或淘汰；OrderedDict 在这里当有序集合用，值没有意义
                self.evictable[b] = None
            else:
                # 没写满、或者没开 prefix cache 的块：内容没有复用价值，直接还给空闲栈
                self.free.append(b)
    # --8<-- [end:release]
    # --8<-- [end:alloc_release]

    # --8<-- [start:prefix_cache]
    # --8<-- [start:prefix_hash]
    def _hash(self, parent: int | None, tokens: list[int]) -> int:
        # 哈希包含父块的哈希：同样的 16 个 token 出现在不同前缀之后，KV 不同（位置和上文都不同）
        # 父哈希又包含祖父哈希……所以一个块的哈希等价于"从开头到本块末尾的全部 token"
        # list 不可哈希，要先转成 tuple；(parent, tuple) 整体作为一个键
        # Python 内置 hash 是 64 位整数，单租户下碰撞概率可忽略；vLLM 可选 SHA-256 防碰撞
        return hash((parent, tuple(tokens)))
    # --8<-- [end:prefix_hash]

    # --8<-- [start:prefix_lookup]
    # ---------------------------------------------------------------- prefix caching
    def allocate_prefix(self, seq: Sequence) -> int:
        """新序列进入时：查找已缓存的前缀块，命中的直接引用。返回命中的 token 数。
        至少留最后 1 个 token 不命中：需要算它才能得到下一个 token 的 logits。"""
        if not self.enable_prefix_cache:
            return 0
        bs, tokens = self.block_size, seq.tokens
        # 第 0 块没有父块，用 None 作为哈希链的起点
        parent = None
        # 只查完整块，且留最后 1 个 token：L 个 token 最多查 (L-1)//bs 块
        # 例：L=336、bs=16 -> 查 20 块（320 token）；L=320 -> 只查 19 块，最后 16 个要重算
        for i in range((len(tokens) - 1) // bs):
            # 第 i 块覆盖 token [i·bs, (i+1)·bs)，连同父哈希算出本块的身份
            h = self._hash(parent, tokens[i * bs:(i + 1) * bs])
            # dict.get 查不到返回 None；O(1)，10K token 也只查 625 次
            b = self.hash_to_block.get(h)
            # 前缀匹配：第一个没命中的块之后父哈希都变了，后面不可能再命中
            if b is None:
                break
    # --8<-- [end:prefix_lookup]
    # --8<-- [start:prefix_hit]
            # ref=0 说明这个块躺在 LRU 里：先捞出来，否则之后会被 _alloc 当成可淘汰块覆盖
            if self.ref[b] == 0:
                self.evictable.pop(b)
            # 引用计数 +1：这个物理块现在被本序列共享，KV 一个字节都不复制
            self.ref[b] += 1
            # 块表和哈希列表同步追加；调用方保证进来时块表为空，所以第 i 次追加就是逻辑块 i
            seq.block_table.append(b)
            seq.block_hashes.append(h)
            # 哈希链往下走一格
            parent = h
        # 命中 token 数 = 命中块数 × bs，一定是 bs 的整数倍
        hit = len(seq.block_table) * bs
        # 命中部分直接记成"已算"：调度器之后只给剩下的 num_pending 分配预算
        # 调度器不知道也不需要知道发生过命中，这就是"prefix 命中对调度透明"
        seq.num_computed = hit
        self.stats["prefix_hit_tokens"] += hit
        return hit
    # --8<-- [end:prefix_hit]

    # --8<-- [start:prefix_register]
    def register_full_blocks(self, seq: Sequence) -> None:
        """计算完成后，把新写满的块登记进哈希表，供后续请求复用。"""
        if not self.enable_prefix_cache:
            return
        bs = self.block_size
        # 已登记到第 len(block_hashes) 块；本步算完后，写满的块到第 num_computed // bs 块为止
        # decode 时多数步这个 range 是空的，每 bs 步才登记 1 块；chunked prefill 每个 chunk 登记一批
        for i in range(len(seq.block_hashes), seq.num_computed // bs):
            # 父哈希就是上一块的哈希；第 0 块没有父块
            parent = seq.block_hashes[-1] if seq.block_hashes else None
            h = self._hash(parent, seq.tokens[i * bs:(i + 1) * bs])
            # 序列自己的哈希列表总是追加，不管下面登没登记进全局表
            seq.block_hashes.append(h)
            b = seq.block_table[i]
            # h 已存在：同样内容已有别的物理块登记过（比如同一步一起 prefill 的两条请求），保留先来的
            # 这样本块不进全局表，引用归零时直接回 free，不会出现两个块抢一个哈希
            # b 已登记：防御性检查，一个物理块只对应一个哈希，保证两张表互为反向映射
            if h not in self.hash_to_block and b not in self.block_to_hash:
                self.hash_to_block[h] = b
                self.block_to_hash[b] = h
    # --8<-- [end:prefix_register]
    # --8<-- [end:prefix_cache]

    # ---------------------------------------------------------------- 为本步计算准备块
    # --8<-- [start:ensure_capacity]
    # --8<-- [start:capacity_check]
    def ensure_capacity(self, seq: Sequence, total_tokens: int) -> list[tuple[int, int]] | None:
        """保证 seq 能容纳 total_tokens 个 token 的 KV。返回需要执行的块复制 [(src, dst)]；块不够时返回 None。

        写时复制：本步要写入的第一个块如果被共享（ref>1，来自 fork），先复制一份私有的再写。
        """
        bs = self.block_size
        # -(-a // b) 是整数向上取整 ceil(a / b) 的惯用写法：// 向下取整，两次取负就变成向上
        # need_new = 本步结束时需要的块数 - 已有块数；只在跨过块边界时为正，decode 时 15/16 的步数是 0
        need_new = -(-total_tokens // bs) - len(seq.block_table)
        # 本步第一个新 token 的位置是 num_computed，它落在第 write_idx 个逻辑块
        write_idx = seq.num_computed // bs
        # 写时复制的条件：这一块已经存在（不是本步新分配的），而且被别的序列共享（ref>1）
        # 只有 fork 会造成"共享且半满"的块；prefix 命中的块都写满了，write_idx 正好指向它们之后
        cow = write_idx < len(seq.block_table) and self.ref[seq.block_table[write_idx]] > 1
        # 一次性检查额度：新块数 + 可能的 1 个复制块；不够就什么都不做，返回 None
        # 先查后分，失败时没有半截分配要回滚；抢占谁由调度器决定，块管理器不做策略
        if need_new + int(cow) > self.num_available:
            return None
        copies = []
        # --8<-- [end:capacity_check]
        # --8<-- [start:capacity_cow]
        if cow:
            # src：和别人共享的半满块；dst：给本序列新分配的私有块
            src = seq.block_table[write_idx]
            dst = self._alloc()
            # 本序列放弃对 src 的引用（ref-1）；src 继续归其他共享者，内容不动
            self._release(src)
            # 块表改指向私有副本，之后本序列的新 token 写进 dst，不会污染兄弟序列
            seq.block_table[write_idx] = dst
            # 被复制的块之后不再与原内容一致
            # write_idx 这一块是半满的、本来就没有哈希，这一行通常删掉 0 个元素，是保险
            del seq.block_hashes[write_idx:]
            # 真正的 KV 拷贝不在这里做：块管理器只管账本，(src, dst) 交给 model runner 执行
            # 一次拷贝 [L, 2, bs, Hkv, D]：9.5M 模型 bs=16 约 96 KB，Llama-3-8B 约 2 MB
            copies.append((src, dst))
            self.stats["cow_copies"] += 1
        # 跨过块边界才追加新块；max(0, ...) 兜底，正常调度下 need_new 不会为负
        for _ in range(max(0, need_new)):
            seq.block_table.append(self._alloc())
        # 返回空列表表示"成功且无需复制"，和 None（块不够）区分开
        return copies
        # --8<-- [end:capacity_cow]
    # --8<-- [end:ensure_capacity]

    # --8<-- [start:fork_free]
    # --8<-- [start:fork_share]
    def fork(self, parent: Sequence, child: Sequence) -> None:
        """子序列共享父序列的全部块（只增加引用计数，不复制任何 KV）。"""
        # .copy() 复制的是块表这个整数列表，物理块本身仍然共享
        # 10K token、bs=16 时只复制 625 个整数，成本和 1.97 GB 的 KV 大小无关
        child.block_table = parent.block_table.copy()
        # 写满块的哈希也继承：对父子来说是同一份内容
        child.block_hashes = parent.block_hashes.copy()
        # 子序列从父序列的进度继续，不重算任何 token
        child.num_computed = parent.num_computed
        # 每个共享块引用计数 +1；之后谁先往半满的最后一块写，谁就在 ensure_capacity 里触发写时复制
        for b in child.block_table:
            self.ref[b] += 1
    # --8<-- [end:fork_share]

    # --8<-- [start:free_release]
    def free_seq(self, seq: Sequence) -> None:
        # 请求结束或被抢占时调用：逐块减引用，共享块要等最后一个引用者释放才真正回收
        for b in seq.block_table:
            self._release(b)
        # 清空三项状态；num_computed 归零是"抢占后重算"的关键：num_pending 变回全部 token
        seq.block_table, seq.block_hashes, seq.num_computed = [], [], 0
    # --8<-- [end:free_release]
    # --8<-- [end:fork_free]
