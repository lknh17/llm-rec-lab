"""vLLM V1 的自定义 logits processor：语义 ID 的 trie 约束解码。【需要在 GPU + 对应 vLLM 版本上验证】

接口按 vLLM 0.10.2+ 的 V1 logits processor 编写（vllm/v1/sample/logits_processor/）：
- 引擎为整个 batch 创建一个 processor 实例（不是每个请求一个）；
- update_state(batch_update)：batch 里有请求加入 / 结束 / 换位置时调用，processor 自己维护"行号 -> 请求状态"；
- apply(logits [num_reqs, V])：返回处理后的 logits。

请求通过 SamplingParams.extra_args 开启约束：
    SamplingParams(..., extra_args={"genrec_prefix_len": s})
表示 prompt 的最后 s 个 token 是当前 beam 已经生成的语义 ID 前缀（前端 beam 循环把 beam 前缀拼在 prompt 末尾）。

trie 数据通过环境变量 GENREC_TRIE 指向 export_hf.py 导出的 semantic_ids.pt
（processor 在 EngineCore 进程里构造，不能直接传 Python 对象）。

版本差异：如果你的 vLLM 里 BatchUpdate.added 的元组字段不同，或者提供了 `process_dict_updates` 辅助函数，
按 vllm/v1/sample/logits_processor/ 里内置 processor（如 MinTokensLogitsProcessor）的写法调整 update_state。
"""

import os

import torch
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality

from stages.s06_recsys.code.rq_kmeans import SemanticIDs, SemanticTokenizer
from stages.s07_capstone.code.trie_gpu import TensorTrie


def load_trie(path: str, device: torch.device) -> TensorTrie:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tok = SemanticTokenizer(SemanticIDs(ck["codes"], [], ck["k"], [], 0.0, []))
    return TensorTrie(tok.item_tokens[1:].tolist(), tok.vocab_size, device)


class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        self.trie = load_trie(os.environ["GENREC_TRIE"], device)
        self.cpu_next = self.trie.next_node.cpu()  # 前缀 -> 节点的遍历在 CPU 上做（每个请求只做一次）
        # 行号 -> (prompt 中前缀所在的节点, 该请求的 output_tok_ids 列表引用)
        self.rows: dict[int, tuple[int, list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        return False  # 会改变 argmax（屏蔽掉了模型最想要的非法 token）

    def _walk(self, node: int, tokens: list[int]) -> int:
        for t in tokens:
            if node < 0:
                break
            node = int(self.cpu_next[node, t])
        return node

    # --8<-- [start:update_state]
    # --8<-- [start:update_added]
    def update_state(self, batch_update: BatchUpdate | None) -> None:
        # None 表示这一步 batch 的组成没变（没有请求加入、结束或换位置），映射不用动
        if batch_update is None:
            return
        # added：新进入 batch 的请求；index 是它在 batch 里的行号，也就是 logits 的第几行
        # output_ids 是引擎持有的列表，之后生成的 token 会被追加进去
        for index, params, prompt_ids, output_ids in batch_update.added:
            # 没设 extra_args 的普通请求取空字典
            extra = getattr(params, "extra_args", None) or {}
            if "genrec_prefix_len" in extra:
                # s = prompt 末尾属于 beam 前缀的 token 数，前端循环的第 s 步就传 s
                s = int(extra["genrec_prefix_len"])
                # 取出这 s 个 token；s=0（第 1 步）时前缀为空，停在根节点
                prefix = list(prompt_ids[len(prompt_ids) - s:]) if s else []
                # 在 CPU 表上从根走 s 步得到节点号，每个请求只在加入时走一次（s ≤ 3，几次查表）
                # output_ids 存引用而不是拷贝：apply 时能看到引擎新追加的 token
                self.rows[index] = (self._walk(0, prefix), output_ids)
            else:
                # 这个行号被一个不受约束的请求占了：清掉旧状态，防止把约束套到它身上
                self.rows.pop(index, None)
    # --8<-- [end:update_added]

    # --8<-- [start:update_moved]
        # removed：已结束请求的行号，直接删掉
        for index in batch_update.removed:
            self.rows.pop(index, None)
        # moved：batch 压缩或重排，每项是 (a, b, 方向)
        #   单向（UNIDIRECTIONAL）：行 a 的请求挪到行 b，b 原本是空位；SWAP：a、b 两行互换
        for a, b, direction in batch_update.moved:
            # 两边先都取出来，避免边读边写把对方覆盖
            va, vb = self.rows.pop(a, None), self.rows.pop(b, None)
            if va is not None:
                self.rows[b] = va
            # 只有 SWAP 时 b 的原状态才搬到 a
            if direction == MoveDirectionality.SWAP and vb is not None:
                self.rows[a] = vb
    # --8<-- [end:update_moved]
    # --8<-- [end:update_state]

    # --8<-- [start:apply]
    # --8<-- [start:apply_nodes]
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # logits [num_reqs, V]，batch 里每个请求一行；没有受约束的请求就原样返回，零开销
        if not self.rows:
            return logits
        # 受约束的行号，排序后和下面的 nodes 一一对应
        idx = sorted(self.rows)
        # 加上已生成的 output token：从 prompt 前缀所在的节点出发继续走
        # 前端 beam 每个请求只生成 1 个 token，这时 output 为空，循环 0 次
        nodes = [self._walk(self.rows[i][0], self.rows[i][1]) for i in idx]
        # 行号搬到 GPU：一次很小的 CPU -> GPU 拷贝
        rows = torch.tensor(idx, device=logits.device)  # [n]
        # clamp(min=0)：走出 trie 的 -1 临时换成根节点 0，保证下标合法，下一行再整行屏蔽
        node_t = torch.tensor(nodes, device=logits.device).clamp(min=0)  # [n]
    # --8<-- [end:apply_nodes]

    # --8<-- [start:apply_mask]
        # 一次 gather 得到 n 个请求的合法掩码
        allowed = self.trie.allowed(node_t)  # [n, V]
        # 走出 trie 的请求所有 token 都不合法
        allowed[torch.tensor(nodes, device=logits.device) < 0] = False
        # logits[rows] 是高级索引，取出来的是拷贝 [n, V]；masked_fill 后再下标赋值写回原 logits
        # [:, : logits.size(1)] 是防御性切片：掩码宽度不超过 logits 宽度，两者通常都等于词表大小
        logits[rows] = logits[rows].masked_fill(~allowed[:, : logits.size(1)], float("-inf"))
        return logits
    # --8<-- [end:apply_mask]
    # --8<-- [end:apply]
