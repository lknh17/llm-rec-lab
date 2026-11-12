"""请求（序列）的状态。对应 vLLM 的 Request / SequenceGroup。"""

from dataclasses import dataclass, field


# --8<-- [start:sequence]
# --8<-- [start:sequence_fields]
# @dataclass 按下面的字段声明自动生成 __init__ 等方法，相当于一个带类型的结构体
@dataclass
class Sequence:
    # 请求 id，引擎按到达顺序递增分配
    seq_id: int
    # prompt 的 token id 列表，长度就是 prompt 长度（打标场景约 2K-5K）
    prompt: list[int]
    # 最多生成多少个 token；这里只按长度停，真实引擎还会检查 EOS 和 stop 字符串
    max_new_tokens: int
    # 0 表示 greedy；测试用 greedy 才能和参考实现逐 token 对比
    temperature: float = 0.0
    # 已生成的 token
    # field(default_factory=list)：每个实例各建一个新列表；直接写 = [] 会让所有实例共用同一个列表
    output: list[int] = field(default_factory=list)
    # 逻辑块 i -> 物理块 id（PagedAttention 的核心数据结构）
    # 长度 = ceil(已分配槽位的 token 数 / bs)；10K token、bs=16 时只有 625 个整数
    block_table: list[int] = field(default_factory=list)
    # 已写满的块的内容哈希（prefix caching 用），block_hashes[i] 对应 block_table[i]
    # 长度 <= len(block_table)：最后那个没写满的块还没有哈希
    block_hashes: list[int] = field(default_factory=list)
    # 已经算过、KV 已写进缓存的 token 数
    # 这是调度的唯一"进度条"：chunked prefill、抢占重算、prefix 命中都只改这一个数
    num_computed: int = 0
    # 到达序号；当前调度只靠队列顺序，没有读它，留给按到达时间排序的策略
    arrival: int = 0
# --8<-- [end:sequence_fields]

# --8<-- [start:sequence_props]
    # @property 把方法变成只读属性：写 seq.tokens 而不是 seq.tokens()，每次访问都现算
    @property
    def tokens(self) -> list[int]:
        # 完整序列 = prompt + 已生成；每次访问都拼一个新列表，O(L)
        # 教学版不在意这点开销；vLLM 用预分配的 numpy 数组存 token，避免反复拼接
        return self.prompt + self.output

    @property
    def num_tokens(self) -> int:
        # 当前总长度 L，也是这条请求最终要占的 KV 槽位数
        return len(self.prompt) + len(self.output)

    @property
    def num_pending(self) -> int:
        """还需要喂给模型计算的 token 数：prefill 阶段是剩余 prompt，decode 阶段是 1（刚采样出的 token）。"""
        # 三种情况都由这一个减法覆盖：
        # 新请求 = prompt 长度（prefix 命中后再减去命中数）；decode = 1；
        # 被抢占后 num_computed 清零 = prompt + 已生成的全部 token，重算自然发生
        return self.num_tokens - self.num_computed

    @property
    def finished(self) -> bool:
        return len(self.output) >= self.max_new_tokens
# --8<-- [end:sequence_props]
# --8<-- [end:sequence]
