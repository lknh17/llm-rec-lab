"""S1 第 5-6 课：采样策略、带/不带 KV cache 的生成、beam search。

beam search 预留了 allowed_fn 钩子：每一步只允许合法 token。S6/S7 的生成式推荐用它约束模型只生成真实存在的 item（语义 ID 前缀树）。
"""

from collections.abc import Callable

import torch

from stages.s01_transformer.code.model import GPT, KVCache


# --8<-- [start:sample_next]
# --8<-- [start:sample_temp_topk]
def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: [B, V] -> 下一个 token [B]。temperature=0 表示 greedy。"""
    # greedy：每行取最大值的下标。argmax(-1) 沿最后一维（词表维）找最大，[B, V] -> [B]
    # 不碰随机数，同样的输入永远得到同样的输出，打标、分类这类要求可复现的任务用它
    if temperature == 0:
        return logits.argmax(-1)
    # 除以温度等价于把所有 logit 之间的差距缩放 1/T：差距放大，softmax 更尖；差距缩小，更平
    # T<1 分布更尖（保守），T>1 更平（多样）
    # 只是一次 [B, V] 的逐元素运算，V=15 万、B=256 时约 3900 万个元素，很便宜
    logits = logits / temperature  # [B, V]
    if top_k is not None:
        # topk 返回按降序排好的前 k 个 (values, indices)，values 形状 [B, k]
        # [:, -1:] 取最后一列即第 k 大的值；用切片 -1: 而不是 -1，保留一维成 [B, 1]，才能和 [B, V] 广播
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]  # [B, 1]
        # 比第 k 大还小的都置 -inf，softmax 后概率为 0；和第 k 大并列的值会被一起保留
        logits = logits.masked_fill(logits < kth, float("-inf"))  # [B, V]
# --8<-- [end:sample_temp_topk]
# --8<-- [start:sample_topp]
    if top_p is not None:
        # nucleus：保留累计概率刚好超过 p 的最小集合。和 top-k 相比，它能随分布形状自适应候选数
        # 降序排序：sorted_logits 是排好的值，sorted_idx 记住每个值原来在词表里的位置
        # 代价：每步对 B 行各排一次 V 个数，V 大时这是采样里最贵的一步
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)  # [B, V], [B, V]
        # 排序后的概率做前缀和：cum[b, j] = 前 j+1 个 token 的概率之和，从小到大单调增到 1
        cum = sorted_logits.softmax(-1).cumsum(-1)  # [B, V]
        # 当前 token 之前的累计已超过 p，则移除
        # cum - p_j 是"排在 j 之前"的累计概率。只有前面的已经超过 top_p，j 才是多余的
        # 这样第一个让累计越过阈值的 token 一定保留，候选集至少有 1 个
        # 例：概率 [0.71, 0.26, ...]、top_p=0.5：j=0 前面累计 0 保留，j=1 前面累计 0.71 移除
        # 这里 softmax 算了两遍，可以复用；推理引擎会融合成一个 kernel
        remove = cum - sorted_logits.softmax(-1) > top_p  # [B, V] bool
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))  # [B, V]
        # scatter 把排好序的值按 sorted_idx 写回原位置：out[b, sorted_idx[b, j]] = sorted_logits[b, j]
        # 先用 full_like 建一个全 -inf 的 [B, V]，写回后词表顺序复原，被移除的仍是 -inf
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
# --8<-- [end:sample_topp]
# --8<-- [start:sample_draw]
    # -inf 的位置 softmax 后正好是 0，剩下的候选重新归一化，和为 1
    probs = logits.softmax(-1)  # [B, V]
    # multinomial 按概率每行抽 1 个下标，返回 [B, 1]；squeeze(-1) 去掉最后那一维得到 [B]
    # 传入 generator 可以固定随机序列，测试和实验因此可复现
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
# --8<-- [end:sample_draw]
# --8<-- [end:sample_next]


# --8<-- [start:generate]
# --8<-- [start:generate_prefill]
# torch.no_grad()：整个生成过程不记录计算图、不保存激活，推理代码必加，否则显存会随步数增长
@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool = True,
    **sample_kwargs,
) -> torch.Tensor:
    """idx: [B, T] -> [B, T + max_new_tokens]

    use_cache=False：每一步把整条序列重新算一遍，第 n 步的计算量 ∝ (T+n)，总计 O(N²)。
    use_cache=True ：先 prefill 整个 prompt，之后每步只算 1 个新 token，历史的 K/V 从缓存读取。
    """
    cache = None
    if use_cache:
        # 随便取一个参数，只为拿到模型所在的设备和 dtype，让缓存和模型保持一致
        p = next(model.parameters())
        # 按"prompt 长度 + 要生成的个数"一次性预分配，生成过程中不再申请显存
        cache = KVCache(model.cfg, idx.size(0), idx.size(1) + max_new_tokens, p.device, p.dtype)
        # prefill：整个 prompt 一次前向，每层的 K、V 顺带写进缓存，cache.length 变成 T
        # 返回了全部 T 个位置的 logits，但只有最后一个有用；vLLM 在 prefill 时只对最后一个位置算 lm_head
        # 量级：Qwen2.5-7B（C=3584、V≈15 万）、T=4K 时，多算的 lm_head 约 4.5 TFLOPs、logits 约 1.2 GB（BF16）
        logits = model(idx, cache)  # prefill：[B, T, V]
# --8<-- [end:generate_prefill]
# --8<-- [start:generate_decode]
    for step in range(max_new_tokens):
        if not use_cache:
            # 无缓存：每步把整条序列（prompt + 已生成的）从头再算一遍，第 n 步处理 T+n 个 token
            logits = model(idx)  # 每步全量重算：[B, T+step, V]
        # 两条路径都只看最后一个位置：logits[:, -1] 是 [B, V]
        next_tok = sample_next(logits[:, -1], **sample_kwargs)  # [B]
        # next_tok[:, None] 把 [B] 变成 [B, 1]，接到序列末尾；只拷贝 token id，很小
        idx = torch.cat([idx, next_tok[:, None]], dim=1)  # [B, T+step+1]
        # 有缓存：只把新 token 喂进去，输入 [B, 1]；历史的 K、V 从缓存读，位置从 cache.length 开始
        # 最后一步不再前向：新 token 已经拿到，再算一次 logits 也用不上
        if use_cache and step < max_new_tokens - 1:
            logits = model(next_tok[:, None], cache)  # decode：[B, 1, V]
    return idx
# --8<-- [end:generate_decode]
# --8<-- [end:generate]


AllowedFn = Callable[[list[list[int]]], torch.Tensor]


# --8<-- [start:beam_search]
# --8<-- [start:beam_signature]
@torch.no_grad()
def beam_search(
    model: GPT,
    prompt: torch.Tensor,
    beam_width: int,
    max_new_tokens: int,
    allowed_fn: AllowedFn | None = None,
) -> list[tuple[list[int], float]]:
    """prompt: [T]。返回按得分（对数概率之和）降序排列的 [(生成的 token 列表, 得分)]。

    每步：W 个 beam 各自扩展 V 个候选 -> W·V 个候选 -> 保留得分最高的 W 个。
    allowed_fn(prefixes) -> bool [W, V]：约束解码，非法 token 的得分置为 -inf。

    这里的 KV cache 是"连续"的：每步按 beam 的来源重排 cache（cache.k[:, origin]），
    要拷贝 W 份完整的历史 KV —— 当历史很长（推荐场景的 10k 用户序列）且 W 很大时，这是巨大浪费。
    S7 用分页 + 引用计数 + 写时复制（copy-on-write）让所有 beam 共享同一份历史 KV。
    """
# --8<-- [end:beam_signature]
# --8<-- [start:beam_prefill]
    p = next(model.parameters())
    w = beam_width
    t = prompt.numel()
    # W 条 beam 各占缓存的一行 batch：[L, W, Hkv, T+N, D]，显存是单条序列的 W 倍
    cache = KVCache(model.cfg, w, t + max_new_tokens, p.device, p.dtype)
    # prefill：W 个 beam 的 prompt 完全一样，这里为了简单直接算 W 份（也是浪费，S7 会改成只算 1 份）
    # prompt[None]：[T] -> [1, T]；expand(w, t)：[1, T] -> [W, T]，只把第 0 维步长设为 0，不复制
    # .contiguous() 才真的复制成 W 行；随后模型对 W 行做完全相同的计算，KV 也存了 W 份一样的
    # [:, -1] 只取每行最后一个位置的 logits：[W, T, V] -> [W, V]
    logits = model(prompt[None].expand(w, t).contiguous(), cache)[:, -1]  # [W, V]
    # scores[i]：第 i 条 beam 到目前为止的累计 log 概率
    # torch.full(形状, 值) 新建一个每个元素都等于该值的张量
    scores = torch.full((w,), float("-inf"), device=p.device)  # [W]
    # 此时 W 行完全相同；若都从 0 分出发，第一步会选出 W 份一模一样的扩展
    # 只让第 0 行是 0、其余 -inf，第一步只从 1 个 beam 扩展，候选只来自这一行的 V 个 token
    scores[0] = 0.0
    # 每条 beam 已生成的 token 用 Python 列表记录，给 allowed_fn 查前缀树用
    beams: list[list[int]] = [[] for _ in range(w)]
# --8<-- [end:beam_prefill]

# --8<-- [start:beam_expand]
    for step in range(max_new_tokens):
        # .float() 先转 FP32 再算 log_softmax：BF16 下累加几十步 log 概率会丢精度，排序可能错
        # log_softmax 一步算出 log p，比 softmax 后再 log 数值更稳（不会出现 log 0）
        logp = logits.float().log_softmax(-1)  # [W, V]
        if allowed_fn is not None:
            # allowed_fn 返回 [W, V] 的布尔表，True 为合法；~ 取反后把非法位置的 log p 置为 -inf
            logp = logp.masked_fill(~allowed_fn(beams), float("-inf"))  # [W, V]
        # scores[:, None] 是 [W, 1]，广播加到每一行：候选 (i, v) 的分数 = 父 beam 分数 + log p(v)
        cand = scores[:, None] + logp  # [W, V]
        # view(-1) 把 W 行摊成一行 [W·V]，再取全局前 W 个；不同父 beam 的候选在这里直接竞争
        # 本课 W=128、V=256 时是 3.3 万个候选；W=100、V=15 万时是 1500 万个，topk 本身就不便宜
        top_scores, flat_idx = cand.view(-1).topk(w)  # [W], [W]
        # 展平下标 = 行号 × V + 列号，所以整除得到行号（父 beam），取余得到列号（token）
        origin = flat_idx // logp.size(1)  # 来自哪个 beam
        token = flat_idx % logp.size(1)  # 选了哪个 token
        # 新 beam = 父 beam 的前缀 + 新 token；.tolist() 把 GPU 张量拷回 CPU，每步一次同步
        beams = [beams[o] + [tok] for o, tok in zip(origin.tolist(), token.tolist())]
        scores = top_scores  # [W]
# --8<-- [end:beam_expand]
# --8<-- [start:beam_reorder]
        # 最后一步已经选出结果，不用再重排缓存和前向
        if step == max_new_tokens - 1:
            break
        # 按来源重排 cache：每个新 beam 继承其父 beam 的历史 KV（一次完整拷贝）
        # cache.k[:, origin] 沿第 1 维（beam）按下标取，会新建一个完整副本，再写回 cache.k
        # 拷的是整个预分配长度 T+N，不只是已写入的部分；多个新 beam 同父时，父的历史被复制多份
        # 本课 W=128、80 个位置、6 层 K+V 约 63 MB（FP32），每步至少读写 126 MB
        # 高级索引先生成临时副本再拷回，实际访存约再翻一倍
        cache.k[:, :] = cache.k[:, origin]  # [L, W, Hkv, S_max, D]
        cache.v[:, :] = cache.v[:, origin]  # [L, W, Hkv, S_max, D]
        # 每条 beam 只喂自己新选的 1 个 token：[W] -> [W, 1]，相当于 batch=W 的一次 decode
        logits = model(token[:, None], cache)[:, -1]  # [W, V]

    # 过滤掉分数仍为 -inf 的 beam（合法 token 少于 W 个时会出现），按分数降序返回
    results = [(b, s) for b, s in zip(beams, scores.tolist()) if s > float("-inf")]
    return sorted(results, key=lambda x: -x[1])
# --8<-- [end:beam_reorder]
# --8<-- [end:beam_search]
