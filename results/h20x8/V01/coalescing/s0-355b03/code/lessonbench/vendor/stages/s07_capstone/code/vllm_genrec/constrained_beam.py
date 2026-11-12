"""在真实 vLLM 上做 trie 约束的 beam search，并与我们的分页 beam 引擎对比。【需要 GPU】

    uv run python -m stages.s07_capstone.code.vllm_genrec.export_hf
    GENREC_TRIE=data/hf/tiger_ml1m/semantic_ids.pt \
        .venv-vllm/bin/python -m stages.s07_capstone.code.vllm_genrec.constrained_beam --users 512 --beam 20

做法（和 vLLM V1 自带的 LLM.beam_search 相同）：
- vLLM V1 的引擎核心不再支持 beam search（V0 里的 fork + 写时复制实现因为复杂度被移除了），
  改为在前端循环：每一步把"历史 + beam 前缀"作为新的 prompt，生成 1 个 token，取 top-k logprobs 扩展 beam。
- beam 之间、步与步之间的历史 KV 共享，完全依赖 **自动前缀缓存**（APC）：同一个用户的所有 beam 的 prompt
  都以同样的历史开头，块哈希相同，历史只 prefill 一次。
- 与我们的引擎相比，代价是：每一步都是一批"新请求"（调度、哈希查找、输入准备的 CPU 开销）；
  attention 没有 cascade（每个 beam 各读一遍历史 KV，除非开启 vLLM 的 cascade attention 后端）。

得分口径：logits processor 先屏蔽非法 token，logprobs_mode="processed_logprobs" 让返回的 logprob
在合法 token 上重新归一化。我们的引擎用 renormalize=True 对齐。
"""

import argparse
import os
import time

import torch

from stages.s07_capstone.code.vllm_genrec.export_hf import ROOT


# --8<-- [start:frontend_beam]
# --8<-- [start:frontend_reqs]
def vllm_beam_search(llm, prompts: list[list[int]], beam: int, steps: int) -> list[list[tuple[list[int], float]]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    # 每个用户一开始只有 1 个空 beam：(已生成的 token 列表, 累计 log 概率)
    beams = [[([], 0.0)] for _ in prompts]
    for step in range(steps):
        # reqs：这一步发给 vLLM 的全部请求；owners 记每个请求属于哪个用户、哪个 beam
        reqs, owners = [], []
        for r, (prompt, bs) in enumerate(zip(prompts, beams)):
            for toks, score in bs:
                # 每个 beam 一个独立请求：prompt = 历史 + 这个 beam 的前缀
                # 第 1 步每个用户 1 个请求，之后每个用户 W 个；512 用户 × beam 20 时每步约 1 万个请求
                # 同一用户的请求开头的历史完全相同 -> APC 块哈希命中，写满的历史块不重算
                # TokensPrompt：直接给 token id 的 prompt，不经过 tokenizer（语义 ID 没有文本形式）
                reqs.append(TokensPrompt(prompt_token_ids=prompt + toks))
                owners.append((r, toks, score))
# --8<-- [end:frontend_reqs]

# --8<-- [start:frontend_expand]
        # max_tokens=1：每个请求只生成 1 个 token 就结束；temperature=0 是贪心，这里只用它的 logprobs
        # logprobs=beam：返回 top-W 的 (token, logprob)，一个 beam 最多贡献 W 个候选，W 个就够了
        # extra_args 把前缀长度 step 传给 TrieLogitsProcessor
        sp = SamplingParams(max_tokens=1, temperature=0, logprobs=beam, extra_args={"genrec_prefix_len": step})
        # 一次 generate：调度 -> APC 查块 -> prefill 没命中的尾巴（不满一块的历史 + beam 前缀）-> 采样 1 个 token
        # 返回的 outs 和 reqs 一一对应；use_tqdm=False 只是关掉进度条
        outs = llm.generate(reqs, sp, use_tqdm=False)
        cand: list[list[tuple[list[int], float]]] = [[] for _ in prompts]
        # outs 和 reqs 顺序相同，用 owners 找回每个结果属于哪个用户的哪个 beam
        for (r, toks, score), out in zip(owners, outs):
            # outputs[0]：这个请求的第 0 条输出序列（每个请求只要 1 条）
            # logprobs[0] 是第 0 个生成位置的 {token_id: Logprob} 字典，lp.logprob 是其中的 float
            for tid, lp in out.outputs[0].logprobs[0].items():
                # 被 processor 屏蔽的非法 token logprob 是 -inf，跳过
                if lp.logprob > float("-inf"):
                    cand[r].append((toks + [tid], score + lp.logprob))
        # 每个用户在最多 W·W 个候选里按累计得分取前 W 个；在前端进程里用 Python 排序
        beams = [sorted(c, key=lambda x: -x[1])[:beam] for c in cand]
    return beams
# --8<-- [end:frontend_expand]
# --8<-- [end:frontend_beam]


def main() -> None:
    from vllm import LLM

    from stages.s01_transformer.code.model import GPT
    from stages.s06_recsys.code.data import load
    from stages.s06_recsys.code.rq_kmeans import SemanticIDs, SemanticTokenizer
    from stages.s06_recsys.code.train_genrec import CKPT, MAX_ITEMS, metrics
    from stages.s07_capstone.code.beam_engine import PagedBeamSearch
    from stages.s07_capstone.code.trie_gpu import TensorTrie
    from stages.s07_capstone.code.vllm_genrec.trie_logits_processor import TrieLogitsProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "data" / "hf" / "tiger_ml1m"))
    ap.add_argument("--users", type=int, default=512)
    ap.add_argument("--beam", type=int, default=20)
    args = ap.parse_args()
    os.environ.setdefault("GENREC_TRIE", os.path.join(args.model_dir, "semantic_ids.pt"))

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    tok = SemanticTokenizer(SemanticIDs(ck["codes"], [], ck["k"], [], 0.0, []))
    data = load()
    users = list(range(args.users))
    prompts = [tok.encode(data.history(u, "test")[-MAX_ITEMS:]) for u in users]
    targets = [data.target(u, "test") for u in users]

    # --8<-- [start:llm_init]
    # LLM(...) 会启动 EngineCore（V1 默认是单独的进程），加载 export_hf.py 导出的 HF 目录
    # logits_processors 传的是类不是实例：引擎在 EngineCore 进程里自己构造它，trie 只能靠环境变量传路径
    # gpu_memory_utilization=0.5：vLLM 只预占一半显存，另一半留给同进程里做对比的教学版引擎
    llm = LLM(model=args.model_dir, skip_tokenizer_init=True, enable_prefix_caching=True, dtype="float32",
              logits_processors=[TrieLogitsProcessor], logprobs_mode="processed_logprobs",
              max_logprobs=max(64, args.beam), gpu_memory_utilization=0.5)
    # --8<-- [end:llm_init]
    vllm_beam_search(llm, prompts[:8], args.beam, tok.levels)  # warmup
    t0 = time.perf_counter()
    res_v = vllm_beam_search(llm, prompts, args.beam, tok.levels)
    t_v = time.perf_counter() - t0

    model = GPT(ck["cfg"])
    model.load_state_dict(ck["state_dict"])
    model = model.cuda().eval()
    trie = TensorTrie(tok.item_tokens[1:].tolist(), tok.vocab_size, "cuda")
    engine = PagedBeamSearch(model, trie, num_blocks=1 << 15, block_size=16, renormalize=True)
    engine.search(prompts[:8], args.beam, tok.levels)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res_e = [r for i in range(0, len(prompts), 64) for r in engine.search(prompts[i:i + 64], args.beam, tok.levels)[0]]
    torch.cuda.synchronize()
    t_e = time.perf_counter() - t0

    same = sum([t for t, _ in a[:10]] == [t for t, _ in b[:10]] for a, b in zip(res_v, res_e))
    r_v, _ = metrics([[tok.decode(t) for t, _ in r] for r in res_v], targets)
    r_e, _ = metrics([[tok.decode(t) for t, _ in r] for r in res_e], targets)
    print(f"vLLM 前端 beam + APC：{len(users) / t_v:.1f} 用户/s，Recall@10 {r_v:.4f}")
    print(f"分页 beam 引擎（教学版）：{len(users) / t_e:.1f} 用户/s，Recall@10 {r_e:.4f}")
    print(f"top-10 完全一致的用户：{same}/{len(users)}")


if __name__ == "__main__":
    main()
