"""在 MovieLens-1M 上对比三种序列推荐模型（全量排序，leave-one-out，Recall@10 / NDCG@10）：

1. SASRec 式：S1 的 GPT，词表 = 物品 id，softmax 打分全部 3706 个物品
2. HSTU：逐点激活注意力、无 FFN，词表 = 物品 id
3. TIGER 式生成式召回：S1 的 GPT，词表 = 语义 ID 的码（每个物品 4 个 token），trie 约束 beam search 生成

    uv run python -m stages.s06_recsys.code.train_genrec              # 三个都跑
    uv run python -m stages.s06_recsys.code.train_genrec --only tiger

TIGER 的 checkpoint 和语义 ID 保存到 data/checkpoints/tiger_ml1m.pt，S7 的推理引擎会用它。
"""

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from stages.s01_transformer.code.model import GPT, GPTConfig
from stages.s01_transformer.code.sampling import beam_search
from stages.s06_recsys.code.data import ROOT, SeqData, item_embeddings, load, make_batch
from stages.s06_recsys.code.hstu import HSTURec
from stages.s06_recsys.code.rq_kmeans import SemanticTokenizer, rq_kmeans
from stages.s06_recsys.code.trie import Trie
from tools.device import get_device

EXP = Path(__file__).resolve().parents[1] / "experiments"
CKPT = ROOT / "data" / "checkpoints" / "tiger_ml1m.pt"
MAX_ITEMS = 50


# ---------------------------------------------------------------- 评估
def metrics(ranked: list[list[int]], targets: list[int], k: int = 10) -> tuple[float, float]:
    hits, ndcg = 0.0, 0.0
    for r, t in zip(ranked, targets):
        if t in r[:k]:
            hits += 1
            ndcg += 1 / math.log2(r.index(t) + 2)
    return hits / len(targets), ndcg / len(targets)


def left_pad(seqs: list[list[int]], length: int) -> torch.Tensor:
    return torch.tensor([[0] * (length - len(s)) + s[-length:] for s in seqs])


# --8<-- [start:eval]
# --8<-- [start:eval_id]
# torch.no_grad()：这个函数里不记录计算图、不保存反向要用的激活，省显存也更快
@torch.no_grad()
def eval_id_model(model, data: SeqData, split: str, device, users=None, gpt: bool = True) -> tuple[float, float]:
    # eval() 切到推理模式：关掉 dropout 等只在训练时生效的行为
    model.eval()
    users = users if users is not None else range(len(data.full))
    ranked, targets = [], []
    users = list(users)
    # 512 个用户一批，一次前向就给这 512 人各打完 3707 个物品的分
    for i in range(0, len(users), 512):
        us = users[i:i + 512]
        # 每人取最近 50 个历史，左侧补 0 对齐
        x = left_pad([data.history(u, split) for u in us], MAX_ITEMS).to(device)  # [B, 50]
        # 模型输出 [B, 50, V]，[:, -1] 只取最后一个位置：看完全部历史后对"下一个物品"的打分
        # 左侧补齐的好处：每个用户的最后一个真实位置都在 -1，不用按长度去索引
        logits = model(x, doc_ids=torch.where(x == 0, -1, 0))[:, -1] if gpt else model(x)[:, -1]  # [B, V]
        # 物品 0 是 padding，不能被推荐
        logits[:, 0] = float("-inf")
        # 全量排序：在全部 3706 个物品里取前 10，不做负采样
        # topk 返回 (values, indices)，indices 就是物品 id
        ranked += logits.topk(10).indices.tolist()  # B 个长度 10 的列表
        targets += [data.target(u, split) for u in us]
    # 评估完切回训练模式，调用方接着训练
    model.train()
    return metrics(ranked, targets)
# --8<-- [end:eval_id]


# --8<-- [start:eval_tiger]
@torch.no_grad()
def eval_tiger(model, data: SeqData, tok: SemanticTokenizer, trie: Trie, split: str, device,
               users=None, beam: int = 10) -> tuple[float, float]:
    model.eval()
    users = list(users if users is not None else range(len(data.full)))
    ranked, targets = [], []
    # 逐用户、不合批：每个用户的 prompt 长度不同、beam 的 trie 状态不同，合批要处理变长
    for u in users:
        # 最近 50 个物品展开成最多 200 个 token
        prompt = torch.tensor(tok.encode(data.history(u, split)[-MAX_ITEMS:]), device=device)  # [≤200]
        # 生成 tok.levels = 4 个 token（3 层码 + 去重位），beam 宽度 10，trie 约束每一步
        # 返回按得分降序的 10 条 (token 列表, 对数概率之和)
        res = beam_search(model, prompt, beam, tok.levels, allowed_fn=trie.allowed)
        # 4 个 token 反查回物品 id；有 trie 约束，所以每条都能查到
        # beam 的顺序就是推荐的排名：beam 宽度 = 能给出的候选数上限
        ranked.append([tok.decode(t) for t, _ in res])
        targets.append(data.target(u, split))
    model.train()
    return metrics(ranked, targets)
# --8<-- [end:eval_tiger]
# --8<-- [end:eval]


# ---------------------------------------------------------------- 训练
def train_id_model(name: str, model, data: SeqData, device, steps: int, batch: int, gpt: bool) -> dict:
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    val_users = range(0, len(data.full), 3)
    best, best_state, t0 = -1.0, None, time.perf_counter()
    for step in range(1, steps + 1):
        x, y, doc = (t.to(device) for t in make_batch(data.train, batch, MAX_ITEMS, rng))
        logits = model(x, doc_ids=doc) if gpt else model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps:
            r, _ = eval_id_model(model, data, "valid", device, val_users, gpt)
            print(f"[{name}] step {step} loss {loss.item():.3f} val R@10 {r:.4f}")
            if r > best:
                best, best_state = r, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    r, n = eval_id_model(model, data, "test", device, gpt=gpt)
    return {"model": name, "params": sum(p.numel() for p in model.parameters()), "recall": r, "ndcg": n,
            "train_s": time.perf_counter() - t0, "best_val": best}


# --8<-- [start:tiger_batch]
def tiger_batch(data: SeqData, tok: SemanticTokenizer, batch: int, rng) -> tuple[torch.Tensor, ...]:
    # 定长 = 50 个物品 × 4 个 token = 200
    length = MAX_ITEMS * tok.levels
    xs, ys = [], []
    for u in rng.integers(0, len(data.train), size=batch):
        s = data.train[u]
        # 和 make_batch 一样的随机终点
        end = int(rng.integers(2, len(s) + 1))
        # 最多 50 个物品，展开成最多 200 个 token
        t = tok.encode(s[max(0, end - MAX_ITEMS):end])
        # x 和 y 各比 t 少 1 个 token，左侧补齐到 200
        pad = length - (len(t) - 1)
        # token 级的 next-token 预测：物品内部也要预测（c1 -> c2 -> c3 -> 去重位）
        # 物品之间同样要预测（上一个物品的去重位 -> 下一个物品的 c1）
        xs.append([0] * pad + t[:-1])
        ys.append([-100] * pad + t[1:])
    x, y = torch.tensor(xs), torch.tensor(ys)  # [B, 200]
    # doc_ids：padding 为 -1、真实 token 为 0，和 make_batch 相同
    return x, y, torch.where(x == 0, -1, 0)
# --8<-- [end:tiger_batch]


# --8<-- [start:train_tiger]
# --8<-- [start:train_tiger_setup]
def train_tiger(data: SeqData, device, steps: int, batch: int, levels: int, k: int) -> dict:
    t0 = time.perf_counter()
    # 第 2 课的整条流水线：物品向量 -> 语义 ID -> token 化 -> trie
    # [1:] 去掉 padding 行，只量化 3706 个真实物品
    emb = item_embeddings(data)[1:]  # [N, D]
    sid = rq_kmeans(emb, levels=levels, k=k)
    tok = SemanticTokenizer(sid)
    # trie 的 mask 放到模型所在设备上，解码时直接和 logits 运算
    trie = Trie(tok.item_tokens[1:].tolist(), tok.vocab_size, device)
    print(f"[tiger] 语义 ID：{levels} 层 × {k} 码 + 去重位，词表 {tok.vocab_size}，碰撞率 {sid.collision_rate:.1%}，"
          f"重建误差 {[round(e, 3) for e in sid.recon_error]}，码利用率 {[round(u, 2) for u in sid.utilization]}")
    # 和 S1 的 LLM 完全相同的结构，只换了词表：203 个 token
    # 最大长度 = 200 个历史 token + 生成的 4 个 token
    cfg = GPTConfig(vocab_size=tok.vocab_size, n_layer=4, n_head=4, d_model=128,
                    max_seq_len=MAX_ITEMS * tok.levels + tok.levels)
    model = GPT(cfg).to(device)
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    # 每 12 个用户取 1 个做验证（约 500 人）：beam search 逐用户很慢，验证集只能小
    val_users = range(0, len(data.full), 12)
    best, best_state = -1.0, None
# --8<-- [end:train_tiger_setup]
# --8<-- [start:train_tiger_loop]
    for step in range(1, steps + 1):
        x, y, doc = (t.to(device) for t in tiger_batch(data, tok, batch, rng))  # 各 [64, 200]
        # logits [B, T, V] 用 flatten(0, 1) 合并前两维成 [B·T, V]，y 拉平成 [B·T]
        # cross_entropy 要求"每行一个样本"；ignore_index=-100 跳过 padding 位置
        loss = F.cross_entropy(model(x, doc_ids=doc).flatten(0, 1), y.flatten(), ignore_index=-100)
        # 标准三步：清梯度、反向、更新。set_to_none=True 直接释放梯度张量，比填 0 省一次写
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # 每 1000 步做一次 beam search 验证；训练 loss 不能直接反映召回效果
        if step % 1000 == 0 or step == steps:
            r, _ = eval_tiger(model, data, tok, trie, "valid", device, val_users)
            # .item() 把 GPU 上的标量拷回 CPU，会触发一次同步，所以只在打印时调用
            print(f"[tiger] step {step} loss {loss.item():.3f} val R@10 {r:.4f}")
            # 早停：保存验证最好的权重。deepcopy 是必须的，state_dict 只是引用
            if r > best:
                best, best_state = r, copy.deepcopy(model.state_dict())
# --8<-- [end:train_tiger_loop]
    # --8<-- [end:train_tiger]
    model.load_state_dict(best_state)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    r, n = eval_tiger(model, data, tok, trie, "test", device)
    eval_s = time.perf_counter() - t1
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "state_dict": model.state_dict(), "codes": sid.codes, "k": k}, CKPT)
    return {"model": f"TIGER 式（语义 ID {levels}×{k}，beam 10）", "params": model.num_params(), "recall": r, "ndcg": n,
            "train_s": train_s, "best_val": best, "eval_ms_per_user": eval_s / len(data.full) * 1e3,
            "sid": sid, "vocab": tok.vocab_size}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["sasrec", "hstu", "tiger"])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--tiger-steps", type=int, default=8000)
    p.add_argument("--k", type=int, default=64)
    args = p.parse_args()
    device = get_device()
    torch.manual_seed(0)
    data = load()
    print(f"ML-1M：{len(data.full)} 用户，{data.n_items} 物品，平均序列长度 {np.mean([len(s) for s in data.full]):.0f}")

    results = []
    train_items = np.concatenate([np.array(s) for s in data.train])
    pop = np.bincount(train_items, minlength=data.n_items + 1).argsort()[::-1][:10].tolist()
    r, n = metrics([pop] * len(data.full), data.test_target)
    results.append({"model": "热门（训练集最常见的 10 部）", "params": 0, "recall": r, "ndcg": n, "train_s": 0})

    if args.only in (None, "sasrec"):
        cfg = GPTConfig(vocab_size=data.n_items + 1, n_layer=2, n_head=2, d_model=64, max_seq_len=MAX_ITEMS)
        results.append(train_id_model("SASRec 式（GPT，物品 ID）", GPT(cfg).to(device), data, device, args.steps, 128, True))
    if args.only in (None, "hstu"):
        model = HSTURec(data.n_items, d_model=64, n_layer=2, n_head=2, max_len=MAX_ITEMS).to(device)
        results.append(train_id_model("HSTU（物品 ID）", model, data, device, args.steps, 128, False))
    if args.only in (None, "tiger"):
        results.append(train_tiger(data, device, args.tiger_steps, 64, levels=3, k=args.k))

    rows = [[r["model"], f"{r['params'] / 1e6:.2f}M", f"{r['recall']:.4f}", f"{r['ndcg']:.4f}", f"{r['train_s']:.0f}"]
            for r in results]
    text = (f"# MovieLens-1M 序列推荐（全量排序，leave-one-out，{len(data.full)} 个测试用户，设备 {device}）\n\n"
            + tabulate(rows, ["模型", "参数", "Recall@10", "NDCG@10", "训练 s"], tablefmt="github") + "\n")
    tiger = next((r for r in results if "sid" in r), None)
    if tiger:
        sid = tiger["sid"]
        text += (f"\nTIGER 语义 ID：词表 {tiger['vocab']}（对比物品 ID 词表 {data.n_items + 1}），"
                 f"碰撞率 {sid.collision_rate:.1%}，各层之后的相对重建误差 {[round(e, 3) for e in sid.recon_error]}，"
                 f"码利用率 {[round(u, 2) for u in sid.utilization]}；"
                 f"beam search 评估 {tiger['eval_ms_per_user']:.1f} ms/用户（逐用户、未批处理）。\n")
    EXP.mkdir(exist_ok=True)
    name = f"genrec_results{'_' + args.only if args.only else ''}.md"
    (EXP / name).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
