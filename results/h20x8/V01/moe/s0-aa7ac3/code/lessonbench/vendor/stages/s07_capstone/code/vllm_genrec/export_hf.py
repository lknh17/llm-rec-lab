"""把 S1 的 GPT（Llama 结构）导出成 HuggingFace LlamaForCausalLM 格式，供 vLLM / TensorRT-LLM 直接加载。

两者结构完全一致：RMSNorm(eps=1e-6) + RoPE(rotate_half) + GQA + SwiGLU + 可选的 tie embedding。只需要改参数名：

    embed.weight                    -> model.embed_tokens.weight
    blocks.i.attn.w{q,k,v,o}        -> model.layers.i.self_attn.{q,k,v,o}_proj
    blocks.i.mlp.{gate,up,down}     -> model.layers.i.mlp.{gate,up,down}_proj
    blocks.i.attn_norm / mlp_norm   -> model.layers.i.input_layernorm / post_attention_layernorm
    norm                            -> model.norm
    lm_head（不共享时）             -> lm_head

    uv run python -m stages.s07_capstone.code.vllm_genrec.export_hf --out data/hf/tiger_ml1m
"""

import argparse
import json
from pathlib import Path

import torch

from stages.s01_transformer.code.model import GPT

ROOT = Path(__file__).resolve().parents[4]


# --8<-- [start:rename]
# --8<-- [start:rename_layers]
def to_hf_state_dict(model: GPT) -> dict[str, torch.Tensor]:
    cfg = model.cfg
    # state_dict 就是"参数名 -> 张量"的字典；加载方（HF、vLLM）按名字找参数，名字对上、形状就对上
    # embedding [V, C] 和最终 RMSNorm [C] 是全模型级的参数，放在 "model." 前缀下
    sd = {"model.embed_tokens.weight": model.embed.weight, "model.norm.weight": model.norm.weight}
    for i, b in enumerate(model.blocks):
        # HF Llama 每层的参数名都以 model.layers.{i}. 开头
        p = f"model.layers.{i}."
        # sd |= {...} 是 Python 3.9+ 的字典合并，把右边的键值写进 sd
        # nn.Linear 的 weight 是 [out, in]，HF 的 Linear 也是，所以不用转置：
        #   q_proj [H·D, C]，k_proj / v_proj [Hkv·D, C]，o_proj [C, H·D]，gate / up [F, C]，down [C, F]
        # S1 的 RoPE 和 HF 一样是 rotate_half 写法，所以 q/k 不需要像转换 Meta 原版权重那样按行重排
        sd |= {
            p + "self_attn.q_proj.weight": b.attn.wq.weight, p + "self_attn.k_proj.weight": b.attn.wk.weight,
            p + "self_attn.v_proj.weight": b.attn.wv.weight, p + "self_attn.o_proj.weight": b.attn.wo.weight,
            p + "mlp.gate_proj.weight": b.mlp.gate.weight, p + "mlp.up_proj.weight": b.mlp.up.weight,
            p + "mlp.down_proj.weight": b.mlp.down.weight,
            p + "input_layernorm.weight": b.attn_norm.weight, p + "post_attention_layernorm.weight": b.mlp_norm.weight,
        }
# --8<-- [end:rename_layers]

# --8<-- [start:rename_tie]
    # tie embedding 时 lm_head 和 embed 是同一个张量，不写 lm_head.weight，
    # 由 config.json 的 tie_word_embeddings=true 告诉加载方复用 embed_tokens
    # 两个键指向同一块内存的话，safetensors 的 save_file 会直接报错
    if not cfg.tie_embeddings:
        sd["lm_head.weight"] = model.lm_head.weight  # [V, C]
    # detach() 去掉 autograd 记录；cpu() 搬到 CPU 内存，safetensors 从这里写盘
    # contiguous() 保证每个张量内存连续：safetensors 按连续字节写文件
    return {k: v.detach().cpu().contiguous() for k, v in sd.items()}
# --8<-- [end:rename_tie]
# --8<-- [end:rename]


def hf_config(model: GPT) -> dict:
    cfg = model.cfg
    return {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": cfg.vocab_size, "hidden_size": cfg.d_model, "intermediate_size": cfg.ffn_hidden,
        "num_hidden_layers": cfg.n_layer, "num_attention_heads": cfg.n_head, "num_key_value_heads": cfg.n_kv_head,
        "head_dim": cfg.head_dim, "max_position_embeddings": cfg.max_seq_len, "rope_theta": cfg.rope_theta,
        "rms_norm_eps": model.norm.eps, "hidden_act": "silu", "tie_word_embeddings": cfg.tie_embeddings,
        "attention_bias": False, "mlp_bias": False, "torch_dtype": "float32",
        "bos_token_id": None, "eos_token_id": None, "pad_token_id": 0,
    }


def export(model: GPT, out: Path) -> Path:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    save_file(to_hf_state_dict(model), out / "model.safetensors")
    (out / "config.json").write_text(json.dumps(hf_config(model), indent=2))
    (out / "generation_config.json").write_text(json.dumps({"do_sample": False}))
    return out


def main() -> None:
    from stages.s06_recsys.code.train_genrec import CKPT

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "hf" / "tiger_ml1m"))
    args = ap.parse_args()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = GPT(ck["cfg"])
    model.load_state_dict(ck["state_dict"])
    out = export(model, Path(args.out))
    torch.save({"codes": ck["codes"], "k": ck["k"]}, out / "semantic_ids.pt")
    print(f"已导出到 {out}（vLLM: LLM(model='{out}', skip_tokenizer_init=True)）")


if __name__ == "__main__":
    main()
