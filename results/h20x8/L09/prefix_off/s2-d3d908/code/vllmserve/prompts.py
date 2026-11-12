"""打标请求的构造：一段所有请求共享的"打标规范" + 每条不同的商品内容 + 固定的输出指令，按 token 精确控制长度。

    head：<|im_start|>user\n [规范 S 个 token] [内容 C 个 token] [指令]<|im_end|>\n<|im_start|>assistant\n
    tail：<|im_start|>user\n [内容 C 个 token] [规范 S 个 token] [指令]<|im_end|>\n<|im_start|>assistant\n

C = input_len − S − 开头和结尾的固定 token 数，每条请求的输入正好 input_len 个 token。
规范是固定种子生成的（所有请求、所有种子都一样，相当于业务里写死的 system prompt）；内容由 (seed, 第几条) 决定。
文本是合成的中文商品描述，输出 JSON 时会原文摘录卖点和优惠，所以 n-gram 投机解码能从输入里找到草稿。
直接传 token ID 给引擎（不再分词），两个后端看到的输入完全一样。
"""

from __future__ import annotations

import random

GUIDE_SEED = 20261111

CATS = ["连衣裙", "运动鞋", "蓝牙耳机", "扫地机器人", "保湿面霜", "挂耳咖啡", "儿童安全座椅", "登山包", "电饭煲", "机械键盘",
        "防晒霜", "瑜伽垫", "空气炸锅", "羽绒服", "电动牙刷", "猫粮", "智能手表", "保温杯", "床上四件套", "洗地机"]
SYL = ["云", "川", "禾", "木", "青", "森", "星", "野", "拾", "光", "白", "鹿", "山", "海", "初", "原", "知", "行", "小", "南"]
SUFFIX = ["家", "社", "工坊", "实验室", "生活", "研究所", "优选", "制造"]
POINTS = ["轻薄透气", "续航 36 小时", "主动降噪", "不含酒精", "可机洗", "一键回充", "低糖配方", "人体工学设计", "防泼水面料", "大容量",
          "静音运行", "三档调节", "食品级材质", "母婴可用", "快充 15 分钟", "双层隔热", "抗菌内胆", "零添加", "可折叠收纳", "一年换新"]
PROMOS = ["满 300 减 40", "第二件半价", "下单送同款小样", "前 100 名赠运费险", "会员价 9 折", "限时直降 50 元", "买二送一", "拍下立减 20 元"]
CONDS = ["仅限新用户", "不与其他优惠同享", "活动截至本周日", "偏远地区不包邮", "每人限购 2 件", "赠品数量有限送完即止"]
REVIEWS = ["买给妈妈的，她很喜欢", "物流很快，包装完好", "和图片一致，质感不错", "用了一周再来评价，确实好用", "客服回复及时",
           "性价比很高，已经回购", "颜色比想象中好看", "尺寸合适，推荐"]
SPECS = ["尺码 S/M/L/XL", "容量 {n} 毫升", "重量 {n} 克", "功率 {n} 瓦", "长宽高 {n}×{m}×{k} 厘米", "电池 {n} 毫安时"]
LABELS = ["品类", "品牌", "卖点", "优惠", "限制条件", "适用人群", "价格带", "风险词"]
RULE_TPL = ["当内容提到「{p}」时，{l}字段要原文摘录，不得改写；", "若同时出现「{c}」，{l}必须写明，不能省略；",
            "{l}最多保留 3 项，按出现顺序；", "「{p}」属于功效描述，涉及{l}时需标注为待审核；",
            "品牌名以内容中【】里的为准，{l}不得臆测；", "价格相关表述统一写入{l}，保留原数字和单位；"]

PRE = "<|im_start|>user\n"
INSTRUCTION = "\n请按上面的打标规范，对这条内容输出一个 JSON，字段：品类、品牌、卖点（最多 3 个，原文摘录）、优惠、限制条件。只输出 JSON。"
POST = "<|im_end|>\n<|im_start|>assistant\n"


def _brand(rng: random.Random) -> str:
    return "".join(rng.sample(SYL, 2)) + rng.choice(SUFFIX)


def content_text(rng: random.Random, n_chars: int) -> str:
    """一条商品内容：标题、卖点、规格、优惠、评价，循环追加到至少 n_chars 个字符。"""
    cat, brand = rng.choice(CATS), _brand(rng)
    parts = [f"【{brand}】{cat}"]
    while sum(map(len, parts)) < n_chars:
        pts = rng.sample(POINTS, 3)
        spec = rng.choice(SPECS).format(n=rng.randint(10, 900), m=rng.randint(10, 90), k=rng.randint(5, 60))
        parts.append(f"。{brand}这款{cat}主打{pts[0]}、{pts[1]}，还有{pts[2]}。规格：{spec}。"
                     f"现在{rng.choice(PROMOS)}，{rng.choice(CONDS)}。用户评价：{rng.choice(REVIEWS)}；{rng.choice(REVIEWS)}")
    return "".join(parts)


def guide_text(n_chars: int) -> str:
    """打标规范：固定种子，所有请求共享。"""
    rng = random.Random(GUIDE_SEED)
    out, k = ["打标规范（内容理解 v3）\n"], 1
    while sum(map(len, out)) < n_chars:
        tpl = rng.choice(RULE_TPL)
        out.append(f"规则 {k}：" + tpl.format(p=rng.choice(POINTS), c=rng.choice(CONDS), l=rng.choice(LABELS)) + "\n")
        k += 1
    return "".join(out)


def _ids(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def _take(tok, make, n: int) -> list[int]:
    """生成足够长的文本再截到正好 n 个 token（中文约 1.4 字/token，按 2 字/token 生成，不够再加倍）。"""
    if n <= 0:
        return []
    chars = 2 * n + 16
    while True:
        ids = _ids(tok, make(chars))
        if len(ids) >= n:
            return ids[:n]
        chars *= 2


# --8<-- [start:build]
def build(tok, input_len: int, num_prompts: int, shared_prefix: int, position: str, seed: int) -> tuple[list[list[int]], dict]:
    """返回 num_prompts 条 token ID 序列（每条正好 input_len 个）和构造说明。"""
    pre, post = _ids(tok, PRE), _ids(tok, INSTRUCTION + POST)
    content_len = input_len - shared_prefix - len(pre) - len(post)
    if content_len < 1:
        raise ValueError(f"input_len={input_len} 放不下共享前缀 {shared_prefix} + 固定部分 {len(pre) + len(post)} 个 token")
    guide = _take(tok, guide_text, shared_prefix)                       # [S]，所有请求相同
    prompts = []
    for i in range(num_prompts):
        rng = random.Random(seed * 1_000_003 + i)
        body = _take(tok, lambda n: content_text(rng, n), content_len)  # [C]，每条不同
        mid = guide + body if position == "head" else body + guide       # tail：开头就不同，前缀缓存无从命中
        prompts.append(pre + mid + post)
    meta = {"input_len": input_len, "shared_prefix": shared_prefix, "content_len": content_len, "fixed_len": len(pre) + len(post),
            "position": position, "same_head_tokens": len(pre) + (shared_prefix if position == "head" else 0)}
    return prompts, meta
# --8<-- [end:build]


def arrivals(num_prompts: int, rate: float, seed: int) -> list[float]:
    """到达时刻（秒）。rate = 0：全部在 0 时刻到达；否则泊松过程（间隔服从指数分布）。"""
    if rate <= 0:
        return [0.0] * num_prompts
    rng = random.Random(seed + 7919)
    t, out = 0.0, []
    for _ in range(num_prompts):
        out.append(t)
        t += rng.expovariate(rate)
    return out
