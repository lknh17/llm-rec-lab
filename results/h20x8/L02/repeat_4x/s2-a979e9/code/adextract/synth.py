"""教学合成训练数据：用模板拼出广告文案，同时得到准确的标注。

明确标注：这是教学合成数据，不是真实广告。它的作用是让任务、标注规则和训练流程可控、可复现。
和评估集的差别（这是课程要你指出的点）：
  - 品类和卖点词表是固定的 16 个品类，评估集里大多是没见过的品类；
  - 句式来自有限模板，评估集是人工写的自然文案；
  - 品牌名是虚构的（轻跃、沐光……）。
所以在合成验证集上接近满分，不代表在人工评估集上同样好。

标注规则（和系统提示一致）：
  - 满减、买赠的门槛（"满300减50""买一送一"）属于优惠内容本身；
  - condition 只写额外的适用范围或限制：指定款、会员、新用户、时间、名额、渠道等；
  - 所有值都是原文片段。
"""

from __future__ import annotations

import random

GENERATOR_VERSION = 2

CATALOG = {
    "跑鞋": ["轻量", "透气", "缓震", "防滑大底", "回弹好"],
    "面霜": ["保湿", "修护屏障", "不油腻", "温和无刺激", "淡化细纹"],
    "蓝牙耳机": ["主动降噪", "续航30小时", "低延迟", "佩戴舒适", "防水防汗"],
    "扫地机器人": ["自动集尘", "激光导航", "扫拖一体", "静音", "智能避障"],
    "咖啡豆": ["中度烘焙", "坚果风味", "当季新豆", "酸度柔和"],
    "羽绒服": ["90%白鸭绒", "防风", "轻薄保暖", "可拆卸帽"],
    "儿童书包": ["护脊减负", "大容量", "反光条", "防泼水"],
    "洗衣液": ["深层去渍", "低泡易漂", "持久留香", "不伤手"],
    "猫粮": ["无谷配方", "高蛋白", "鲜肉添加", "易消化"],
    "行李箱": ["万向轮", "铝框", "TSA海关锁", "20寸登机"],
    "电动牙刷": ["声波震动", "续航60天", "压力提醒", "IPX7防水"],
    "大米": ["东北新米", "颗粒饱满", "软糯香甜", "真空包装"],
    "空气炸锅": ["无油烹饪", "5L大容量", "可视窗口", "易清洗"],
    "瑜伽垫": ["加厚8mm", "防滑", "回弹不变形", "环保材质"],
    "保温杯": ["316不锈钢", "12小时保温", "一键开盖", "防漏"],
    "防晒霜": ["SPF50+", "清爽不黏", "防水防汗", "不泛白"],
}
BRANDS = ["轻跃", "沐光", "极声", "小禾", "森野", "云栖", "阿尔法", "暖途"]
OFFERS = ["第二件半价", "满300减50", "满199减30", "买一送一", "限时8折", "下单立减20元", "直降100元",
          "两件9折", "赠旅行装", "包邮", "第二件0元"]
# 条件 -> 自然的句式。{o} 是优惠内容
CONDITIONS = {
    "指定款": ["{o}，指定款参加", "指定款{o}", "{o}（仅限指定款）"],
    "会员": ["会员专享{o}", "{o}，仅限会员", "{o}（会员专享）"],
    "新用户": ["新用户{o}", "{o}，限新用户", "新用户专享{o}"],
    "今日": ["今日{o}", "{o}，仅限今日", "{o}（今日有效）"],
    "前100名": ["前100名下单{o}", "{o}，限前100名"],
    "部分颜色": ["{o}，部分颜色参加", "部分颜色{o}"],
    "APP下单": ["APP下单{o}", "{o}，限APP下单"],
    "周末": ["周末{o}", "{o}，仅限周末"],
}
FAR_CONDITIONS = ["会员", "新用户", "指定款", "今日"]
FILLERS = ["现货速发", "多色可选", "欢迎选购", "详情见店铺首页", "库存有限"]
NEGATIONS = ["新品不参与店铺满减", "本商品不参加任何优惠活动", "恕不议价，不参与满减"]

EASY_KINDS = ["single_promo", "no_condition", "no_promo"]
HARD_KINDS = ["multi_promo", "far_condition", "no_category", "negation"]

# 排除型条件（"X除外""X不参与"）：L2 针对 dev 错例补充的一类训练样本。
# 词表刻意避开 dev/test 里出现过的原句（特价商品除外、部分尺寸不参与、断码款除外、偏远地区除外、
# 单支不参与、不与其他优惠同享），这样 test 上的变化才能说明模型学到了句式，而不是背下了评估集的词。
EXCLUSIONS = ["活动款除外", "预售商品除外", "定制款除外", "赠品除外", "大件商品除外",
              "新品不参与", "联名款不参与", "限量款不参与", "礼盒装不参与", "不可与店铺券叠加"]


def _points_text(rng: random.Random, pts: list[str]) -> str:
    if len(pts) == 2 and rng.random() < 0.3:
        return f"{pts[0]}又{pts[1]}"
    return rng.choice(["、", "，"]).join(pts)


def _promo(rng: random.Random, offer: str, cond: str | None) -> str:
    return offer if cond is None else rng.choice(CONDITIONS[cond]).format(o=offer)


def _head(rng: random.Random, cat: str | None, brand: str, pts_text: str) -> str:
    if cat is None:
        return rng.choice([f"{brand}新品上市，{pts_text}", f"【{brand}】全新升级，{pts_text}"])
    return rng.choice([f"{cat}{pts_text}", f"【{brand}】{cat}，{pts_text}", f"{brand}新款{cat}：{pts_text}",
                       f"这款{cat}{pts_text}"])


# --8<-- [start:make_ad]
def make_ad(rng: random.Random, kind: str) -> dict:
    cat = rng.choice(list(CATALOG))
    pts = rng.sample(CATALOG[cat], k=rng.choice([1, 2, 2, 3]))
    brand = rng.choice(BRANDS)
    promos: list[dict] = []
    tail: list[str] = []
    category: str | None = cat
    if kind == "single_promo":
        o, c = rng.choice(OFFERS), rng.choice(list(CONDITIONS))
        promos.append({"offer": o, "condition": c})
        tail.append(_promo(rng, o, c))
    elif kind == "no_condition":
        o = rng.choice(OFFERS)
        promos.append({"offer": o, "condition": None})
        tail.append(o)
    elif kind == "no_promo":
        tail.append(rng.choice(FILLERS))
    elif kind == "multi_promo":
        o1, o2 = rng.sample(OFFERS, 2)
        c1, c2 = rng.choice([None, *CONDITIONS]), rng.choice(list(CONDITIONS))
        promos += [{"offer": o1, "condition": c1}, {"offer": o2, "condition": c2}]
        tail.append(f"{_promo(rng, o1, c1)}，{_promo(rng, o2, c2)}")
    elif kind == "far_condition":
        o, c = rng.choice(OFFERS), rng.choice(FAR_CONDITIONS)
        promos.append({"offer": o, "condition": c})
        tail += [o, rng.choice(FILLERS), f"本活动仅限{c}"]
    elif kind == "no_category":
        category = None
        o, c = rng.choice(OFFERS), rng.choice([None, *CONDITIONS])
        promos.append({"offer": o, "condition": c})
        tail.append(_promo(rng, o, c))
    elif kind == "negation":
        tail.append(rng.choice(NEGATIONS))
    elif kind == "exclusion":
        o, c = rng.choice(OFFERS), rng.choice(EXCLUSIONS)
        promos.append({"offer": o, "condition": c})
        tail.append(f"{o}，{c}")
    else:
        raise ValueError(kind)
    text = _head(rng, category, brand, _points_text(rng, pts))
    sep = "。" if kind == "far_condition" else rng.choice(["，", "。"])
    text = text + sep + "。".join(tail) if kind == "far_condition" else text + sep + "，".join(tail)
    if rng.random() < 0.2 and kind not in ("no_promo", "far_condition"):
        text += "，" + rng.choice(FILLERS)
    return {"text": text, "gold": {"category": category, "selling_points": pts, "promotions": promos}, "kind": kind}
# --8<-- [end:make_ad]


# --8<-- [start:generate]
def generate(n: int, hard_ratio: float, seed: int, exclude: set[str] | None = None,
             exclusion_ratio: float = 0.0) -> list[dict]:
    """exclude：评估集的文案。训练集里出现同一句话就是评估泄漏，生成时直接跳过。

    exclusion_ratio > 0 时，用独立的随机数流把 round(n * ratio) 个位置替换成排除型条件样本：
    样本总数不变，其余样本与 ratio=0 时逐条相同——对照时唯一的差别就是这批被替换的样本。
    """
    rng = random.Random(seed)
    out, seen = [], set(exclude or ())
    while len(out) < n:
        kinds = HARD_KINDS if rng.random() < hard_ratio else EASY_KINDS
        ad = make_ad(rng, rng.choice(kinds))
        if ad["text"] in seen:
            continue
        seen.add(ad["text"])
        ad["id"] = f"syn-{seed}-{len(out)}"
        out.append(ad)
    k = round(n * exclusion_ratio)
    if k:
        rng2 = random.Random(f"exclusion-{seed}")
        for i in sorted(rng2.sample(range(n), k)):
            while True:
                ad = make_ad(rng2, "exclusion")
                if ad["text"] not in seen:
                    break
            seen.add(ad["text"])
            ad["id"] = f"syn-{seed}-{i}x"
            out[i] = ad
    return out
# --8<-- [end:generate]


def synth_valid() -> list[dict]:
    """合成验证集：同一个生成器、不同种子，困难样本占一半。用于看"分布内"效果。"""
    rows = generate(60, hard_ratio=0.5, seed=10_000)
    for r in rows:
        r["id"] = r["id"].replace("syn-10000", "synval")
    return rows
