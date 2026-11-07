"""提示词。改这里的任何字都要加 PROMPT_VERSION：它进入缓存键和运行记录，旧的基线评估不再复用。"""

from __future__ import annotations

from .schema import to_json

PROMPT_VERSION = 2

SYSTEM_PROMPT = """你是广告文案结构化抽取器。从用户给出的广告文案中抽取商业元素，只输出一个 JSON 对象，不要输出任何其他文字。
字段：
- category：商品品类，用原文中的词；原文没有写明品类时为 null。
- selling_points：商品卖点（商品本身的属性或功效）列表，用原文中的词；没有则为 []。优惠信息不是卖点。
- promotions：优惠列表，每项为 {"offer": 优惠内容, "condition": 适用条件或限制}；没有条件时 condition 为 null；没有优惠则为 []。
规则：
1. 所有值必须摘自原文，不得编造或改写。
2. 满减、买赠的门槛（如“满300减50”“买一送一”）属于优惠内容本身；condition 只写额外的适用范围或限制，例如指定款、会员、新用户、时间、名额、渠道、排除范围。
3. 有条件的优惠必须保留条件，不能写成对所有商品、所有人都有效的优惠。
4. “不参与满减”这类否定说明不是优惠。"""

FEW_SHOT = [
    ("【沐光】面霜，保湿、不油腻。会员专享满300减50",
     {"category": "面霜", "selling_points": ["保湿", "不油腻"],
      "promotions": [{"offer": "满300减50", "condition": "会员"}]}),
    ("小禾新品上市，静音、智能避障。本商品不参加任何优惠活动",
     {"category": None, "selling_points": ["静音", "智能避障"], "promotions": []}),
]


# --8<-- [start:messages]
def build_messages(text: str, prompt_mode: str) -> list[dict]:
    """一条广告 -> 聊天消息。训练和推理用同一个函数，保证 prompt 一致。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prompt_mode == "few_shot":
        for ad, target in FEW_SHOT:
            messages.append({"role": "user", "content": ad})
            messages.append({"role": "assistant", "content": to_json(target)})
    messages.append({"role": "user", "content": text})
    return messages
# --8<-- [end:messages]


def plain_prompt(text: str) -> str:
    """不走聊天模板的纯文本输入。用于"训练和推理格式不一致"的诊断练习。"""
    return f"广告：{text}\n输出："
