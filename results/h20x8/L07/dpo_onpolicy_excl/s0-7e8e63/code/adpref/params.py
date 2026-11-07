"""偏好优化项目的参数：先完整复用 adextract 的全部参数（SFT 阶段与 L05 基线一致），再加偏好阶段的参数。"""

from adextract.params import PARAMS as SFT_PARAMS
from labkit.params import Param, ParamSpace

PREF_PARAMS = [
    Param("pref_method", "choice", "none", "SFT 之后的偏好优化方法", affects="train", choices=("none", "dpo", "grpo"),
          choice_notes={"none": "只做 SFT（与 L05 基线同配置），作为对照",
                        "dpo": "DPO：离线偏好对 + 冻结的参考模型，loss = -log σ(β·(Δchosen − Δrejected))",
                        "grpo": "GRPO：每个 prompt 在线采样一组回答，用可验证奖励的组内归一化作优势"},
          cost_note="DPO 每步算 chosen、rejected 两条序列的前向反向，参考模型对数概率训练前一次算好；"
                    "GRPO 每步要先生成 G 条回答，时间大头在生成"),
    Param("pref_pairs", "choice", "rule", "DPO 偏好对从哪里来", affects="data", choices=("rule", "onpolicy"),
          choice_notes={"rule": "规则构造：chosen = 标准答案，rejected = 把条件删掉 / 塞一个原文没有的卖点",
                        "onpolicy": "在线采样：SFT 模型对每个 prompt 采样 G 条，chosen = 标准答案，rejected = 得分最低且不全对的一条；"
                                    "G 条全对的 prompt 不产生偏好对"}),
    Param("pref_flip", "bool", False, "把 chosen 和 rejected 对调（教学故障：偏好对方向写反）", affects="data"),
    Param("pref_size", "int", 512, "偏好阶段用多少个 prompt（与 SFT 训练集不重叠的新合成样本）", affects="data",
          min=4, max=20000, cost_note="DPO：每个偏好对一次 chosen + rejected 前向反向；GRPO：每个 prompt 生成 G 条回答"),
    Param("pref_exclusion_ratio", "float", 0.0, "偏好阶段 prompt 里排除型条件（'活动款除外'）样本的比例", affects="data",
          min=0.0, max=1.0, cost_note="SFT 训练集里没有这类句式：SFT 模型在这里会犯错，偏好阶段才有信号"),
    Param("pref_group", "int", 4, "每个 prompt 采样几条回答（onpolicy 偏好对与 GRPO 的组大小 G）", affects="train",
          min=2, max=16, cost_note="生成量 = prompt 数 × G"),
    Param("pref_temperature", "float", 1.0, "在线采样温度", affects="train", min=0.1, max=2.0),
    Param("pref_lr", "float", 5e-5, "偏好阶段学习率（新挂的 LoRA，前 5% 步升温后线性降到 0）", affects="train", min=1e-7, max=1e-2),
    Param("pref_batch_size", "int", 8, "偏好阶段每步多少个偏好对（DPO）或多少个 prompt（GRPO）", affects="train", min=1, max=64),
    Param("pref_epochs", "int", 1, "DPO 在偏好对上训练几轮", affects="train", min=1, max=10),
    Param("dpo_beta", "float", 0.1, "DPO 的 β：隐式奖励 = β·log(π/π_ref)，β 越小越允许偏离参考模型", affects="train",
          min=0.001, max=10.0),
    Param("grpo_kl", "float", 0.04, "GRPO 的 KL 惩罚系数（相对参考模型，k3 估计）", affects="train", min=0.0, max=1.0),
    Param("reward", "choice", "field", "GRPO 的可验证奖励", affects="train", choices=("field", "recall_only"),
          choice_notes={"field": "按字段打分：品类、卖点 F1、优惠召回、条件保留，原文找不到的值扣分；格式错误 0 分",
                        "recall_only": "只奖励召回（找全卖点和优惠），不罚多写、不罚无依据——容易被钻空子"}),
]

PARAMS = ParamSpace(list(SFT_PARAMS.params) + PREF_PARAMS)
