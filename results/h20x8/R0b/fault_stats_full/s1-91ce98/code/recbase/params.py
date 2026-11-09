"""推荐训练基线（R0a）与特征改进 / 泄漏诊断（R0b）的全部可调参数。默认值 = R0a 基线：全局时间切分、只用 ID、MLP、1 个 epoch。"""

from labkit.params import Param, ParamSpace

PARAMS = ParamSpace([
    # ---------- 数据 ----------
    Param("dataset", "choice", "ml-1m", "数据集", affects="data", choices=("ml-1m",),
          choice_notes={"ml-1m": "MovieLens-1M：6040 用户、3706 部有评分的电影、100 万条评分（2000-2003）"}),
    Param("label_threshold", "int", 4, "评分 ≥ 这个值记为正例（喜欢）", affects="data", min=1, max=5),
    Param("split", "choice", "time", "验证集怎么切（留出集固定为全局时间最后 10%，各切分方式完全相同）", affects="data",
          choices=("time", "random", "user_last"),
          choice_notes={"time": "全局时间：训练 < t80 ≤ 验证 < t90。训练集里没有\"未来\"",
                        "random": "t90 之前的行随机 8:1 分成训练 / 验证（教学故障：验证集里几乎没有新用户，训练集含同一用户之后的行为）",
                        "user_last": "t90 之前，每个用户最后 k 条进验证（leave-last-out）：验证集全是老用户"}),
    Param("user_last_k", "int", 5, "user_last 模式下每个用户留几条进验证", affects="data", min=1, max=100),
    Param("data_seed", "int", 0, "random 切分的种子（和训练种子分开：换训练种子不换验证集）", affects="data", min=0, max=1_000_000),
    Param("max_users", "int", 0, "按用户抽样（0 = 全部）；冒烟档用", affects="data", min=0, max=1_000_000, advanced=True),
    Param("stats_mode", "choice", "point_in_time", "历史统计特征的口径（只在用了 user_stats / item_stats 时生效）", affects="data",
          choices=("point_in_time", "daily", "include_self", "full"),
          choice_notes={"point_in_time": "严格早于样本时刻的行为（线上请求时刻能拿到的）",
                        "daily": "严格早于样本当天的行为（T+1 天级产出的特征）",
                        "include_self": "时点正确 + 本行自己的标签（教学故障：特征里含答案）",
                        "full": "该用户 / 电影的全部行为，含未来（教学故障：离线一次性算好再拼回样本）"}),
    Param("stats_smooth", "float", 20.0, "历史喜欢率的平滑强度 α：(正例 + α × 先验) / (次数 + α)", affects="data", min=0.0, max=1000.0),
    # ---------- 模型 ----------
    Param("feature_groups", "multi", ["ids"], "用哪些特征组", affects="model",
          choices=("ids", "user_profile", "item_attr", "user_stats", "item_stats"),
          choice_notes={"ids": "用户 ID、电影 ID", "user_profile": "性别、年龄段、职业、邮编首位",
                        "item_attr": "题材（多值）、上映年代", "user_stats": "用户此前评分次数（分桶）和平滑喜欢率",
                        "item_stats": "电影此前被评分次数（分桶）和平滑喜欢率"}),
    Param("arch", "choice", "mlp", "模型结构", affects="model", choices=("mlp", "deepfm", "dcn"),
          choice_notes={"mlp": "Embedding 拼接 + MLP", "deepfm": "MLP + FM 二阶交叉", "dcn": "MLP 并联 3 层 DCN-v2 cross"}),
    Param("embedding_dim", "int", 32, "每个特征的 embedding 维度", affects="model", min=2, max=1024,
          cost_note="ID 参数 ≈ (6041 + 3707) × d"),
    Param("hidden", "str", "256,128", "MLP 隐层宽度（逗号分隔）", affects="model"),
    Param("dropout", "float", 0.0, "MLP 的 dropout", affects="model", min=0.0, max=0.9),
    # ---------- 训练 ----------
    Param("lr", "float", 1e-3, "AdamW 学习率（常数）", affects="train", min=1e-6, max=1.0),
    Param("weight_decay", "float", 0.0, "权重衰减（只给 MLP，embedding 不衰减）", affects="train", min=0.0, max=1.0),
    Param("batch_size", "int", 2048, "batch 大小", affects="train", min=8, max=1_000_000),
    Param("epochs", "int", 1, "训练几遍（工业 CTR 模型通常只训 1 遍）", affects="train", min=1, max=100),
    Param("seed", "int", 0, "训练种子（初始化和样本顺序）", affects="train", min=0, max=1_000_000),
    # ---------- 评估与运行 ----------
    Param("bootstrap", "int", 100, "按用户 bootstrap 估计 AUC 抽样误差的次数", affects="eval", min=0, max=10_000),
    Param("device", "choice", "auto", "设备", affects="runtime", choices=("auto", "cuda", "mps", "cpu")),
    Param("max_minutes", "int", 20, "超时（分钟）", affects="runtime", min=1, max=1440),
])
