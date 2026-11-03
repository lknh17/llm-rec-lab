"""推荐项目的全部可调参数。网页表单、运行配置、讲解文字共用这一份定义。"""

from labkit.params import Param, ParamSpace

FEATURE_GROUPS = ("ids", "user_profile", "item_attr", "user_stats", "item_stats")

PARAMS = ParamSpace([
    # ---------- 数据（改了会让数据缓存失效） ----------
    Param("dataset", "choice", "ml-1m", "数据集", affects="data", choices=("ml-1m", "ml-20m", "kuairand-pure", "synth"),
          choice_notes={"ml-1m": "MovieLens-1M：100 万条评分、6040 用户、3706 电影，有用户画像。课程的可比基线",
                        "ml-20m": "MovieLens-20M：2000 万条评分、13.8 万用户、2.7 万电影，没有用户画像（不能开 user_profile）",
                        "kuairand-pure": "KuaiRand-Pure：快手真实曝光日志约 144 万条，标签 = is_click（曝光未点击才是负样本）",
                        "synth": "合成数据：规模由 synth_* 参数决定，已知真实点击概率，可算理论最优 AUC"},
          cost_note="换数据集后指标和其他数据集不可直接比较；ML-20M 的预处理约需数分钟和 10 GB 内存"),
    Param("synth_users", "int", 6000, "合成数据：用户数（dataset=synth 时生效）", affects="data", min=100, advanced=True),
    Param("synth_items", "int", 4000, "合成数据：物品数", affects="data", min=100, advanced=True),
    Param("synth_rows", "int", 1_000_000, "合成数据：交互条数（去重前）", affects="data", min=1000, advanced=True,
          cost_note="行数决定预处理和每轮训练的时间：1 亿行约需 30 GB 内存"),
    Param("synth_alpha", "float", 2.0, "合成数据：偏好信号强度。越大，理论最优 AUC 越高", affects="data", min=0.0,
          max=10.0, advanced=True),
    Param("synth_seed", "int", 0, "合成数据：生成世界的随机种子（和训练种子 seed 无关）", affects="data", advanced=True),
    Param("label_threshold", "int", 4, "评分 >= 该值记为正样本（喜欢）。只对 MovieLens 生效", affects="data", min=2, max=5,
          cost_note="改变任务定义：正样本比例随之变化，和旧运行的 LogLoss 不可直接比较"),
    Param("split", "choice", "time", "训练/验证/测试的切分方式", affects="data", choices=("time", "random"),
          choice_notes={"time": "按全局时间 80/10/10 切分，模拟用过去预测未来",
                        "random": "随机切分。未来的行为会进入训练集，离线指标偏乐观，用于诊断练习"}),
    Param("stats_mode", "choice", "point_in_time", "统计特征（用户/电影历史喜欢率）的计算口径", affects="data",
          choices=("point_in_time", "full_data"),
          choice_notes={"point_in_time": "只用样本时间点之前的行为计算，线上可得",
                        "full_data": "教学故障：用全量数据（含该样本自己的标签和未来行为）计算，属于标签泄漏"}),
    # ---------- 模型 ----------
    Param("feature_groups", "multi", ["ids", "user_profile", "item_attr"], "参与模型的特征组", affects="model",
          choices=FEATURE_GROUPS,
          choice_notes={"ids": "user_id、movie_id 两个 ID embedding",
                        "user_profile": "性别、年龄段、职业、邮编首位",
                        "item_attr": "上映年代、题材（多值，mean pooling）",
                        "user_stats": "用户历史评分数（分桶 embedding）+ 平滑喜欢率（数值）",
                        "item_stats": "电影历史评分数（分桶 embedding）+ 平滑喜欢率（数值）"},
          cost_note="每多一个离散字段，MLP 输入多 embedding_dim 维；统计特征需要额外的数据处理"),
    Param("embedding_dim", "int", 16, "每个离散字段的 embedding 维度", affects="model", min=2, max=256,
          cost_note="ID embedding 参数量 = (6040+3706) x dim；dim 翻倍参数和访存翻倍"),
    Param("hidden", "str", "128,64", "MLP 隐层宽度，逗号分隔", affects="model"),
    Param("dropout", "float", 0.0, "MLP dropout", affects="model", min=0.0, max=0.9),
    # ---------- 训练 ----------
    Param("lr", "float", 1e-3, "AdamW 学习率", affects="train", min=1e-6, max=1.0),
    Param("weight_decay", "float", 0.0, "AdamW 权重衰减", affects="train", min=0.0, max=1.0),
    Param("batch_size", "int", 1024, "每个 batch 的样本数", affects="train", min=16, max=65536,
          cost_note="batch 越大每 epoch 的更新次数越少：80 万训练样本 / 1024 ≈ 781 步"),
    Param("epochs", "int", 2, "训练轮数", affects="train", min=1, max=50),
    Param("max_steps", "int", 0, "最多训练步数，0 表示不限（预算控制）", affects="train", min=0),
    Param("seed", "int", 0, "随机种子：参数初始化、打乱顺序、随机切分", affects="train"),
    # ---------- 评估 ----------
    Param("final_test", "bool", False, "是否在保留测试集上评估。只在阶段性最终检验时打开", affects="eval",
          cost_note="测试集看多了就不再独立：迭代选择一律用验证集"),
    # ---------- 运行 ----------
    Param("device", "choice", "auto", "运行设备", affects="runtime", choices=("auto", "cpu", "mps", "cuda"),
          choice_notes={"auto": "cuda > cpu（小 MLP 在 Mac 上 CPU 往往比 MPS 快）"}),
    Param("max_minutes", "float", 20.0, "时间预算（分钟），超时自动停止并记录原因", affects="runtime", min=0.5),
    Param("resume_from", "str", "", "从某个运行的最新检查点续训（填运行编号）；留空从头训练", affects="runtime",
          advanced=True),
])
