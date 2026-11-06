from labkit.params import Param, ParamSpace

PARAMS = ParamSpace([
    Param("rows", "int", 2000, "样本数", affects="data", min=100),
    Param("noise", "float", 0.1, "标签翻转比例", affects="data", min=0.0, max=0.5),
    Param("leak", "bool", False, "教学故障：把标签当特征", affects="data"),
    Param("lr", "float", 0.5, "学习率", min=0.0),
    Param("epochs", "int", 3, "训练轮数", min=1),
    Param("seed", "int", 0, "随机种子"),
    Param("device", "str", "cpu", "设备", affects="runtime"),
    Param("max_minutes", "float", 5.0, "时间预算（分钟）", affects="runtime"),
])
