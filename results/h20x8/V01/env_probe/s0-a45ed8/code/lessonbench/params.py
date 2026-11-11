"""讲义验证（V01）的参数：跑哪一项（item），按哪张卡的规格对照（hw）。"""

from labkit.params import Param, ParamSpace

from .items import ITEMS

PARAMS = ParamSpace([
    Param("item", "choice", "env_probe", "验证项（每项对应讲义 GPU 验证清单的一条）", affects="runtime", choices=tuple(ITEMS),
          choice_notes={k: v["title"] for k, v in ITEMS.items()}),
    Param("hw", "choice", "H20", "讲义脚本的 --hw（屋顶线和报告文件名按这张卡）", affects="runtime", choices=("H20", "H100")),
    Param("seed", "int", 0, "种子（讲义脚本各自固定种子，这里只用于运行编号）", affects="runtime", min=0, max=1_000_000),
    Param("max_minutes", "int", 60, "超时（分钟）", affects="runtime", min=1, max=1440),
])
