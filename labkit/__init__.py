"""labkit：推荐 + LLM 实践课的执行内核。

- params / store / runtime / snapshot / cache / worker：一次运行怎么定义、怎么执行、证据怎么落盘；
- scenarios / batch：课程场景清单展开成一批运行，按卡并行执行并检查期望断言；
- redact / jobs / publish：脱敏导出、作业单协议、推回 GitHub；
- assets / probe：公开数据集和模型的下载、环境预检。
"""

LABKIT_VERSION = "0.2.0"
