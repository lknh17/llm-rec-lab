# L01 数据与分词：去重、字节级 BPE 的词表大小、bits-per-byte、验证集泄漏

从中文维基的一个分片出发，走一遍"清洗 → 去重 → 训练字节级 BPE → token 化 → 从零训练一个小 GPT → 用 bits-per-byte 评估"。看词表大小怎样同时改变序列长度、embedding 参数和输出层算力；为什么跨分词器只能比 bpb；Qwen2.5 的大词表在维基和广告上各压缩得怎样；最后诊断一次"关掉去重，验证集变得过于乐观"，并量化重复几次的文章会被模型记住。

## 怎么学（只看 GitHub 就够）

按下面“步骤”表从第 1 步往下点。每一页先给问题和选项，**先选定一个再展开“实测”**；实测里有结果表、证据图和运行目录链接。作业进度见 [作业看板](../../jobs/README.md)。参考答案和解析只在私有工作台里。

## 任务

**出发点**：内容理解模型的训练和推理成本，最终都按 **token** 计。同一段文本切成多少 token，由分词器决定；模型在验证集上的分数可不可信，由语料去重决定。这两件事都发生在"训练开始之前"，事后很难发现。

这一课用一个可以在几分钟内从零训完的小 GPT（主体约 1060 万参数）做实验：

- **语料**：中文维基一个分片（约 23 万篇）。清洗后先做一次"底稿去重"，留出 10 MB 干净测试集；再**模拟网页转载**：20% 的文章被复制 1-40 份（长尾分布），一半逐字相同，一半删句、加转载声明；然后按场景做去重，按"URL 哈希"切出验证集，训练集固定取 100 MB 文本；
- **分词器**：在训练集上训练字节级 BPE（预切分规则与 Qwen2 相同），对照 UTF-8 字节直接当 token、Qwen2.5 现成的分词器；
- **评估**：bits-per-byte（bpb）= 总交叉熵 / ln 2 / 原文字节数。和每 token loss 不同，bpb 和分词器无关，可以跨分词器比较。另外统计压缩率（每 token 多少字节）和 60 条人工广告文案每条多少 token。

| 集合 | 大小 | 用途 |
| - | - | - |
| 训练集 | 100 MB 文本 | 训练 BPE（前 20 MB）和小 GPT |
| 验证集 | 约 10 MB 原文（去重不干净时还会多出副本） | 训练中看曲线；按"同源文档是否进了训练集"拆组 |
| 干净留出集 test | 10 MB | 注入副本之前就拿走，各场景完全相同：真实泛化的标尺 |
| 人工广告 | 60 条 | 只评估：业务文本的压缩率和 bpb |

## 规模与边界

- **H20 档**：小 GPT（d=384、6 层、6 头、SwiGLU 1024、RoPE、tied embedding，序列 512 token），batch 64 × 512，AdamW lr 2e-3，100 MB 文本过一遍；bf16 autocast；每个场景 3 个种子（种子只改初始化和数据顺序，语料和分词器固定）。
- **保留了**：清洗、MinHash + LSH 近似去重、字节级 BPE 训练与词表大小、bpb 口径、验证集泄漏的完整机制。
- **缩小了**：模型和数据都很小（Chinchilla 意义上严重欠训练），所以 bpb 的绝对值、"哪个词表最好"的结论不能外推；转载副本是人为注入的。
- **未验证**：大模型上词表大小的最优点（文献里随模型增大而增大）；真实网页语料的重复结构（模板、镜像站、机器改写）比这里复杂。

## 步骤

<!-- labkit:path -->
| # | 阶段 | 步骤 | 这一步回答的问题 | 场景 | H20 结果 |
| - | - | - | - | - | - |
| 1 | 理解任务 | [场景与痛点](steps/01-pain.md) | 训练开始之前的两个决定——怎么切 token、怎么去重——会让成本差几倍、让验证集失真多少？ | — | — |
| 2 | 理解任务 | [流程结构](steps/02-flow.md) |  | — | — |
| 3 | 理解任务 | [追踪一段文本](steps/03-trace.md) |  | — | — |
| 4 | 理解任务 | [代码导读](steps/04-code.md) |  | — | — |
| 5 | 先估算 | [先估算](steps/05-estimate.md) | 词表换成 64K（65536）。embedding 参数占模型总参数的比例大约是？ | — | — |
| 6 | 词表大小与分词方式 | [词表 16K → 4K](steps/06-vocab-4k.md) | 和 16K 比，4K 词表的干净留出集 bpb（3 种子均值）会怎样？ | `bpe_4k` | 0 / 3 |
| 7 | 词表大小与分词方式 | [词表 16K → 64K](steps/07-vocab-64k.md) | 和 16K 比，64K 词表的干净留出集 bpb（3 种子均值）会怎样？ | `bpe_64k` | 0 / 3 |
| 8 | 词表大小与分词方式 | [不分词：UTF-8 字节当 token](steps/08-bytes.md) | 和 16K BPE 比，字节级的干净留出集 bpb（3 种子均值）会怎样？ | `bytes` | 0 / 3 |
| 9 | 词表大小与分词方式 | [直接用 Qwen2.5 的分词器](steps/09-qwen.md) | 在这份维基验证集上，Qwen2.5 分词器的每 token 字节数，和本语料训练的 16K BPE 比？ | `qwen` | 0 / 3 |
| 10 | 词表的领域 | [BPE 训练文本里混入广告](steps/10-tok-ads.md) | 60 条人工广告，平均每条 token 数和基线比会怎样？ | `tok_ads` | 0 / 3 |
| 11 | 去重与验证集 | [只做精确去重](steps/11-dedup-exact.md) | 只做精确去重时，验证集里有多少比例的文档，它的同源文档（原文或副本）进了训练集？ | `dedup_exact` | 0 / 3 |
| 12 | 去重与验证集 | [关掉去重，验证集变乐观了](steps/12-fault.md) | 不去重的模型上，"留出集 − 验证集"的 bpb 差，比基线大多少？ | `fault_nodedup` | 0 / 3 |
| 13 | 去重与验证集 | [诊断：验证集为什么更乐观](steps/13-diagnose.md) | 验证集变得更乐观的主要原因是什么？ | `baseline` | 0 / 3 |
| 14 | 去重与验证集 | [背过几遍才算背下来](steps/14-leak.md) | 不去重的模型上，"训练集里有 8 篇以上同源文档"的那部分验证文档，bpb 比从没被转载过的原文低多少？ | `fault_nodedup` | 0 / 3 |
| 15 | 带走 | [回顾](steps/15-keycard.md) |  | — | — |
<!-- /labkit:path -->

## 场景与结果

<!-- labkit:results -->
<details><summary>展开结果（先自己预测，再看）</summary>

**h20x8**：8 × NVIDIA H20，每张卡一个运行。中文维基一个分片（约 9300 万字）→ 训练集固定 100 MB 文本；小 GPT（d=384、6 层，主体约 1060 万参数）从零训练一遍（已有 0 / 24 个运行）

_还没有结果：等待批量运行。_

</details>
<!-- /labkit:results -->

## 证据图

<!-- labkit:figures -->
<details><summary>展开证据图（先自己预测，再看）</summary>

- **h20x8**：还没有结果，图在结果推回后自动生成。

</details>
<!-- /labkit:figures -->

## 代码

<!-- labkit:code -->
| 代码段 | 位置 |
| - | - |
| `clean` | [projects/tinylm/corpus.py#L44](../../projects/tinylm/corpus.py#L44) |
| `minhash` | [projects/tinylm/corpus.py#L111](../../projects/tinylm/corpus.py#L111) |
| `build` | [projects/tinylm/corpus.py#L219](../../projects/tinylm/corpus.py#L219) |
| `flops_budget` | [projects/tinylm/experiment.py#L42](../../projects/tinylm/experiment.py#L42) |
| `docmask` | [projects/tinylm/model.py#L61](../../projects/tinylm/model.py#L61) |
| `moe` | [projects/tinylm/model.py#L88](../../projects/tinylm/model.py#L88) |
| `attention` | [projects/tinylm/model.py#L142](../../projects/tinylm/model.py#L142) |
| `mla` | [projects/tinylm/model.py#L161](../../projects/tinylm/model.py#L161) |
| `norm` | [projects/tinylm/model.py#L190](../../projects/tinylm/model.py#L190) |
| `gpt` | [projects/tinylm/model.py#L200](../../projects/tinylm/model.py#L200) |
| `kvcache` | [projects/tinylm/model.py#L296](../../projects/tinylm/model.py#L296) |
| `train_bpe` | [projects/tinylm/tokenizer.py#L85](../../projects/tinylm/tokenizer.py#L85) |
| `compression` | [projects/tinylm/tokenizer.py#L137](../../projects/tinylm/tokenizer.py#L137) |
| `schedule` | [projects/tinylm/train.py#L31](../../projects/tinylm/train.py#L31) |
| `batch` | [projects/tinylm/train.py#L71](../../projects/tinylm/train.py#L71) |
| `evaluate` | [projects/tinylm/train.py#L80](../../projects/tinylm/train.py#L80) |
| `budget` | [projects/tinylm/train.py#L141](../../projects/tinylm/train.py#L141) |
| `train` | [projects/tinylm/train.py#L162](../../projects/tinylm/train.py#L162) |
<!-- /labkit:code -->
