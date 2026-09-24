> For English version, see [README_en.md](README_en.md)

# 数据管道

下载、转换和管理决策模型评估的基准数据集。所有基准数据统一转换为 `TypedQuestion` 格式。

## 快速开始

```bash
pip install -e '.[data]'

# 下载并转换指定基准
python -m data.pipeline banking77 sst2 arc

# 下载所有基准
python -m data.pipeline
```

转换后的数据以 JSONL 文件保存在 `data/benchmarks/` 目录下。

## 基准数据

26 个加载器，涵盖 3 种问题类型和多个领域：

**Choice**（选择题）：
banking77、agnews、arc、race、hellaswag、fever、swag、codesearchnet、mmlu、winogrande、piqa、commonsenseqa、logiqa、typed_decisions

**Noul**（是/否概率）：
sst2、mnli、tabfact、multirc、mednli、contractnli、anli、boolq

**Score**（有序等级）：
stsb、sst5、yelp

**合成数据**：
synthetic（通过合成管道生成）

## TypedQuestion 格式

所有基准转换为 `TypedQuestion`（定义于 `format.py`）：

```python
TypedQuestion(
    id="banking77-test-00042",
    state="I was charged twice for the same transaction",
    question={"type": "choice", "instructions": "...", "criteria": {...}},
    label="transaction_charged_twice",
    source="banking77",
    split="test",
    group="banking77-42",
)
```

三种问题类型：`noul`（布尔标签）、`choice`（字符串标签）、`score`（浮点标签）。

## 合成数据管道

```bash
python -m data.synthetic --stages base,counterfactual,paraphrase,negation,shuffle,dedup,validate
python -m data.synthetic --stages report    # 检查状态，不执行
```

阶段：`base` → `counterfactual,paraphrase,negation` → `shuffle` → `fill-variants` → `repair` → `dedup` → `validate` → `llm-label` → `jev-label` → `report`

数据存储在 HuggingFace：[`oaklight/krino-synthetic`](https://huggingface.co/datasets/oaklight/krino-synthetic)（私有）。

## 结构

```
data/
├── format.py           # TypedQuestion 数据类
├── pipeline.py         # 26 个基准加载器
├── sampler.py          # 分层采样工具
├── synthetic.py        # 合成数据生成编排器
├── synthetic_*.py      # 各阶段实现
├── balance.py          # 类别平衡
├── lsh.py              # MinHash/LSH 去重
└── benchmarks/         # 下载的数据（已 gitignore）
    ├── synthetic/      #   合成数据 + HF 数据集卡片
    └── openjev/        #   OpenJev 基准数据
```

## 关键文件

| 文件 | 用途 |
|---|---|
| `format.py` | `TypedQuestion` 数据类——统一数据格式 |
| `pipeline.py` | 全部 26 个基准加载器 + CLI 入口 |
| `synthetic.py` | 合成数据生成管道编排器 |
| `lsh.py` | 独立的 MinHash/LSH 去重模块 |
| `sampler.py` | 分层采样，用于平衡的评估子集 |
