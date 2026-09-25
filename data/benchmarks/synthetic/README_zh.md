# Krino 合成数据集

用于类型化决策模型的合成训练数据——模型输出结构化概率答案（是/否概率、分类分布、有序评分），而非自由文本。

## 快速开始

```python
# 安装
pip install krino

# 加载全部 154K 合成数据（含教师标签）
from data.pipeline import load_all
items = load_all(["synthetic"])
print(f"{len(items)} 条数据已加载")

# 每条数据都有多教师软标签
item = items[0]
print(item.question)        # {"type": "noul", "instructions": "..."}
print(item.label)           # True/False（金标签）
print(item.teacher_probs)   # {"jev": {"true": 0.92, ...}, "gpt_5_6_luna": {"true": 0.98, ...}}
```

无需中间文件——数据从各领域源文件实时组装。

## 数据集概要

- **154K 条数据**，覆盖 23 个领域、12 种认知类型
- **多教师标签**：Jev (System One) + GPT-5.6 Luna (reasoning_effort=high)
- **86% 教师一致率**——14% 分歧数据是校准训练的核心
- **混合生成**：14 个种子驱动领域（真实基准数据）+ 9 个纯生成领域
- **家族结构**：每个 state 产生约 35 条相关数据（基础 + 反事实 + 释义 + 否定 + 选项打乱变体）

## 领域一览

### 原始领域（10 个）

| 领域 | 类型 | 种子来源 | 数据量 |
|---|---|---|---|
| 医疗分诊 | 种子 | MedNLI | 6,849 |
| 法律判断 | 种子 | ContractNLI | 6,382 |
| 代码审查 | 种子 | CodeSearchNet | 6,980 |
| 金融分析 | 种子 | TabFact | 6,860 |
| 科学推理 | 种子 | ARC | 6,983 |
| 内容分析 | 种子 | FEVER | 6,969 |
| 空间推理 | 纯生成 | — | 6,968 |
| 产品分类 | 纯生成 | — | 6,985 |
| 教育评估 | 纯生成 | — | 6,899 |
| 安全审核 | 纯生成 | — | 6,980 |

### 推理领域（5 个）

| 领域 | 类型 | 种子来源 | 数据量 |
|---|---|---|---|
| 学术推理 | 种子 | MMLU | 6,917 |
| 常识决策 | 种子 | CommonsenseQA | 6,984 |
| 逻辑推断 | 种子 | LogiQA | 6,916 |
| 对抗推理 | 种子 | ANLI | 6,941 |
| 段落决策 | 种子 | BoolQ | 6,978 |

### 序列决策领域（5 个）

| 领域 | 类型 | 数据量 |
|---|---|---|
| 游戏策略 | 纯生成 | 6,722 |
| 导航规划 | 纯生成 | 6,972 |
| 资源管理 | 纯生成 | 6,988 |
| 序列动作 | 纯生成 | 6,978 |
| 多智能体协调 | 纯生成 | 6,957 |

### 长上下文领域（3 个）

| 领域 | 类型 | 种子来源 | 数据量 |
|---|---|---|---|
| 长文档 | 种子 | QuALITY | 2,374 |
| 多跳推理 | 种子 | HotpotQA | 6,704 |
| 数值推理 | 种子 | DROP | 6,806 |

## 多教师标签

每条数据都有两个教师的概率分布：

| 教师 | 模型 | 智能度 | 校准度 | 覆盖率 |
|---|---|---|---|---|
| Jev | System One (jev-1.13.0) | 基准 | 基准 | 99.3% |
| Luna | GPT-5.6 Luna (reasoning_effort=high) | 96.8 | 89.8 | 91.7% |

**教师一致率：86.1%**。分歧集中在模糊领域（导航 79%、空间 81%、安全 82%）——正是校准训练最有价值的地方。

## 问题类型

遵循 [TypeSafe System One](https://docs.typesafe.ai) API 的三种类型化问题：

- **Noul**（伯努利）：是/否命题 → P(true) ∈ [0, 1]
- **Choice**：从标记选项中选一个 → 分类概率分布
- **Score**：在有序等级上评分 → 有序概率分布

类型比例：noul 57% / choice 28% / score 14%

## 上下文长度分布

```
     0-100 字符     354 条 (  8%)  短句状态
   100-500          1354     ( 30%)  段落级状态
   500-1K           1020     ( 23%)  多段落状态
  1K-2K             1021     ( 23%)  序列决策状态
  2K-5K              398     (  9%)  长文本状态
  5K-10K             282     (  6%)  长上下文（文档、多跳）
  10K+                47     (  1%)  超长（法律合同）
```

中位数：695 字符。平均：1,441 字符。

## 文件结构

```
{domain}_families.jsonl              — 原始家族数据（state + 问题 + 金标签）
{domain}_variants.jsonl              — 变体数据（反事实、释义、否定）
{domain}_jev_labels.jsonl            — Jev 软标签标注
{domain}_gpt_5_6_luna_labels.jsonl   — Luna 软标签标注
```

数据由 `load_synthetic()` 实时组装——无需单独的输出文件。避免了重复（每个家族约 35 条变体数据共享相同的 state）。

## CLI 使用

```bash
# 查看生成状态
python -m data.synthetic --stages report

# 生成新领域数据
python -m data.synthetic --domains game_strategy --families-per-domain 200 \
    --stages base,counterfactual,paraphrase,negation,shuffle

# 添加 Jev 标签
python -m data.synthetic --domains game_strategy --stages jev-label

# 添加 Luna 标签
python -m data.synthetic --domains game_strategy --stages llm-label \
    --llm-teacher argo:gpt-5.6-luna

# 推送到 HuggingFace
python -m data.synthetic --push-to-hf oaklight/open-decisions-synthetic
```

## 数据格式

```json
{
    "id": "synthetic-medical_triage-0042-noul-causal",
    "state": "前提：78 岁女性高血压患者左下腹疼痛就诊...",
    "question": {
        "type": "noul",
        "instructions": "高血压是否可能加重腹部症状？"
    },
    "label": true,
    "source": "synthetic",
    "split": "train",
    "group": "synthetic-medical_triage-0042",
    "teacher_probs": {
        "jev": {"true": 0.82, "false": 0.18},
        "gpt_5_6_luna": {"true": 0.95, "false": 0.05}
    }
}
```

## 训练集成

```python
from data.pipeline import load_all

# 加载合成 + 基准数据
items = load_all(["synthetic", "banking77", "mnli", "arc"])

# 使用 teacher_probs 进行蒸馏
for item in items:
    if item.teacher_probs:
        jev_probs = item.teacher_probs.get("jev")
        luna_probs = item.teacher_probs.get("gpt_5_6_luna")
        # 选择教师或集成
```

## 引用

```bibtex
@misc{krino-synthetic-2026,
    title={Krino Synthetic Dataset},
    author={Peng Ding},
    year={2026},
    url={https://huggingface.co/datasets/oaklight/open-decisions-synthetic}
}
```

源代码：[Oaklight/krino](https://github.com/Oaklight/krino)

---

[English version](README.md)
