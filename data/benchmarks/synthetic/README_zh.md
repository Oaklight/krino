<p align="center">
  <a href="README_en.md">English</a> | <a href="README_zh.md">中文</a>
</p>

# 合成数据生成流水线

在 10 个领域、12 种认知类型下生成约 50K 条类型化决策训练数据，包含对比对、表面形式变体和 Jev API 软标签标注。

## 流水线设计

### 混合策略

| 方法 | 领域 | 数据来源 |
|---|---|---|
| **种子驱动** | 医疗 (MedNLI)、法律 (ContractNLI)、金融 (TabFact)、科学 (ARC)、代码 (CodeSearchNet)、内容 (FEVER) | 真实基准测试的 state + LLM 生成的问题 |
| **纯生成** | 空间推理、产品分类、教育评估、安全审核 | LLM 同时生成 state 和问题 |
| **增强** | 所有领域 | 反事实、释义、否定、选项顺序打乱变体 |

### 跨模型生成

每个阶段使用不同的 LLM，避免单模型偏差：

- **基础家族**: Claude Sonnet（通过 `LLM_GEN_MODEL`）
- **变体**: GPT-4.1（通过 `LLM_VARIANT_MODEL`）
- **验证**: Gemini Flash（通过 `LLM_JUDGE_MODEL`）

### 家族结构

每个家族从一个基础 state 生成约 20 条训练数据：

```
Family {
    base:           4 noul + 3 choice + 2 score 问题
    counterfactual: 修改 state → 翻转标签
    paraphrase:     改写 state/问题 → 标签不变
    negation:       翻转问题极性 → 翻转 noul 标签
    shuffle:        打乱 choice 选项顺序 → 标签不变
}
```

### 可组合的阶段

每个阶段可独立运行，自动从缓存的中间文件恢复：

| 阶段 | 描述 | 中间文件 |
|---|---|---|
| `base` | 生成家族（state + 问题 + 金标签） | `{domain}_families.jsonl` |
| `counterfactual` | 修改 state 并翻转标签 | `{domain}_variants.jsonl` |
| `paraphrase` | 改写 state/问题，保持标签不变 | `{domain}_variants.jsonl` |
| `negation` | 翻转 noul 问题的极性 | `{domain}_variants.jsonl` |
| `shuffle` | 打乱 choice 选项顺序 | （转换时计算） |
| `dedup` | 基于 LSH 的近重复检测 | 覆盖 `_families.jsonl` |
| `validate` | 使用裁判模型进行跨模型验证 | 内存中过滤 |
| `jev-label` | Jev API 软标签标注 | 添加 `teacher_probs` 字段 |

### 质量控制

- **标签验证**：choice 标签对照选项键检查，score 标签对照范围检查
- **防泄露提示**：state 只包含原始场景，不含解释或判断
- **置信度分桶**：Jev 软标签分为高（>0.9）、中（0.6–0.9）、不确定（<0.6）
- **LSH 去重**：基于字符 n-gram 的 MinHash 检测近重复 state

## CLI 使用

```bash
# 完整运行：全部 10 个领域，每个 200 个家族
python -m data.synthetic

# 试运行：单领域 10 个家族
python -m data.synthetic --domains spatial_reasoning --pilot

# 仅生成基础家族（不生成变体）
python -m data.synthetic --stages base --domains education_assessment

# 为已有家族添加变体
python -m data.synthetic --stages counterfactual,paraphrase,negation

# 仅去重（无 LLM 调用）
python -m data.synthetic --stages dedup

# 仅 Jev 软标签
python -m data.synthetic --stages jev-label

# 推送到 HuggingFace
python -m data.synthetic --push-to-hf Oaklight/jev-synthetic

# 自定义并发和模型
LLM_GEN_MODEL="argo:claude-sonnet-4.6" \
LLM_VARIANT_MODEL="argo:gpt-4.1" \
python -m data.synthetic --max-concurrent 10
```

## 配置

在仓库根目录的 `.env` 中设置：

```
LLM_BASE_URL=http://your-llm-endpoint:port
LLM_API_KEY=your-key-if-needed
LLM_GEN_MODEL=claude-sonnet-4-20250514
LLM_VARIANT_MODEL=GPT-4.1-mini
LLM_JUDGE_MODEL=gemini-2.0-flash
```

## 文件结构

```
data/benchmarks/synthetic/
├── README_en.md                         # 英文版
├── README_zh.md                         # 本文件
├── README.md -> README_en.md            # 符号链接
├── {domain}_families.jsonl              # 原始家族数据（可恢复）
├── {domain}_variants.jsonl              # 原始变体数据（可恢复）
└── synthetic.jsonl                      # 最终 TypedQuestion 数据
```

## 输出格式

每条数据是一个 `TypedQuestion` 字典：

```json
{
    "id": "synthetic-spatial_reasoning-0042-noul-causal",
    "state": "一个长方形博物馆展厅，东西方向 20 米...",
    "question": {"type": "noul", "instructions": "重新排列展品是否会导致瓶颈？"},
    "label": true,
    "source": "synthetic",
    "split": "train",
    "group": "synthetic-spatial_reasoning-0042",
    "teacher_probs": {"true": 0.82, "false": 0.18}
}
```

`teacher_probs` 字段（由 `jev-label` 阶段添加）包含 Jev 的完整概率分布，用于蒸馏训练。
