<h1 align="center">Krino</h1>

<p align="center">
  <a href="README_en.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<p align="center"><em>κρίνω — 判断、分离、决定</em></p>

Krino 是一个研究**决策模型（decision model）**的项目——这是一类新型 AI 系统，输出结构化选项上的校准概率，而非生成文本。本项目对 [TypeSafe AI](https://typesafe.ai) 的 Jev（首个商用决策模型）进行逆向工程、复现和扩展研究。

## 什么是决策模型？

决策模型接收非结构化输入（工单、合同条款、代码片段）和一组类型化问题，返回结构化的概率答案——不生成文本，不需要解析，不需要重试。

| 问题类型 | 回答什么 | 输出 |
|---|---|---|
| **Noul** | 这个说法是否成立？ | 概率 0–1 |
| **Choice** | 哪个选项？ | 每个选项的概率分布 |
| **Score** | 在这个尺度上的位置？ | 概率加权位置 |

关键区别：这些概率是**校准的（calibrated）**——当模型输出 0.8 时，答案应该在约 80% 的情况下是正确的。这使得基于阈值的自动化（路由、拦截、升级）无需为每个任务单独训练分类器。

## 项目组成

| 目录 | 功能 | 亮点 |
|---|---|---|
| [`probing/`](probing/) | Jev API 黑箱探测 | 21 个探针、5,620 次 API 调用；5 个预注册的对照实验 |
| [`model/`](model/) | 开源决策模型复现 | Banking77 准确率 95.2%（Jev 为 75%）；10 个骨干网络 × 19 个基准 |
| [`data/`](data/) | 数据管道 + 合成数据生成 | 26 个基准加载器；含反事实和释义的合成数据 |
| [`research/`](research/) | 分析文档 + 生态调研 | 架构指纹分析；复现项目全景 |

## 主要发现

1. **重排序器预训练可迁移到决策评分。** 一个 150M 的交叉编码器重排序器（Ettin-150m）配合训练的决策头，在 Banking77 上达到 95.2%——在相同参数量下比普通编码器高 +4.8pp，超过 Jev 本身（75.0%）。

2. **零训练 logit 读取效果出奇的好。** 直接从冻结 LLM 的 logit 中读取选项概率（无训练、无文本生成），使用 Qwen2.5-7B 在 19 个基准上平均准确率达 68.5%——与 Jev 的 87.2% 仅差 19pp。

3. **决策模型生态正在快速形成。** JevBench 追踪了 52 个系统，涵盖专有、开源训练和零训练方案。多个独立项目（SemIf、djev、Decider）证明决策能力是预训练 LLM 的潜在能力，可以通过结构化 logit 读取来释放。

## 快速开始

```bash
# 克隆并安装
git clone https://github.com/Oaklight/krino.git
cd krino
pip install -e '.[data]'

# 下载并转换基准数据
python -m data.pipeline banking77 sst2

# 评估骨干网络（零训练 logit 读取）
python model/scripts/eval_logit.py --model Qwen/Qwen3-0.6B

# 在冻结骨干网络上训练决策头
python model/scripts/train.py --encoder --model answerdotai/ModernBERT-base \
    --train-data data/benchmarks/banking77.jsonl \
    --eval-data data/benchmarks/banking77.jsonl \
    --epochs 5
```

## 链接

| | |
|---|---|
| 论文 | [Oaklight/krino-paper](https://github.com/Oaklight/krino-paper) |
| PyPI | [krino](https://pypi.org/project/krino/) |
| 模型 | [oaklight/krino-*](https://huggingface.co/models?search=oaklight/krino) |
| 数据集 | [oaklight/krino-synthetic](https://huggingface.co/datasets/oaklight/krino-synthetic) |
| 文档 | [oaklight.github.io/krino](https://oaklight.github.io/krino) |
| TypeSafe 文档 | [docs.typesafe.ai](https://docs.typesafe.ai) |

## 引用

```bibtex
@misc{krino-2026,
    title={Krino: Probing, Replication, and Analysis of JEV-Class Decision Models},
    author={Peng Ding},
    year={2026},
    url={https://github.com/Oaklight/krino}
}
```

## 参与贡献

这是我个人持续更新的研究探索项目。欢迎提 Issue、参与讨论或提交 Pull Request——如果你对决策模型感兴趣，随时加入。

## 许可证

MIT
