> For English version, see [README_en.md](README_en.md)

# 探测（Probing）

TypeSafe AI Jev 决策模型的黑箱探测套件。所有实验仅观察外部 HTTP 行为——不需要模型权重或内部访问。

## 功能

向 Jev API 发送结构化请求并分析响应，推断运行特性：输出量化方式、选项交互效应、表面形式敏感性、执行拓扑和生成签名。

## 结构

```
probing/
├── scripts/              # 探针实现
│   ├── probe_01_*.py     #   标准探针（13 个实验）
│   ├── probe_02_*.py
│   ├── ...
│   ├── controlled/       #   预注册对照探针（5 个实验）
│   │   ├── probe_1_output_serialization.py
│   │   ├── probe_2_option_interaction.py
│   │   ├── probe_3_likelihood_sensitivity.py
│   │   ├── probe_4_execution_topology.py
│   │   └── probe_5_generation_signatures.py
│   ├── jev_client.py     #   Jev API 客户端（仅标准库）
│   └── httpclient.py     #   HTTP 客户端
├── results/              # 原始 JSONL 输出
│   └── controlled/       #   对照探针输出
└── controlled/
    └── README.md         # 预注册：假设、证伪矩阵
```

## 运行探针

所有探针脚本仅使用 Python 标准库。需要在环境中设置 `TYPESAFE_API_KEY`。

```bash
# 标准探针
python probing/scripts/probe_01_token_accounting.py
python probing/scripts/probe_02_noul_precision.py

# 对照探针（预注册，带安全标志）
python probing/scripts/controlled/probe_1_output_serialization.py \
    --mode smoke --seed 20260918 --dry-run

# 实际运行（需要明确确认）
python probing/scripts/controlled/probe_2_option_interaction.py \
    --mode full --seed 20260918 \
    --output probing/results/controlled/probe_2.jsonl \
    --confirm-live
```

参数：`--mode smoke|full`、`--seed`、`--output`、`--dry-run`、`--confirm-live`。

## 主要发现

- **21 个探针，5,620 次 API 调用**，经过 3 轮严谨性自审
- 架构指纹识别出 Qwen 系列骨干网络、类型化决策头和 RLCD 校准
- 排序偏差有所减少但未消除
- 概率量化为 0.01 精度
- 预注册的证伪矩阵详见 [controlled/README.md](controlled/README.md)
