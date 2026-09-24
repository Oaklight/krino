> For English version, see [README_en.md](README_en.md)

# 决策模型

开源类型化决策模型复现。实现了通过 TypeSafe Jev 黑箱探测识别出的操作管道：

```
State → 共享编码 → 逐选项评分 → softmax → 校准概率
```

## 结果

### 最佳训练头准确率（Banking77，77 类意图分类）

| 骨干网络 | 类型 | 参数量 | 准确率 |
|----------|------|--------|--------|
| **Ettin-150m r128** | 重排序器预训练编码器 | 150M + ~800K | **95.2%** |
| Ettin-150m r64 | 重排序器预训练编码器 | 150M + ~400K | 93.8% |
| Qwen3-0.6B r64 | 因果语言模型 | 596M + 402K | 93.2% |
| Ettin-400m r64 | 重排序器预训练编码器 | 400M + ~400K | 91.6% |
| ModernBERT-base r64 | 普通编码器 | 149M + 301K | 89.0% |
| Jev（参考） | — | — | 77.8% |

### 关键发现

- **重排序器预训练可迁移：** Ettin-150m 在相同架构和参数量下比普通 ModernBERT-base 高 +4.8pp
- **训练优于规模：** 0.6B + 训练头 超过 7B 零训练 logit 读取
- **头部不泛化：** Banking77 训练的头在其他基准上仅约 35%——需要多任务训练
- **表面形式偏差持续存在：** 架构减少但未消除——需要带不变性损失的校准

详细分析见 [research/04-model-replication.md](../research/04-model-replication.md)。

## 状态

第 2 步（训练头）已完成。第 3 步（校准训练）基础设施已就绪，扫描待执行。
详见 epic [#19](https://github.com/Oaklight/krino/issues/19)。

## 结构

```
model/
├── src/            # 模型代码（骨干网络、头部、推理）
├── data/           # 数据管道和基准转换
├── training/       # 训练循环和损失函数
├── evaluation/     # 指标和比较工具
├── configs/        # 实验配置
├── scripts/        # 入口脚本（训练、评估、服务）
└── experiments/    # 日志、检查点、结果（已 gitignore）
```

## 研究路线

1. **Logit 读取**（#21）— 冻结 Qwen3-0.6B，直接 log-prob 评分
2. **训练头**（#22）— 冻结骨干 + 轻量决策头
3. **校准**（#23）— Brier/MMCE/focal + RLCD 风格 RL
4. **新架构**（#24）— 双向编码器对比
