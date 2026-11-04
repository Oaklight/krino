# 吞吐杠杆：Jev 类仓库与 Unsloth 的调研

日期：2026-11-03。范围：公开文档与仓库文件的只读调研，未运行他人代码。

## 我们的现状

- D20K：frozen Qwen3-Reranker-4B + 251,073 参数 marker 头，chunk=1，max_length 2048，batch=1，bf16。
- 实测稳态 6.6 items/s，`max_memory_allocated` **8.1 GiB**（H100 NVL 96 GiB）。
- 结论：瓶颈是每步延迟，不是显存。每步只推一个 1-token suffix，GPU 大部分时间空转。

## 两家调研的收敛点

**把 K 次 suffix 前向合成一次共享 prefix 前向。** 三个独立来源给出同一做法：

- Unsloth `v0.1.904-beta`（2026-10-08）的 Jev 风格 Decision Model：`unsloth/_vendor/clef/joint_schema_model.py` 的 `encode_record` 把 `prefix + state + schema(全部选项) + suffix` 拼成一条 `input_ids`，`JointSchemaHead` 在每个 `option_span` 上打分，`use_cache=False`。
- `jaredpalmer/kev` 的 `kev/shared_prefix.py`：训练期共享 prefix，成本从 `q×(state+branch)` 降到 `state+Σbranch`。
- `Mapika/decider` 的 `engine.py`：`score_shared()` 前缀算一次后 fork cache；`slot_logits()` 只对 slot 位置做线性层。

要点：若要保持「每个选项只看 prefix + 自己」，必须用 block-diagonal attention mask，并按选项重置 position ids（Unsloth packing 文档提到 reset position ids；具体实现在本项目为推断）。

## 其余可迁移手段

| 手段 | 来源 | 对本项目的适用性 |
| --- | --- | --- |
| 固定 (B,T) + CUDA graph + `torch.compile(dynamic=False)` | `Mapika/decider` `engine.py`；torchtitan PR #5569 | 高。我们的形状本来就固定 |
| 只对 marker 位置算 head 与 loss，用固定大小缓冲 | torchtitan PR #5569 `AnswerTokenLoss`；`decider.slot_logits()` | 高，已部分做到 |
| 按 padded token 数而非条数组 batch，长度桶压到几十个 | `kev --length_sort`；`lexmount/WebJev` 8,192 padded tokens/批 | 中。需要 batch>1 才有意义 |
| 取数移出主路径：全内存预 tokenize + 预取 micro-batch | `kev`、torchtitan PR #5569 | 中 |
| 头与 backbone 分开学习率、两段式先冻结后解冻 | Unsloth `decision.py` `head_learning_rate`、`freeze_backbone` | 低（当前不训 backbone） |
| packing / padding-free 的 2–3× | Unsloth 博客 | 低。官方写明 batch=1 时无 padding，收益不适用 |
| QLoRA 的 2× | Unsloth 发布说明 | 不适用。其加速都让 backbone 参与反传 |

## 未找到

- TypeSafe 官方 Jev 训练代码。社区清单 `AbdelStark/awesome-typesafe-jev` 明确写部分项目「training code not published」。
- 任何仓库做「预计算冻结骨干 hidden states 再训头」。按 2048 token × 2560 维 × bf16 ≈ 10 MB/条估算，百万条约 10 TB，不实用；共享 prefix 一次前向是同一目标的可落地版本。
- Unsloth 对 frozen backbone + 自定义 251K 头的吞吐数字；其对 Qwen3-Reranker-4B 的支持也未在文档中确认。

## 已实现并实测（2026-11-03）

采用「batch 维并行」而不是 4D mask：transformers 5.21 的 Qwen3 不再接受 4D attention mask，改为把 prefix KV cache 沿 batch 展开 `option_batch` 行，每行 = prefix + 一个 option + marker，一次前向出 K 个分数。语义与 chunk=1 相同（选项之间互不可见）。

实现：`DecisionModel.option_batch` + `_forward_options_batched`；`train_residual.py --option-batch`；`eval_gate.py --option-batch` 可覆盖 checkpoint 配置。测试：`tests/test_option_marker.py` 23 passed。

实测（rbh101 H100，D20K checkpoint，60 个 JevBench choice/score 项，预热后取两次最好）：

| option_batch | items/s | 加速 | 峰值显存 |
| --- | --- | --- | --- |
| 1 | 6.31 | 1.00× | — |
| 16 | 15.43 | **2.44×** | 17.0 GiB |
| 32 | 15.43 | 2.44×（K=5 时与 16 等价） | — |

一致性：同一 checkpoint 的 231 题 JevBench，`option_batch=16` 与顺序路径 cc 完全相同（0.4603174603）、acc 0.6450、hard 0.4775、easy 1.000。单条 logit 最大差 0.026，是 bf16 不同 GEMM 形状的数值差，没有改变任何判定。

结论：`option_batch=16` 可作为默认；4 epoch/20K 的墙钟时间从约 4 h 降到约 1.6 h。

## 建议的下一个实验

实现 `forward_choice_joint`：一次前向编码 `prefix + Σ(option_i + marker)`，用 block-diagonal mask 隔离选项，marker 位置 gather 打分。验收：

1. 输出与现有 chunk=1 路径一致：fp32 下同一权重、同一批 item，最大绝对差 < 1e-4；
   bf16 下 cache-rounding 差异可达 ~0.03（单个 option 概率），1e-4 的 logit
   检查不可达，因此 bf16 改用决策级检查：231 题 JevBench 的 cc 完全相同，
   与 `option_batch` 的验收方式一致；
2. 吞吐 ≥ 2×（6.6 items/s → ≥ 13 items/s）；
3. JevBench 231 的 cc 不变（差 < 0.005）。

若第 1 条不成立，说明 mask/位置编码有误，不能替换。第 2、3 条只决定是否默认启用。

## 参考

- https://github.com/unslothai/unsloth/releases/tag/v0.1.904-beta
- https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth
- https://github.com/unslothai/unsloth/blob/main/unsloth/_vendor/clef/joint_schema_model.py
- https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision_from_lm.py
- https://github.com/jaredpalmer/kev
- https://github.com/Mapika/decider
- https://github.com/pytorch/torchtitan/pull/5569
- https://github.com/ikermoel/open-alternative-jev
- https://github.com/lexmount/WebJev
- https://github.com/shibing624/train-jev
