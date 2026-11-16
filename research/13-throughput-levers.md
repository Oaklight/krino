# 吞吐杠杆：Jev 类仓库与 Unsloth 的调研

日期：2026-11-03。范围：公开文档与仓库文件的只读调研，未运行他人代码。

## 我们的现状

- D20K：frozen Qwen3-Reranker-4B + 251,073 参数 marker 头，chunk=1，max_length 2048，batch=1，bf16。
- 实测稳态 6.6 items/s，`max_memory_allocated` **8.1 GiB**（<accelerator> NVL 96 GiB）。
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

实测（gpu-host-a <accelerator>，D20K checkpoint，60 个 JevBench choice/score 项，预热后取两次最好）：

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

## 实测步剖分与吞吐计划（2026-11-16）

ce 臂、8 rank、每 rank 8 行、每行 478 token、4096 token 上限、64 MB bucket 的 20 步剖分，分项之和与墙钟一致（分项合计 12.016 s/step，对 12.05 s 的墙钟，残差 0.034 s；mix 剖面合计 11.970，逐项相差不超过 0.05 s），这是已交付路径的形状，不是 LoRA 的形状：

| 相位 | s/step | 占比 |
|---|---:|---:|
| tokenise | 0.012 | 0.1% |
| forward | 2.656 | 22.1% |
| backward | 7.065 | 58.8% |
| all-reduce | 2.161 | 18.0% |
| clip | 0.014 | 0.1% |
| optimizer | 0.108 | 0.9% |
| 合计 | 12.016 | 100% |

backward 占 58.8%，all-reduce 占 18.0%；tokenise 每行 1.5 ms，因此预分词方案排除。尺寸扫描（1 MiB 0.15 ms、4 MiB 0.28、16 MiB 0.83、64 MiB 3.08、256 MiB 12.26 ms，对应 ring 带宽 12.23、26.21、35.37、38.13、38.32 GB/s；ring all-reduce 搬运 2(n-1)/n 倍载荷，n=8 时为 1.75 倍，因此 256 MiB 点的 21.9 GB/s 载荷即 38.32 GB/s 的 ring 流量，早先的 38.6 也落在拟合精度内）拟合出约 0.07 ms 的延迟截距，64 MiB 处已接近 38.32 GB/s 的渐近值，再增大 bucket 收益不足 1%，120 次调用的启动开销合计约 8 ms。正确的分母是可训练梯度量 6.78 GB（6,782,576,640 字节），而不是含未训练参数的 8.04 GB checkpoint；它装在 120 个约 56 MB 的 bucket 里，bucket 路径整步 0.345 到 0.838 s，其中 torch.cat 0.007 s、copy-back 0.011 s，collective 跨 rank 0.327 到 0.820 s；最快的 rank 为每 bucket 2.7 ms，与独立扫描同速，因此本地拷贝约占 5%，分配与拷贝都不是杠杆。2.161 s 是在等待而不是在搬运：backward 加 all-reduce 在各 rank 上几乎恒定，9.14 到 9.37 s，两项却分别有 0.66 s 与 0.89 s 的离散并互相抵消，同一批字节在 step 1 到 2 运行 0.52 到 1.65 s、在 step 15 到 20 运行 1.67 到 2.56 s；各 rank 拥有不同行，每次 collective 等最后一个 rank 到齐，因此该相位随行长方差增大；这是可能机制而非已证结论，区分办法是把被追踪的 collective 时间按步画出并与该相位对照：若 collective 时间持平而相位增长，就是在等最慢 rank，若 collective 时间也同步增长，则是 collective 本身在负载下退化，配平行不会修复后者。所以杠杆是把一步内的行在各 rank 之间配平，这与合批是同一项改动，应并入该工作而不是给 reduce 单独立项；按计划测算，把每步 64 行按 token 数分给各 rank 可把最忙 rank 与均值之比从 1.363 降到 1.05，这是尚未上机验证的开放问题。上限方面（ce 剖面）：forward 加 backward 为 9.721 s，算力部分改善 2.5x 得 3.888 s，加 2.161 s 的 reduce 与 0.134 s 其他项为 6.183 s/step，端到端 1.94x；若要端到端 4x 则必须同时改 reduce；算力与等待会一起移动，这比单看 MFU 更有理由先做合批。追踪 caveat：原位追踪在 cat 之后、collective 之后和 copy-back 之后各加一次 synchronise，因此它只用于分解各段，不给绝对步时。来源说明：本节数字读自八个 rank 的逐行日志，而不是产物文件，因为八个 rank 写同一个 `phase_profile.json`，该文件在 1,040 字节处被截断；一 rank 一个文件才是应有的记录，#396 落地后本节可改指 `phase_profile_rank<N>.json`。

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
