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

## 形状驱动的步成本与分桶计划（2026-11-17 更正）

成本按不同的序列形状计，而不是按 token 或按 forward 计。形状探针在计算节点、8 rank 上测得每 rank 每步秒数，step 1 冷启动对 step 2 重放同一形状：

| 不同形状数 | 首步 forward | 首步 backward | 次步 forward | 次步 backward |
|---|---:|---:|---:|---:|
| 1 | 0.389 | 2.081 | 0.386 | 0.826 |
| 2 | 0.836 | 3.045 | 0.390 | 0.819 |
| 4 | 1.676 | 5.008 | — | — |
| 8 | 3.418 | 8.974 | 0.384 | 0.821 |

每多一个不同形状，首次出现约多 0.45 s forward 与 0.99 s backward，而热步在四条腿上持平，因为 token 工作量相同。这不是分配器增长：八形状腿在 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 下为 3.392/9.199，对 3.418/8.974，落在 rank 离散内；剩余候选是逐形状的 kernel 或 attention 后端选择，未确认，因此不指名原因。

一个补齐形状胜过八次单独 forward：八行补齐到共同宽度、6.7% padding，冷 0.696/1.721、热 0.158/0.388，而八次单行 forward 在热形状上为 0.384/0.821，所以分桶同时去掉冷成本并摊薄每次调用成本。分桶路径（PR #400，分支 `feature/bucketed-forwards`）在 a20k 上、每 rank 预热 14.8 到 15.0 s 后取三步剖面：forward 0.433 / 0.433 / 0.290，backward 1.065 / 1.072 / 0.698，all-reduce 3.942 / 4.049 / 1.613；同节点交付路径为 1.810 / 8.632 / 1.379。第三步算力为 0.99 s 对 10.44 s，端到端约 2.7 s 对 11.9 s，但这是三步中的一步而不是均值，二十步运行尚未完成。把三步一并计入，算力均值为 1.330 s、forward 加 backward 加 reduce 为 4.532 s，对交付路径为 7.85 倍与 2.61 倍；第 1、2 步仍带 3.9 到 4.0 s 的 all-reduce，而第三步只有 1.6 s，这正是二十步运行要定论的部分。工作线程里引用过三步中最好的一步，正式记录应以均值为准。

算力塌缩后 all-reduce 成为最大项，且仍在等待而非搬运：其传输下界是 6.78 GB 乘 1.75 的 ring 因子除以 38.6 GB/s（拟合渐近值，与同一扫描在 256 MiB 处实测的 38.32 GB/s 在其精度内一致），约 0.31 s，即便好的一步也是下界的五倍；ring 扫描给出 1 / 4 / 16 / 64 / 256 MiB 的 0.15 / 0.28 / 0.83 / 3.08 / 12.26 ms，即 12.23 / 26.21 / 35.37 / 38.13 / 38.32 GB/s，1.75 倍因子已计入；deal 的成本模型（每次调用 0.10 s 加每个 padded token 0.00011 s）与各 rank 到达离散仍是开放问题，因此不能声称偏斜已解决。

设计规则因此改为：固定桶宽 512 / 1024 / 2048 / 4096，把每行放进能容纳它的最小宽度，桶内补齐，每 rank 每桶一次 forward，启动时预热这些形状并记录 `prewarm_seconds_this_rank`，按建模的 bucket-seconds 配平。1.25 长度比规则作废，因为混合长度会最大化冷形状数，而这一点在形状探针之前不可见；不过 main 上 `CHUNK_LENGTH_RATIO` 仍默认 1.25、`--chunk-ratio` 也仍默认为 1.25，所以在分桶路径落地前，被撤回的规则仍是 trainer 的默认行为；上一版的每 token 模型、1.94x 上限与 1.363 到 1.05 的配平估计一并作废，原因相同。等价性方面，`--lr 0` 下分桶与交付路径记录的步损失相对差在 1e-6 以内，所以这项改动只改变速度、不改变训练；机制仍未确认。来源：分桶剖面读自八个 per-rank 的 `phase_profile_rank<N>.json`；形状表来自树外的形状探针脚本输出；预热时间按次运行记录为 `prewarm_seconds_this_rank`；`--lr 0` 的等价性由 PR #400 在 `tests/test_panel2_lora_rows.py` 中新增的 `test_bucketed_and_shipped_paths_record_the_same_history` 覆盖；同文件已有的 `test_chunking_does_not_change_the_step_losses` 是另一项检查。

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
