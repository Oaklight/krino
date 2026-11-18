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

一个补齐形状胜过八次单独 forward：八行补齐到共同宽度、6.7% padding，冷 0.696/1.721、热 0.158/0.388，而八次单行 forward 在热形状上为 0.384/0.821，所以分桶同时去掉冷成本并摊薄每次调用成本。分桶路径（PR #400，分支 `feature/bucketed-forwards`）在 a20k 上、每 rank 预热 14.8 到 15.0 s 后取三步剖面（`p2_bucket_a20k`）：forward 0.433 / 0.433 / 0.290，backward 1.065 / 1.072 / 0.698，all-reduce 3.942 / 4.049 / 1.613；同节点交付路径为 1.810 / 8.632 / 1.379。第三步算力为 0.99 s 对 10.44 s，端到端约 2.7 s 对 11.9 s，但这是三步中的一步而不是均值。同配置的二十步运行（`p2_bucket_a20k_20`）把 all-reduce 的走势定下来：八个 per-rank 剖面里十四步在 3.93 到 4.42 s，五步（3、11、13、15、17）在 1.62 到 1.82 s，第 20 步 0.85 s，二十步均值 3.34 s。三步剖面里的 3.94 / 4.05 落在高组、1.61 落在低组，算力均值三步 1.330 s 对二十步 1.333 s，forward 加 backward 加 reduce 三步 4.532 s 对二十步 4.669 s（相对交付路径为 7.85 倍与 2.61 倍，按三步均值计），二十步运行的 all-reduce 均值 3.34 s 约为 0.31 s 传输下界的 10.8 倍（3.34 / 0.31）。同一次运行内 payload 固定——可训练梯度在各步之间不变——所以同一 payload 下 0.85 到 4.42 s 的散布不可能是传输时间，只能是等待；而二十步运行里最好的一步 0.85 s 仍是 0.31 s 下界的约 2.7 倍，即最快的集合通信也没有跑到带宽。按 bucket-forward 次数分组后，同组内的 all-reduce 仍然散布：次数为 10 的十一步（1、2、3、6、7、10、14、15、17、18、19）里，逐 rank 最慢的一次集合从 2.123 到 5.378 s，差 3.26 s；前向路径本身不发集合通信，因此前向流量竞争不能解释这一组内的差。八个 per-rank 的值把等待与传输分开：延迟进入集合的 rank 测得的是它自己在集合内的时间，最早到达者测到全程加等待，最后到达者接近集合本身。两步给出同一个自身时长：第 1 步（十个 forward、到达散布 4.75 s）最小 0.399 s，第 3 步（十个 forward、到达散布 1.75 s）最小 0.400 s，而两步的 all-reduce 均值相差约 2.6 倍（4.24 s 对 1.62 s），所以集合本身的时长在固定 payload 下不变，约为 0.40 s，即 0.31 s 下界的 1.3 倍；其上是等待，第 1 步 4.98 s（对 4.75 s 的散布），第 3 步 1.72 s（对 1.75 s）。写成式子即 T ≈ 0.40 s 恒定、wait = max − T；这一不变性排除了任何依赖该步形状的解释。八个值如下，列序按到达先后排，不是 rank 号：

| 步 | 到达偏移（s） | 集合内时间（s） |
| --- | --- | --- |
| 1 | 0.00 / 0.00 / 0.00 / 0.50 / 0.50 / 1.00 / 1.50 / 4.75 | 5.209 / 5.324 / 5.378 / 4.814 / 4.804 / 4.371 / 3.642 / 0.399 |
| 3 | 0.00 / 0.00 / 0.00 / 0.50 / 0.50 / 0.50 / 0.50 / 1.75 | 2.086 / 2.104 / 2.123 / 1.518 / 1.577 / 1.547 / 1.585 / 0.400 |

这一步分解是推断而非剖面自身的算术，它假定 per-rank 值就是该 rank 在集合内的时间；检查与该假定一致——最早到达的 rank 持最大值，最后到达的 rank 持 0.399 与 0.400 s，两步相差 1 ms。这条运行定论的是 all-reduce 的走势，不是成本拟合，后者另有其 203 个 forward 的样本。工作线程里引用过三步中最好的一步，正式记录应以均值为准。

算力塌缩后 all-reduce 成为最大项，且仍在等待而非搬运：其传输下界是 6.78 GB 乘 1.75 的 ring 因子除以 38.6 GB/s（拟合渐近值，与同一扫描在 256 MiB 处实测的 38.32 GB/s 在其精度内一致），约 0.31 s，三步剖面（`p2_bucket_a20k`）里最好的一步是下界的五倍；ring 扫描给出 1 / 4 / 16 / 64 / 256 MiB 的 0.15 / 0.28 / 0.83 / 3.08 / 12.26 ms，即 12.23 / 26.21 / 35.37 / 38.13 / 38.32 GB/s，1.75 倍因子已计入；deal 的成本模型已从单点定标改为二次拟合：每个 forward 连同其 backward 的成本为 `0.0219 s + 1.134e-4 × padded token + 1.358e-9 × padded token²`，拟合值为 0.021870 / 1.134163e-4 / 1.358e-9，trainer 中写作 0.0219 / 1.134e-4 / 1.358e-9。拟合对象是提交在 `model/scripts/bucket_cost_fit.tsv` 的 203 个 padded forward：表的 rank 与 step 列把这 203 个 forward 归到 160 个 rank-step 上，即每 rank-step 1.27 个 forward，来自 43 个 rank-step 各出现两个 forward 的情形，而这 86 行全部是 512 宽，即同一个桶拆成两次 forward，不是两个桶；剔除 4 个超过其桶均值 1.5 倍的样本后取 199 个，样本内 rms 0.025693 s。留出用 k = 5、折大小 41/41/41/40/40（五个逐折 rms 由 `fit_bucket_cost.py --per-fold` 打印）；折号按表内行序取 `index % k`，表的行键是 `(rank, step, width, presentation)`，其中 `presentation` 让 43 个带两个 forward 的 rank-step 有确定次序，否则两行可互换。表是 rank 主序、同一 rank 内步号递增，所以 `index % 5` 在 rank 内逐步行进，不是逐宽度轮替。每折都含各宽度的行但不等量（4096 宽在五折中为 3/3/2/4/2），四个被排除的行落在第 0、2、4 折，第 1、3 折的测试集不含它们。逐折样本外 rms 从 0.0241 到 0.1849，故本段引用合并值 0.120956；被排除的行占样本外平方误差的 95.2%，逐折 rms 的差异主要就是它们的多少。把留出误差限制在 199 个未排除行上，rms 为 0.026822 s，与样本内 0.025693 s 只差约 1 ms；四个被排除行的留出 rms 为 0.841 s。这两个数与 0.120956 同属一次留出预测（逐折 `index % k`、训练折按 1.5 倍规则过滤、203 行全部评分）；若改用样本内口径（系数拟合在 199 个保留行上再对全部行预测），同一三量为 0.025693 / 0.840420 / 95.56%，差别只在口径，读者复算时会看到这一对。这四个行全部是 512 宽，落在采样最多的桶内，所以它们不是覆盖缺口，而是同一桶内的另一种状态；排除规则因此丢弃它们，而不是让拟合去平滑，增加 512 的 forward 也不会消除它们。所以这三个数要一起读：合并 0.121 s，其中 95% 来自四个 forward；其余 199 行约 0.027 s。样本外 rms 0.120956 s（训练折按同一规则过滤），训练折不过滤时 0.120224 s，两者差 0.0007 s 且方向对剔除不利，即剔除在样本外没有可测收益；把同系数用在全部 203 个上为 0.120684 s，全部 203 重拟合并自评为 0.119427 s，0.12 s 量级由四个被排除的慢 forward 主导，不是典型误差。留出证据的要害是形式而非系数散布：线性式截距为负，样本外逐宽度偏差在 2048 为 −0.212 s、在 4096 为 +0.129 s，二次式的对应值为 −0.008 s 与 −0.0006 s，宽度相关偏差在留出上消失。支撑偏向便宜端：73.4% 的拟合 forward 是 512 宽，4096 桶以 14 个 forward 贡献 30.3% 的拟合 padded token，截距几乎只由 512 记录决定，是三个系数中支撑最弱的一个。一次 forward 的拟合不等于一步：一步还携带 all-reduce 与各 rank 到达偏斜，因此 deal 的输出是分区次序，不是预测的步时间；调用项、斜率与二次项之间的分配改变两个桶的建模成本比，也改变一个分区能负担的 forward 次数。按该拟合，2048 over 1024、batch_rows 8 的单次 forward 成本比为 2.15（旧两常数下 1.90），512 到 4096、batch_rows 8 为 10.2（旧两常数下 6.73）；该比值随 batch_rows 上升：2048 over 1024 在 batch 1 为 1.86、batch 4 为 2.05、batch 8 为 2.15，因为每次 forward 都按满 batch 与桶宽计费，batch 越大，宽桶相对窄桶越贵，可达到的配平越差。比值随 batch 增大趋向宽度比平方（该对为 4，512 到 4096 为 64），但二次项要到 padded token 超过 1.134e-4 / 1.358e-9 = 83,505 才压过斜率项，而拟合支撑里最宽的点是 32,768，所以平方极限是支撑之外的推断，不是观测。拟合的 203 行每行的 padded token 都等于 8 倍宽度，batch 8 是唯一测过的 batch，因此 1.86 与 2.05 是模型在未测 batch 上的算术，不是测量结果。分配还受形状约束：`bucket_cost` 对每次 forward 都按桶宽和满 batch 计费，batch 维度也是形状的一部分，形状不变才留热，所以一个 rank 的一步由它所持最宽桶决定，各 rank 可达到的 max/mean 受桶宽比限制，进一步拉平只能放弃形状复用。各 rank 到达离散仍是开放问题，因此不能声称偏斜已解决。

设计规则因此改为：固定桶宽 512 / 1024 / 2048 / 4096，把每行放进能容纳它的最小宽度，桶内补齐，每 rank 每桶一次 forward，启动时预热这些形状并记录 `prewarm_seconds_this_rank`，按建模的 bucket-seconds 配平。1.25 长度比规则作废，因为混合长度会最大化冷形状数，而这一点在形状探针之前不可见；不过 main 上 `CHUNK_LENGTH_RATIO` 仍默认 1.25、`--chunk-ratio` 也仍默认为 1.25，所以在分桶路径落地前，被撤回的规则仍是 trainer 的默认行为；上一版的每 token 模型、1.94x 上限与 1.363 到 1.05 的配平估计一并作废，原因相同。等价性方面，`--lr 0` 下分桶与交付路径记录的步损失相对差在 1e-6 以内，所以这项改动只改变速度、不改变训练；机制仍未确认。来源：分桶剖面读自八个 per-rank 的 `phase_profile_rank<N>.json`；形状表来自树外的形状探针脚本输出；成本拟合读自合并后的 `model/scripts/bucket_cost_fit.tsv`，并由同目录的 `fit_bucket_cost.py` 复算；预热时间按次运行记录为 `prewarm_seconds_this_rank`；`--lr 0` 的等价性由 PR #400 在 `tests/test_panel2_lora_rows.py` 中新增的 `test_bucketed_and_shipped_paths_record_the_same_history` 覆盖；同文件已有的 `test_chunking_does_not_change_the_step_losses` 是另一项检查。

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
