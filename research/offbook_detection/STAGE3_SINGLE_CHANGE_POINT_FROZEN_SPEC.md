# 第三阶段单变点模型冻结实施规格

## 1. 状态与输入

- 状态：`FROZEN_FOR_IMPLEMENTATION`
- 随机种子：`42`
- 输入：第二阶段完整 `sequence_cache_61145_h128_seed42`
- 时间 Transformer、空间 Transformer 和两个既有缓存全程冻结。
- 不使用旧人工锚点或 reviewerMark 训练、调参或选模。

## 2. 观察序列与候选集合

每个原始棋局分别构造黑方、白方两个目标视角。单变点模型的观察序列只保留：

```text
actor_is_target == true and is_pass == false
```

观察序列保留该棋手直到终局的全部有效决策，不按 `strict_ply` 截断。

令一局共有 `m` 个有效决策，`tau=k` 表示第 `k` 个有效决策开始进入后段：

```text
前段 = 1 ... k-1
后段 = k ... m
```

候选必须同时满足：

- `3 <= k <= 19`；
- 第 `k` 个决策对应的 `strict_ply <= 38`，包含 38；
- `m-k+1 >= 3`，即后段至少三个有效决策。

没有有限合法候选的序列只允许 `no_change`。

## 3. 节点表示

节点输入为第二阶段最终层 hidden state 与时间旁路的拼接：

```text
[h_i(128), thinking_time(1), target_remaining_time_after(1), opponent_remaining_time_after(1)]
```

三个时间字段沿用第二阶段 train-only 的 `log1p + median/IQR` 标准化。使用全部 train
有效决策节点拟合 16 维 Incremental PCA；PCA 随后冻结，Student-t 与高斯共享同一投影。

## 4. 概率模型

显式枚举 `no_change` 和所有合法的唯一变点。所有棋手共享参数，不输入棋手身份。

- 主模型：前段、后段各一个对角 Student-t，固定自由度 `nu=5`；
- 基线：前段、后段各一个对角高斯；
- 对角尺度：`softplus(raw_scale) + 1e-4`；
- `P(no_change)=0.5`；
- 有限变点总先验为 `0.5`，在每局合法候选中均匀分配。

训练目标为 train split 的整局边际负对数似然，按有效决策节点数聚合。无有限候选的序列
不用于参数优化或 validation 选模，但仍在最终输出中报告为 `no_offbook`。

## 5. 训练、选模与数据边界

- 优化器：AdamW；
- 学习率：`1e-3`；
- weight decay：`1e-4`；
- batch size：`512`；
- 最多 `100` epochs；
- validation 每有效决策节点 NLL 早停，patience `10`；
- Student-t 与高斯分别保存各自 validation 最佳 checkpoint；
- 正式模型由两者的最佳 validation 每节点 NLL 决定；
- test 不参与训练、早停、参数选择或模型选择；全部冻结后仅评估一次。

## 6. 后验与报告

每局输出全部合法候选后验及 `P(no_change)`。若 `no_change` 是所有状态中最大的后验，输出
`no_offbook`；否则输出 MAP 有限候选。另报告：

- MAP 目标棋手决策序号、原始节点和 `strict_ply`；
- MAP 后验概率；
- 后验熵；
- 有限候选条件后验下 strict-ply 的均值和标准差；
- game ID、目标视角、执色和节点映射；
- 模型版本、配置、数据哈希、PCA 哈希和随机种子。

首版不另设人为置信阈值。
