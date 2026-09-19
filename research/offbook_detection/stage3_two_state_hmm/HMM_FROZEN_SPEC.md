# Stage 3 双状态 HMM 冻结实施规格

## 1. 隔离与数据边界

- 状态：`FROZEN_FOR_IMPLEMENTATION`。
- 主随机种子：`42`。
- 本实验位于独立的 `stage3_two_state_hmm` 目录，不修改或覆盖已冻结的单变点模型及其产物。
- 输入只读复用 `projected_pca16_seed42`；不重新拟合 PCA。
- PCA 由 Stage 2 train split 的整局有效决策拟合；HMM 仍只使用 train 训练。
- 每个原始棋局继续构造黑、白两个目标视角，沿用既有 train/validation/test 划分。

## 2. HMM 观察序列

从投影缓存中仅保留同时满足以下条件的节点：

- 目标棋手的非 pass 决策（缓存已经完成该筛选）；
- 目标棋手决策序号 `3 <= decision <= 19`；
- `strict_ply <= 38`，包含 38。

不设置单变点模型的最短前段或最短后段限制。截取后所有序列都参与 HMM 拟合。

## 3. 模型

- 两个未命名隐藏状态，训练前不赋予谱内/谱外语义。
- 初始状态分布自由学习，不限制从哪个状态开始。
- 完整可学习的 2×2 转移矩阵，允许 A→A、A→B、B→A、B→B。
- 主模型：16 维对角 Student-t 发射，固定自由度 5。
- 公平基线：16 维对角 Gaussian 发射。
- 尺度参数为 `softplus(raw_scale) + 1e-4`。
- 使用 log-space 精确 forward 边际似然训练；后验使用精确 forward-backward；路径使用 Viterbi。

## 4. 初始化与优化

- 5 次确定性初始化，运行 seed 为 42、43、44、45、46。
- 每次初始化对全部 train 开局节点执行二类 MiniBatchKMeans；聚类标签不带语义。
- 两个分布族共享相同的初始中心、尺度、初始状态概率和转移概率。
- 初始状态概率 `[0.5, 0.5]`。
- 初始转移矩阵 `[[0.9, 0.1], [0.1, 0.9]]`，随后四项全部自由学习。
- Adam，learning rate `3e-3`，无 weight decay。
- batch size 2048，FP32，不使用 AMP。
- 最多 200 epochs；validation NLL/decision 早停，patience 15，最小改善量 `1e-5`。

## 5. 选模和 test 闸门

- 每个分布族在 5 次初始化中按 validation NLL/decision 选择最佳 checkpoint。
- Student-t 与 Gaussian 再按相同 validation NLL/decision 选择主模型。
- 状态占用、切换、熵、坍缩和随机跳动指标用于人工验收，不临时改变 NLL 排名。
- 若 validation 呈现明显坍缩或随机跳动，停止并报告，不运行 test。
- 状态命名、配置、checkpoint 和输出规则冻结且经用户确认后，才允许执行一次 test。
- test 一次性评估两个已冻结族；主模型身份不得依据 test 改变。

## 6. 训练后状态命名

对每个族的最佳 checkpoint 分别只使用 train 后验命名：

1. 长度小于 3 的序列仍参与训练，但不参与命名。
2. 对其余序列按 `numpy.array_split` 等价规则切成连续三段。
3. 第一段为前期，第三段为后期，中段不参与命名。
4. 在所有前期节点和所有后期节点上分别计算两个原始状态的平均后验。
5. 比较两种映射的“前期谱内均值 + 后期谱外均值”，选择较大者。
6. 若分数完全相同，固定原始状态 0 为谱内。
7. 映射写入冻结 manifest；validation/test 不得重新交换。

## 7. 后验和输出定义

边界后验定义为：

```text
xi[t,i,j] = P(z[t-1]=i, z[t]=j | x[1:T]), t >= 1
```

它附着在转入后的当前决策。首个决策的两个跨状态转移后验均为 `null`。

每个节点输出原始节点编号、strict ply、目标决策序号、两种语义状态后验、Viterbi
状态、`P(in→off)` 和 `P(off→in)`。每条目标视角序列输出完整路径、全部 Viterbi
转换、全部 Viterbi 谱外区间及切换次数。

`primary_offbook_entry` 仅在 Viterbi 路径至少包含一次 `in→off` 时输出；否则为
`null`。存在该类转换时，它取整条序列中 `P(in→off)` 最大的边界，仅作为诊断摘要，
不得替代完整路径或其他转换。

## 8. 验收指标

- 后验软状态占用与 Viterbi 硬状态占用；
- 学得的四种转移概率与后验期望转移计数；
- 每序列平均 Viterbi 切换次数；
- 全程单状态序列比例；
- 至少三次切换的多次往返序列比例；
- 前期/后期平均状态后验；
- 每节点二元状态后验熵（nats），同时给出节点加权和序列等权均值。

30 局抽样、HMM 审核页面和最终 delivery 暂不在本轮范围内。
