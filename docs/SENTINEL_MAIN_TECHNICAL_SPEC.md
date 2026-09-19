# Sentinel 主功能完整技术规格

文档状态：按当前实现整理  
主功能版本：Sentinel V1 / Reference v11-matchup600  
当前 Reference 配置：`sentinel_reference_config.json`（v11-matchup600-20260911）  
最后核对日期：2026-09-11

## 1. 文档目的与范围

本文档描述 Player Analysis Toolkit 当前 Sentinel 主功能的完整算法、数据合同、判定规则、
运行阶段和审计约束。它覆盖从最近对局获取、Level22 分析、确定性脱谱、二维 Reference
评分、伪玩家完整扫描、候选冻结，一直到条件式个人模型复核、非模型复核和最终报告。

本文档刻意不重复“行棋质量等价 Elo / estimated Elo”的计算方法。该功能是与主异常扫描
并存的独立分析模块，详见：

- `docs/SENTINEL_ESTIMATED_ELO_IMPLEMENTATION_SPEC.md`
- `docs/SENTINEL_ESTIMATED_ELO_ACCEPTANCE_REPORT_20260819.md`

本文中的“Elo 匹配”仅指 Sentinel 主扫描为了估计条件 Reference 期望而进行的双方 Elo
二维分箱与插值，不是 estimated-Elo 反推。

Sentinel 的输出不是作弊概率，也不是对作弊事实的自动裁决。主扫描回答的问题是：目标账号
最近一组对局中，是否存在相对于同颜色、相同分析 scope、相近双方赛前 Elo 的 Reference，
表现为异常低 `loss_ge4_rate` 的单局、集中前缀或整体模式。

## 2. 当前实现的来源优先级

当旧说明和当前程序不一致时，以以下来源为准：

1. `src/player_analysis_toolkit/sentinel.py`：主统计算法和数据结构。
2. `scripts/analysis/detect_offbook.py`：确定性脱谱算法。
3. `scripts/analysis/sentinel_analysis.py`：Reference 构建及兼容 CLI。
4. `scripts/analysis/sentinel_unified_analysis.py`：统一单选手 Sentinel 命令面。
5. `scripts/analysis/run_player_investigation.py`：生命周期、条件分支和报告编排。
6. `sentinel_reference_config.json`：当前正式 Reference 指针和冻结参数。
7. 当前派生 Reference 的 `reference_build_audit.json` 与 SHA-256 manifest。

`METHOD.md` 和本文件后部的历史章节可能保留较早版本的实验数字。当前正式快照以
`sentinel_reference_config.json`、其派生 Reference 的 `reference_build_audit.json` 和
SHA-256 manifest 为准；本文第 6 节记录当前审计快照。

## 3. 入口、职责与阶段总览

### 3.1 正式生命周期入口

正式单选手运行入口是：

```powershell
python scripts/analysis/run_player_investigation.py start-sentinel `
  --account example_player `
  --output-dir "C:\path\example-player-sentinel" `
  --reference-config sentinel_reference_config.json
```

`run_player_investigation.py` 只负责生命周期编排、断点恢复、输入输出核验和最终报告组装；
主异常检测公式不实现在编排器中。

与单个选手调查有关的 Sentinel 命令统一经
`scripts/analysis/sentinel_unified_analysis.py` 暴露。旧
`scripts/analysis/sentinel_analysis.py` 仍保留 `acquire`、`build-reference`、`score`、
`scan`、`freeze` 兼容和维护接口。

### 3.2 主功能阶段

| 顺序 | 阶段 | 核心作用 | 主要输出 |
|---:|---|---|---|
| 1 | `acquisition` | 获取棋谱、剔除零坐标落子局、选择最近最多 30 局 | `account_bundle.json`、`selected_account_bundle.json` |
| 2 | `profile` | 获取目标与相关玩家的 OQ Profile 快照 | `profiles/` |
| 3 | `level22` | 对选择局运行固定 Level22 分析 | `engine_level22/` |
| 4 | `offbook_detection` | 对目标方运行确定性脱谱算法 | `offbook_records.json` |
| 5 | `sentinel_reference_scoring` | 构造目标方记录、匹配 Reference、计算每局残差 | `per_game_reference_scores.json/csv` |
| 6 | `sentinel_pseudo_scan` | 伪玩家抽样、完整 k 扫描、偏差校正和分类 | `pseudo_scan_replicates.csv`、`pseudo_scan_summary.json`、`sentinel_scan_results.json` |
| 7 | `sentinel_group_freeze` | 冻结举报组、对照组、参数和输入哈希 | `selection_manifest.json`、`model_review_groups.json` |
| 8 | 后置统一分析 | 合并主 Sentinel 摘要与其他独立分析产物 | 单独的统一输出；独立算法不在本文展开 |
| 9 | 条件式 Hint/模型复核 | 仅在存在正式举报组且模型对照不少于 8 局时运行 | `hints/`、`model/` |
| 10 | 条件式非模型复核 | 有正式举报组时运行子损、WLD、用时分析 | `non_model/` |
| 11 | `final_report` | 组装最终 Sentinel 报告 | `report.json` |

## 4. 关键术语和方向约定

### 4.1 directed record

一局棋按目标方视角拆成记录。Reference 中同一局必须分别生成黑方目标和白方目标两条
directed record；调查目标账号每局只生成该账号所在颜色的一条记录。

### 4.2 analysis scope

Sentinel 主指标有两个互不混用的 scope：

- `post_offbook_inclusive`：存在算法脱谱锚点，从锚点手开始，包含锚点手。
- `full_game_fallback_no_offbook`：没有可信锚点，使用目标玩家整局全部实际着手。

Reference 匹配不得跨颜色或跨 scope 借用分布。

### 4.3 loss 与强度方向

节点子损统一定义为：

```text
disc_loss = max(0, lossClipped)
```

主指标为：

```text
loss_ge4_rate = 目标 scope 内 disc_loss >= 4 的有效节点数 / 有效 loss 节点数
```

Reference 外部强度残差定义为：

```text
externalStrengthResidual
  = referenceExpectedLossGe4Rate - actualLossGe4Rate
```

因此正值表示调查局的 `loss_ge4_rate` 比匹配 Reference 更低，即按本指标表现得更强。
负值表示比匹配 Reference 更高。

### 4.4 统计量不是作弊概率

字段名中的 `NormalExceedanceRate` 是伪玩家经验超越率，表示 Reference 生成的正常伪玩家
达到或超过当前统计量的频率。它不是“正常概率”的完整贝叶斯后验，也不是作弊概率。

## 5. 调查对局获取与选择算法

### 5.1 输入

`acquire` 可以从 Othello Quest 获取账号 bundle，也可以通过 `--bundle` 使用已有 bundle。
正式 Sentinel 当前只允许 `5min` 模式。

### 5.2 目标账号映射

账号比较使用规范化后的 `account_key`。每局必须能把目标账号映射到恰好一方；映射不到或
同时映射到两方均视为输入错误。

### 5.3 零坐标落子过滤

程序检查 `detail.position.moves`，只要至少有一个事件的 `m` 满足大小写不敏感的
`^[a-h][1-8]$`，该局就属于有坐标落子的候选局。

过滤顺序是固定的：

1. 先保留确实包含目标账号的局。
2. 再剔除没有任何坐标落子的局，例如开局即掉线。
3. 最后才应用最近 30 局上限。

零坐标落子局不占 30 局名额，ID 会记录在
`selection.excludedZeroPlacementGameIds`。

### 5.4 最近 30 局

候选局按以下复合键降序排列：

```text
(created, gameId)
```

取前 30 局。`gameId` 是时间相同情况下的确定性稳定破平键。若有效局少于 30，则全部使用。
程序输出选择后的 bundle、逐局颜色和双方 `oldR` 目录，以及简化 CSV 元数据。

## 6. 当前正式 Reference 快照

### 6.1 配置

当前 `sentinel_reference_config.json` 的关键合同为：

| 字段 | 当前值 |
|---|---:|
| `version` | `v11-matchup600-20260911` |
| `formalEloMinimum` | 1600 |
| `dynamicFormalMaximumElo`（源 Reference） | 2571 |
| `topBinLower`（源 Reference） | 2400 |
| `eloBandWidth` | 100 |
| `formalBinCount` | 9 |
| `targetDimension` | `black_white_directed_cell` |
| `targetPerBlackWhiteCell` | 600 |
| `wldFromGlobalPlacementPlyInclusive` | 39 |
| `pseudoPlayerReplicates` | 10,000 |
| `fixedCandidateBootstrapReplicates` | 10,000 |
| `randomSeed` | 20260911501 |

正式运行入口的 CLI 默认 Sentinel 扫描 seed 为 `20260814`。实际调查以运行目录
`run_config.json` 冻结的值为准，而 Reference 配置中的 seed 描述 Reference 推荐合同。

### 6.2 当前 v11-matchup600 审计事实

来自
`research/offbook_detection/data/oq_sentinel_reference_level22_1600plus_v11_20260911/reference_build_audit.json`：

| 项目 | 数量 |
|---|---:|
| 唯一源对局 | 21,615 |
| 主矩阵对局 | 21,499 |
| 低 Elo 扩展对局 | 116 |
| 全部 directed records | 43,230 |
| 正式主矩阵 directed records | 42,998 |
| 排除的低 Elo directed records | 232 |
| `offbook` directed records | 40,334 |
| `no_offbook` directed records | 2,896 |

当前四个颜色/scope 池都至少有两条记录，审计字段
`leaveOnePseudoCalibratableScopeColors` 均为 `true`。这取代了旧版 Reference 中
`no_offbook` 池过小的历史限制。

### 6.3 Elo 分箱

当前共有 9 个轴向分箱；正式矩阵按黑棋 oldR 桶为行、白棋 oldR 桶为列：

```text
[1600,1700) center=1650
[1700,1800) center=1750
[1800,1900) center=1850
[1900,2000) center=1950
[2000,2100) center=2050
[2100,2200) center=2150
[2200,2300) center=2250
[2300,2400) center=2350
[2400,2500] center=2450
```

最后一档上界包含 2500。完整 cell 空间为：

```text
9 个黑棋 Elo 档 × 9 个白棋 Elo 档 × 2 种目标颜色 × 2 种 scope = 324 cells
```

空 cell 仍会出现在 cell summary 中，但不会进入有效匹配记录。

## 7. Level22 分析合同

### 7.1 调查运行参数

正式 Sentinel 调查强制以下参数：

- Level：22
- Console worker：12
- 每 Console 线程：16
- hash：25
- WLD 起点：`global_placement_ply >= 39`
- 支持 `--resume`

运行前会验证 worker、线程和 hash 值；偏离固定合同会直接失败。

### 7.2 Reference 的引擎合同

Reference 构建不重新运行引擎，而是读取已完成且审计通过的 Level22 JSON。冻结合同为：

- Level 22
- 12 workers
- 每 Console 16 threads
- hash 25
- 默认 book 开启
- WLD 从实际落子序号 39 起，包含 39

构建时会核对 source audit、每局引擎 JSON 的 SHA-256、`gameId`、数量和引擎合同。
派生目录不复制 Level22 文件，只保存对原文件的路径和哈希引用。

### 7.3 外部引擎边界

Egaroucid 的搜索实现不属于 Sentinel 自身算法。Sentinel 使用其已审计节点字段，包括但不限于
`bestEval`、`actualEval`、`lossClipped`、`thinkingTimeMs`、颜色、账号及实际落子序号。
Sentinel 不在报告中把引擎启发值改写为概率。

## 8. 确定性脱谱算法

算法标签为：

```text
first-log-time-or-abs6-with-post-fast-v5
```

### 8.1 动态时间阈值

设该局时限为 `T_ms`，`T_sec = T_ms / 1000`。长考候选阈值为：

```text
long_threshold_ms(T)
  = 5500 × ln(1 + T_sec) / ln(301)
```

后续快下阈值为：

```text
fast_threshold_ms(T)
  = 2000 × ln(1 + T_sec) / ln(301)
```

在 300 秒局中两者分别为 5500 ms 和 2000 ms。

比较符号不同：

- 长考候选要求 `thinkingTimeMs > long_threshold_ms`。
- 快下判定要求 `thinkingTimeMs <= fast_threshold_ms`。

恰好等于长考阈值不触发，恰好等于快下阈值算快下。

### 8.2 目标方节点

只检查目标账号自己的实际着手节点。每个节点必须具备有限数值的 ply、思考时间和
`bestEval`，颜色必须与目标方固定颜色一致。算法从 `ply >= 5` 开始，没有最大 ply 上限。

### 8.3 评价截点

评价截点是目标方第一个满足以下条件的实际着手节点：

```text
ply >= 5 and abs(bestEval) > 6.0
```

比较是严格大于，因此 `+6.0` 和 `-6.0` 均不触发。该规则使用 `bestEval` 的绝对值，
不是 loss。

### 8.4 时间候选搜索区间

若存在评价截点，时间候选只允许出现在该截点之前，严格不包含截点。若不存在评价截点，
则一直搜索到棋局结束。所有满足条件的时间候选按目标方着手顺序排列。

### 8.5 post-fast 校验

对每个候选，向后查看最多 4 个目标方着手。计算其中连续快下的最长 streak：

- 后续目标方着手少于 3 个：状态 `insufficient`，候选接受。
- 后续存在至少 3 个连续快下：状态 `rejected`，候选拒绝。
- 否则：状态 `passed`，候选接受。

算法选择第一个通过 post-fast 校验的时间候选。被拒绝的候选不会终止搜索，会继续检查
后续时间候选。

### 8.6 fallback 和最终标签

若没有时间候选通过，并且存在评价截点，则对评价截点本身执行相同 post-fast 校验：

- 通过：评价截点成为锚点，`anchorSource=absolute_evaluation_cutoff`。
- 拒绝：没有锚点。

最终：

- 有锚点：`algorithmLabel=offbook`，`offBookPly` 为锚点手。
- 无锚点：`algorithmLabel=no_offbook`，`offBookPly=null`。

所有候选、拒绝原因、动态阈值、评价截点和 post-fast 明细都会写入
`algorithmEvidence`，不允许人工补锚点。

### 8.7 双方独立运行

Reference 对同一局黑白双方分别调用一次完全相同的算法。双方不共享锚点，因此同一局可以
一方为 `offbook`、另一方为 `no_offbook`。

## 9. Reference 派生算法

### 9.1 输入完整性

`build-reference` 要求目标派生目录为空，并要求冻结 Reference 至少包含：

- `selected_games_with_partitions.json`
- `selected_account_bundle.json`
- `engine_game_index.json`
- `reference_completion_audit.json`
- `partition_engine_index_audit.json`
- `engine_level22/audit.json`
- `final_sha256_manifest.json`

构建前记录这些输入的 SHA-256，构建后再次计算；任何源文件变化都会使审计失败。

### 9.2 每局两条 directed records

对每个源游戏：

1. 核对选择清单、bundle、engine index 和 Level22 JSON 的 `gameId` 一致。
2. 核对引擎 JSON SHA-256。
3. 以黑方为目标独立运行脱谱算法并生成一条记录。
4. 以白方为目标独立运行脱谱算法并生成一条记录。
5. 验证最终 directed record 数恰好是源局数的两倍。

### 9.3 正式分母

一条 directed record 只有同时满足以下条件才是 `formalReferenceEligible=true`：

1. 源局属于主矩阵 `inMainMatrix=true`。
2. 目标方 `oldR` 落在 `[1600,2495]`。
3. 对手方 `oldR` 落在 `[1600,2495]`。

低 Elo 扩展局仍生成双方脱谱、GE4、GE10 和 WLD 记录，用于审计和研究，但不进入正式
Reference index 和统计分母。

### 9.4 directed record 的 scope 指标

对目标方节点：

- `offbook`：保留 `ply >= offBookPly`，锚点手包含在内。
- `no_offbook`：保留全部目标方着手。
- pass 不作为目标 loss 节点。
- `lossClipped=null` 的节点不进入 loss 分母。

记录以下主/诊断指标：

```text
loss_ge4_count
loss_ge4_rate
loss_ge10_count
loss_ge10_rate
game_equal_mean_disc_loss
engine_wld_loss_total_from_ply39
```

注意：WLD total 使用该目标方整局中 `global_placement_ply >= 39` 的节点，不按脱谱 scope
截断。它是独立次要指标。

### 9.5 派生产物

Reference 派生目录包含：

- `directed_target_records.jsonl/csv`
- `offbook_records_by_target_side.json`
- `reference_cell_summary.json/csv`
- `reference_source_manifest.json`
- `reference_build_audit.json`
- `reference_sha256_manifest.json`

正式运行会逐项核对 SHA-256 manifest，并要求 build audit 的 `ok=true`。

## 10. 调查目标记录构造

调查运行从所选 bundle、Level22 目录和 `offbook_records.json` 构造每局一条目标方记录。
程序要求：

- 每个 Level22 `gameId` 同时存在 bundle detail 和脱谱记录。
- 目标账号每局恰好匹配一方。
- 每个 offbook anchor 必须确实是目标方的一个着手 ply。
- `no_offbook` 必须具有空锚点。
- bundle 中的所选局必须全部具有 Level22 输出，不允许静默缺局。

调查记录沿用第 9.4 节的 scope 和指标定义，并额外保留 `created` 供候选时间分布诊断使用。

## 11. 二维 Reference 匹配

### 11.1 匹配维度

GE4 主指标的 cell key 为：

```text
(targetEloBand, opponentEloBand, targetColor, analysisScope)
```

只有 `formalReferenceEligible=true` 且 `loss_ge4_rate` 非空的记录进入 index。

### 11.2 轴向线性插值

对目标方和对手方的精确 `oldR` 分别计算轴向权重。若 rating 位于相邻分箱中心
`c_left <= r <= c_right`：

```text
w_right = (r - c_left) / (c_right - c_left)
w_left  = 1 - w_right
```

二维角点的原始权重为：

```text
w_corner = w_target × w_opponent
```

低于第一个中心 1650 但仍在边界内时，全部权重落到第一档；高于最后中心 2447.5 但不高于
2495 时，全部权重落到最后一档。

### 11.3 超界处理

rating 超出正式范围时只做边界夹取，不外推：

- 目标低于 1600：夹到 1600，标记 `target_below_reference`。
- 目标高于 2495：夹到 2495，标记 `target_above_reference`。
- 对手超界：夹到相应边界，当前字段标记 `nearest_opponent_band`。

超界标志必须保留在逐局输出中。

### 11.4 缺 cell fallback

每个二维预期角点独立解析。若预期 cell 在全局排除后没有记录：

1. 固定目标 Elo 档、颜色和 scope，在对手轴选择中心距离最近的可用 cell。
2. 若仍无记录，固定颜色和 scope，在二维中心平面选择欧氏距离平方最近的可用 cell。
3. 二维平局依次按对手中心距离、目标档下界、对手档下界破平。
4. 若同颜色和同 scope 完全没有记录，则该局 `not_calibratable`。

fallback 永不跨颜色，也永不跨 GE4 scope。多个预期角点若回落到同一实际 cell，其权重会
合并，然后整体重新归一化。

### 11.5 cell 内局等权

设解析后的 cell 权重为 `W_c`，cell 中有 `n_c` 条可用记录。cell 内每条记录权重为：

```text
w_record = W_c / n_c
```

最终 Reference 期望为：

```text
E_ref = Σ_record w_record × metric_record
```

因此先进行 cell 间 Elo 插值，再在每个 cell 内对单局记录等权。不会按节点数给某局更大权重。

### 11.6 调查与 Reference 重叠排除

在任何正式评分之前，程序收集全部调查 `gameId`。若某个 ID 存在于 Reference，则从
Reference 全局删除该 ID 的所有 directed records，也就是同时删除黑、白两个方向。

该排除应用于：

- cell 记录与 Reference 期望；
- 经验位置和预测区间；
- 后续伪玩家抽样池；
- WLD Reference。

排除 ID 和记录数量写入逐局评分 payload。

## 12. 每局评分

### 12.1 主残差

对每个可校准调查局：

```text
actual_i   = loss_ge4_rate_i
expected_i = 二维匹配后的 Reference 期望
r_i        = expected_i - actual_i
```

`r_i` 即 `externalStrengthResidual`，是后续全部 GE4 选择的基础分数。

### 12.2 Reference 经验位置

对匹配记录及其权重，输出：

- `weightedCdfLessOrEqual = Σ w × I(value <= actual)`
- `weightedLowerTailStrict = Σ w × I(value < actual)`
- `weightedUpperTailInclusive = Σ w × I(value >= actual)`

95% 匹配 Reference 区间使用离散加权分位数：按值排序，累计权重首次达到 0.025 或 0.975
时取该记录值，不做分位数间线性插值。

### 12.3 不可校准规则

出现以下任一情况，该调查局不进入主扫描：

- 目标 scope 没有有效 `lossClipped` 节点。
- 同颜色、同 scope 没有任何可用 Reference。
- 排除重叠局后没有有效指标记录。
- 匹配池只有一个 distinct `gameId`，无法执行强制 leave-one 伪玩家校准。

最后一种情况的明确原因是：

```text
insufficient_distinct_reference_games_for_required_leave_one_pseudo_calibration
```

不可校准局仍保留在评分输出和模型对照候选中，但不参与主统计扫描。

## 13. 候选扫描空间

### 13.1 稳定强度排序

将所有可校准调查局按以下键排序：

```text
(-externalStrengthResidual, gameId)
```

即残差从强到弱排列，残差相同按 `gameId` 升序稳定破平。时间、session、比赛轮次和人工
观察均不参与选择。

### 13.2 只扫描前缀

Sentinel 不枚举任意对局组合。主扫描只检查：

- 最强单局：独立的 isolated test。
- 全部可校准局：独立的 uniform test。
- 排序后的前缀 `k=2...K_max`：concentrated test。

其中：

```text
K_max = min(floor(n / 2), n - 3)
```

`n` 为可校准调查局数。`k=1` 和 `k=n` 已由单局与全局专门统计量处理，不进入前缀 k
集合。`n` 太小时，前缀集合可以为空。

### 13.3 前缀效应

设排序后的残差为 `r_(1)...r_(n)`。对每个候选 k：

```text
externalEffect(k) = mean(r_(1)...r_(k))

internalEffect(k)
  = mean(r_(1)...r_(k)) - mean(r_(k+1)...r_(n))
```

`externalEffect` 衡量候选相对于外部 Reference 的强度；`internalEffect` 衡量候选相对于该
账号剩余调查局的强度。

另外：

```text
bestSingleScore = max_i r_i
allGamesEffect  = mean_i r_i
```

## 14. 伪玩家生成算法

### 14.1 目的

直接在多个 k 中挑最显著者会产生选择偏差。Sentinel 通过让每个伪玩家执行与真实玩家完全
相同的扫描，构造“正常情况下扫描后最优结果”的经验分布。

### 14.2 slot-preserving 抽样

每个真实可校准调查局定义一个 slot，保留该 slot 的：

- 精确目标 `oldR`
- 精确对手 `oldR`
- 目标颜色
- analysis scope

先按第 11 节规则得到该 slot 的加权 Reference 单局池。每个伪玩家对每个 slot 从完整
directed-game record 池按权重有放回抽取一条记录。不同 slot 可以抽到同一 Reference 局，
同一 slot 在不同 replicate 中也可重复抽到同一局。

默认生成 10,000 名伪玩家，随机数实现为 Python `random.Random` 的 MT19937；seed 冻结在
运行配置和输出 summary 中。

### 14.3 sampled-game leave-one

若某 slot 抽到 Reference 游戏 `g`，不能用包含 `g` 自身的 Reference 期望来评价 `g`。
程序会针对该 slot 重新匹配，并从所有 cell 全局排除 `gameId=g` 的两个方向：

```text
pseudoResidual(slot, g)
  = E_ref(metric | slot, exclude gameId g globally) - metric(g)
```

这不是只删除抽中的 directed row，而是删除同一原始游戏的黑白两条记录。结果按
`(slotIndex, gameId, metric)` 缓存，但缓存不改变计算定义。

### 14.4 伪玩家完整扫描

每名伪玩家获得与真实玩家 slot 数相同的一组残差，然后执行完全相同的：

1. 残差稳定降序排列；
2. 最强单局；
3. 全局均值；
4. 全部合法前缀 k 的 external/internal effect。

因此校正分布包含了“先扫描、再选择”的机会，而不是固定 k 的简单 null 分布。

## 15. 经验超越率与 Wilson 区间

### 15.1 固定 k 的单侧上尾

对真实统计量 `x` 和 R 个伪玩家统计量 `X_b`：

```text
p_upper(x) = (1 + #{b: X_b >= x}) / (R + 1)
```

`+1` 修正确保有限模拟中 p 不为 0。external 和 internal 分别计算：

```text
p_ext(k) = p_upper(externalEffect_real(k))
p_int(k) = p_upper(internalEffect_real(k))
p_joint(k) = max(p_ext(k), p_int(k))
```

取最大值意味着集中型候选必须同时具备外部和内部证据；较弱的一项决定联合结果。

### 15.2 真实候选 k 的选择

真实玩家从全部 k 中最小化 `p_joint(k)`。平局依次选择：

1. `externalEffect` 更大；
2. `internalEffect` 更大；
3. k 更小；
4. `gameIds` 元组字典序更小。

### 15.3 每名伪玩家的自扫描排名

为校正选 k，程序也为每名伪玩家在每个 k 上计算 leave-one 经验上尾排名。对当前伪玩家的
值，从 R 个值组成的已排序分布中等价地去掉自身，再使用 plus-one；实现形式为：

```text
(包含自身在内、>= 当前值的数量) / R
```

每名伪玩家随后按与真实玩家相同的 tie-break 选择自己的最佳 `p_joint`，形成
`pseudo_best_joint` 分布。

### 15.4 扫描校正

较小的最佳 p 更极端，所以真实玩家的扫描校正使用下尾：

```text
p_scan
  = (1 + #{b: pseudo_best_joint_b <= real_selected_joint}) / (R + 1)
```

程序另行对 internal-only 候选执行同类完整扫描校正。

最强单局则将真实最大残差与每名伪玩家的最大残差比较；全局异常将真实全局均值与每名伪玩家
的全局均值比较。两者均使用单侧上尾 plus-one 公式。

### 15.5 Wilson 95% 区间

经验超越计数使用 plus-one 后的：

```text
successes = extreme_count + 1
trials    = R + 1
p_hat     = successes / trials
```

Wilson 区间使用 `z=1.959963984540054`：

```text
denom  = 1 + z²/n
center = (p_hat + z²/(2n)) / denom
half   = z × sqrt(p_hat(1-p_hat)/n + z²/(4n²)) / denom
CI     = [max(0, center-half), min(1, center+half)]
```

分类门槛使用 Wilson 上界，而不是只看点估计 p。

## 16. 固定候选整局聚类 bootstrap

### 16.1 与选择校正的职责分离

伪玩家完整扫描负责校正“扫描多个 k 后再选择”的偏差。候选确定后，整局 bootstrap 只描述
这个固定候选的效应不确定性，不能代替扫描校正。

### 16.2 抽样单位

设固定候选组为 C，剩余可校准局为 R：

- 从 C 中有放回抽取 `|C|` 个整局。
- 从 R 中有放回抽取 `|R|` 个整局。
- 不在局内独立抽节点。

每次 replicate 计算：

```text
external* = mean(candidate sampled residuals)
internal* = mean(candidate sampled residuals) - mean(remainder sampled residuals)
```

默认 10,000 次，使用 Sentinel seed `+1`。95% 区间取 bootstrap 分布的 2.5% 和 97.5%
分位数。

### 16.3 集中型的正方向要求

集中型异常不仅要求两个点估计为正，还要求：

```text
externalEffect95CI.lower > 0
internalEffect95CI.lower > 0
```

即两个固定候选区间都完全位于正方向。

## 17. WLD 次要指标

### 17.1 节点 WLD loss

引擎分数先压缩为三档 rank：

```text
score > 0 -> win rank 2
score = 0 -> draw rank 1
score < 0 -> loss rank 0
```

实际着法后的分数优先使用 `actualEval`；缺失时根据 `next_best_score` 和
`same_side_after_move` 调整符号。节点 WLD loss 为：

```text
drop = max(0, rank(best-before-move) - rank(actual-move))
wld_loss = drop / 2
```

可能值为 0、0.5 或 1。

### 17.2 固定起点

每局 WLD total 只累计目标方满足以下条件的节点：

```text
global_placement_ply >= 39
```

`global_placement_ply` 是忽略 pass 的全局实际落子序号，合法范围为 1 到 60，39 包含在内。

### 17.3 WLD Reference 匹配

WLD 使用相同双方 Elo 和目标颜色，但 `scope_aware=false`，即不区分
`post_offbook_inclusive` 与 `full_game_fallback_no_offbook`。WLD 仍不跨颜色。

```text
externalWldStrengthResidual
  = expectedReferenceWldTotalFromPly39 - actualWldTotalFromPly39
```

正值同样表示实际 WLD loss 更低。

### 17.4 不参与选组

WLD 永远在 GE4 候选冻结之后计算，不得：

- 改变排序；
- 选择另一个 k；
- 新增或删除 `reportedGameIds`；
- 把原本没有正式举报组的分类强制变为举报组。

对于集中型，WLD supportive 需要 external 和 internal 区间下界均为正；其他需要 WLD
支持的外部分类只要求 external 区间下界为正。否则标记
`secondary_metric_not_supportive`。对非外部异常分类只作 diagnostic。

## 18. 分类决策树

分类按固定优先级执行。

### 18.1 `concentrated_external_internal_anomaly`

必须同时满足：

```text
存在合法 selected k
p_scan <= 0.05
Wilson95(p_scan).upper <= 0.05
selected externalEffect > 0
selected internalEffect > 0
fixed bootstrap external CI lower > 0
fixed bootstrap internal CI lower > 0
```

输出：

```text
reportedGameIds = selected prefix game IDs
```

### 18.2 `external_uniform_anomaly`

若未满足集中型，但：

```text
allGamesEffect > 0
Wilson95(all-games upper-tail rate).upper <= 0.05
```

则分类为全局外部异常。由于全部局共同偏强，系统没有可辩护的内部非举报对照组，因此：

```text
reportedGameIds = []
```

### 18.3 `isolated_external_anomaly`

若前两类未满足，但：

```text
bestSingleScore > 0
Wilson95(best-single scan-corrected rate).upper <= 0.05
```

则：

```text
reportedGameIds = [bestSingleGameId]
```

最强单局与每名伪玩家自己的最大单局比较，因此已校正“最多 n 局中挑一局”的机会。

### 18.4 `internal_variation_only`

若未满足集中型，且 internal-only 完整扫描结果满足：

```text
selected internalEffect > 0
Wilson95(internal-only scan rate).upper <= 0.05
```

则表示账号内部存在变化，但外部 Reference 证据不足：

```text
reportedGameIds = []
```

### 18.5 `no_clear_signal`

其他情况，包括没有任何可校准调查局，均为 `no_clear_signal`，不产生举报组。

### 18.6 分类优先级

程序优先级是：

```text
concentrated
  > external_uniform
  > isolated_external
  > internal_variation_only
  > no_clear_signal
```

因此一个同时满足全局和单局条件、但不满足集中型条件的结果会先归为 uniform，且不会产生
人为内部对照组。

## 19. 举报组、统计对照和模型对照

### 19.1 statistical controls

只有存在非空举报组时，`statisticalControlGameIds` 才非空。它由所有可校准调查局中排除
举报局得到。

### 19.2 model controls

只有存在非空举报组时，`modelControlGameIds` 才非空。它由全部调查局中排除举报局得到，
可以包含未进入主扫描的不可校准局。

### 19.3 模型复核门槛

```text
modelReviewReady
  = reportedGameIds 非空 and len(modelControlGameIds) >= 8
```

如果有举报局但模型对照少于 8 局，不运行个人模型；非模型复核仍运行。若举报组为空，个人
模型和非模型举报组比较均不运行，不允许人工补组来进入后续流程。

## 20. 候选时间分布诊断

候选冻结后，程序按 `(created, gameId)` 升序查看候选在时间上的分布，并计算连续候选 run
数量。输出 pattern：

- `isolated`：候选不超过 1 局。
- `continuous`：所有候选构成一个连续 run。
- `alternating_or_interleaved`：候选不少于 2 局，且没有相邻候选。
- `discrete_clusters`：其他多簇情况。

字段 `selectionUsedTimeOrSession=false` 是合同：时间分布只用于解释，不能反向参与选组。

## 21. 冻结与可复现性

### 21.1 selection manifest

`selection_manifest.json` 冻结：

- 全部调查 game IDs；
- 每局是否可校准及外部残差；
- 所有 tested k 和 per-k 结果；
- selected k；
- 举报、统计对照和模型对照 IDs；
- classification 和 `modelReviewReady`；
- 全局排除的 Reference game IDs；
- Reference config 和 manifest 的路径及 SHA-256；
- 调查 bundle、Level22 audit、脱谱记录 SHA-256；
- seed、伪玩家次数、bootstrap 次数；
- 选择政策和冻结政策。

### 21.2 规范化 payload hash

冻结 payload 使用 UTF-8 JSON、键排序、无额外空格的规范化形式计算 SHA-256：

```text
payloadSha256 = SHA256(canonical JSON payload)
```

hash 字段在计算时尚未加入 payload 自身，避免自引用。

### 21.3 冻结政策

```text
model results cannot modify this selection manifest
```

后续个人模型、WLD、用时或人工解释只能作为支持、不支持、冲突或限制说明，不能修改
`reportedGameIds`。

## 22. 条件式个人模型复核

本节描述 Sentinel 在满足第 19.3 节门槛后调用的后置复核。它不参与主选组。

### 22.1 安全 Hint 重算

先从所选 bundle 生成包含 pass 语义的原始节点数据，再对全部需要节点执行两个固定阶段：

| 阶段 | Level | 每 Console 线程 | Book | 返回数 | 默认 worker | batch | timeout |
|---|---:|---:|---|---:|---:|---:|---:|
| hint1 | 2 | 1 | 关闭 | 1 | 12 | 64 | 60 秒 |
| hint6 | 18 | 16 | 开启 | 6 | 12 | 128 | 900 秒 |

两阶段 hash 都固定为 25，单 batch 最多尝试 2 次。安全 runner 冻结引擎资源哈希、请求 ID、
棋盘、`setboard` 回显、hint 回显、worker/batch ID，并支持已提交事务的恢复。装配阶段核对
行数、实际落子数、pass 数和游戏数。

### 22.2 数据物化与 split

模型数据物化使用：

- 固定 base checkpoint 与 12 成员 ensemble manifest；
- 固定 preprocessing；
- Human Frequency Book；
- 安全重算后的 hint1/hint6；
- OQ Profile 上下文；
- Sentinel 冻结的 reported/control 分组。

有脱谱锚点的举报局只评价 `global_placement_ply >= offBookPly`，包含锚点。没有锚点的举报局
使用目标玩家整局全部着手，不创建合成锚点。模型控制局是 train split，举报局是 test
split；个人 adapter 不使用举报局标签训练。

当前 Profile policy 是 retrospective current profile，并显式允许 temporal leakage。它可能
包含对局之后累积的信息，因此报告必须保留该限制。

### 22.3 12 成员个人残差 adapter

每个基础 ensemble 成员的主网络被冻结，只训练从 64 维 hidden state 到输出 logits 的
残差：

```text
personal_logits = base_logits + hidden @ deltaW + deltaB
```

severity adapter 形状为 `64×4 + 4`，类别是：

```text
zero, 1-3, 4-9, >=10
```

WLD adapter 形状为 `64×3 + 3`。两者零初始化，因此优化前严格等于 base 模型。

对每个控制游戏先取节点损失均值，再对游戏等权。目标函数为：

```text
L = game_equal_cross_entropy
  + 0.25 × game_equal_KL(base || personal)
  + 0.01 × ||deltaW||²
  + 0.01 × ||deltaB||²
```

优化器固定为 deterministic LBFGS、strong-Wolfe line search、`max_iter=200`、
`tolerance_grad=1e-7`、`tolerance_change=1e-9`。12 个成员各自生成 adapter，基础网络不更新。

### 22.4 举报局推理与 bootstrap

每个成员输出 severity 类概率；主复核量为：

```text
actual_loss_ge4 - personal_ensemble_probability_loss_ge4
```

先在每局内对节点取均值，再对举报局等权。默认 bootstrap 10,000 次，每次：

1. 从 12 个个人成员中有放回抽取 12 个并求成员均值。
2. 从举报局中有放回抽取与原组相同数量的整局。
3. GE4、zero、GE10 共用同一次成员抽样和整局抽样。
4. 完整控制集固定，不在举报局 bootstrap 中重抽。

输出 2.5% 和 97.5% 区间。它量化有限 ensemble 与举报局抽样的不确定性，不是作弊概率。

### 22.5 与主选择的关系标签

最终报告只用 combined GE4 的 `actual-minus-expected` 95% 区间给个人模型复核贴标签：

- `upper < 0`：`supportive`，实际 GE4 率低于个人模型期望。
- `lower > 0`：`conflicting`，实际 GE4 率高于个人模型期望。
- 其他：`not_supportive`。

无论标签是什么，都不能改变冻结的主 Sentinel 选择。

## 23. 条件式非模型复核

只要存在正式 `reportedGameIds`，非模型复核就运行，即使个人模型因对照少于 8 局而跳过。
它使用 Level22 中目标玩家的整局节点；这些后置比较不重新选择举报组。

### 23.1 逐局和聚合子损

逐局记录：

- 有效着手数；
- total / mean / median / maximum disc loss；
- zero-loss rate；
- positive-loss mean；
- GE4 和 GE10 count/rate；
- 从实际落子 39 起的 WLD total。

聚合同时提供：

- 局等权平均子损：先算每局 mean，再对局取平均。
- 着法等权平均子损：合并所有有效节点后取平均。

### 23.2 举报组与对照组比较

提供两套 universe：

- `sameColorComparison`：只使用与举报局颜色相同的调查局。
- `allGamesComparison`：使用全部调查局。

比较量包括局等权/着法等权平均子损差、zero-loss rate 差、GE4/GE10 rate 差和平均每局 WLD
差。差值方向是：

```text
reported - control
```

负的子损或 GE4 差表示举报组损失更低。

### 23.3 整局聚类 bootstrap

举报组和对照组内分别有放回抽取与原组相同数量的整局，再计算差值。GE4 与 GE10 共用同一
轮整局抽样。均值、zero rate 和 WLD 也以整局为聚类单位生成 95% 区间。

### 23.4 精确组合位置

在相应 universe 中，从所有与举报组局数相同的组合计算局均子损的经验位置：

- 组合总数不超过 1,000,000：完整枚举，给出 exact lower/upper/two-tail 位置。
- 超过上限：从组合字典序 rank 中无放回均匀抽取 1,000,000 个 rank，给出 Monte Carlo
  lower/upper/two-tail 位置，seed 冻结。

### 23.5 单步两部分模型

该模型用于补充描述零子损概率与正子损幅度：

1. Logistic regression 预测 `loss == 0`。
2. 对 `loss > 0` 的节点，用 OLS 拟合 `log(loss)`，再用 smearing correction 返回原尺度。

自变量为：

```text
reported indicator
(ply - 30) / 20
white indicator
tournament indicator
```

输出 zero-loss odds ratio、positive-loss mean ratio、举报状态下预测均值、反事实非举报状态
预测均值及差值。区间按整局分别重抽举报/对照组后重新拟合；拟合失败的 replicate 不进入区间。

### 23.6 用时分析

用与举报局同色的非举报局拟合控制基线。令 `x=ply/maxPly`，基函数为：

```text
1, x, x², x³,
max(0, x-0.2)³,
max(0, x-0.4)³,
max(0, x-0.6)³,
max(0, x-0.8)³
```

用 ridge `alpha=0.1` 解线性系统，截距不惩罚，预测值下限为 0。同 ply 控制分布的第 90
百分位作为 long-think 阈值；缺少完全相同 ply 时使用最近 ply 的阈值。

逐举报局计算：

- total / mean time；
- 相对基线平均残差和平均绝对残差；
- 实际用时与基线曲线的 Pearson/Spearman 相关；
- long-think count/rate；
- 超出 P90 的总时长和最大时长。

举报组指标与全部同局数控制组合比较，按预定义单侧方向使用 plus-one p。

## 24. 最终报告合同

最终 Sentinel 报告 schema 为：

```text
player-anomaly-sentinel-report-v1
```

主功能字段至少包括：

- `classification`
- `selection`
- `perGameReferenceScores`
- `sentinelScan`
- `pseudoScanSummary`
- 冻结举报/对照组
- 条件式个人模型结果或未运行原因
- 条件式非模型子损/WLD/用时结果
- 参数、主要 artifact 哈希和限制说明

独立的 estimated-Elo 字段由另一份技术规格定义，不在本文重复。

报告必须保留以下解释限制：

- 统计区间和经验 p 不是作弊概率。
- WLD 不能创建或替换 GE4 冻结举报组。
- 个人模型只能支持、不支持或冲突，不能修改选择。
- 当前累计 Profile 可能包含调查对局之后的信息。
- `no_offbook` 举报局在个人模型中使用整局节点，不会伪造锚点。

## 25. 断点恢复与产物完整性

### 25.1 运行目录

新运行目录必须为空。配置和进度分别写入：

- `run_config.json`
- `progress.json`

### 25.2 阶段命令哈希

每个阶段记录完整命令的 SHA-256。已标记 completed 的阶段只有同时满足以下条件才跳过：

1. 当前命令哈希与已记录哈希完全一致。
2. 所有声明输出仍存在。
3. 每个输出文件或目录树的 SHA-256 与已记录值一致。

命令变化、输出缺失或输出 hash 变化都会失败，不会静默重用。长阶段自身还维护嵌套
`progress.json` 并使用 `--resume`。

### 25.3 UTF-8

子进程显式设置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。JSON/JSONL/CSV 文本写入使用
UTF-8，JSON 保留非 ASCII 字符。

## 26. 主要输出索引

| 文件 | 作用 |
|---|---|
| `account_bundle.json` | 原始账号棋谱 bundle |
| `selected_account_bundle.json` | 过滤后最近最多 30 局 |
| `game_catalog.json` | 每局时间、颜色、双方 oldR |
| `engine_level22/` | Level22 逐局结果、audit、progress |
| `offbook_records.json` | 目标方确定性脱谱记录及证据 |
| `per_game_reference_scores.json/csv` | 每局匹配、期望、残差和可校准状态 |
| `pseudo_scan_replicates.csv` | 每名伪玩家的最佳扫描及 WLD 诊断 |
| `pseudo_scan_summary.json` | seed、抽样政策和 Reference pool hash |
| `sentinel_scan_results.json` | per-k 结果、分类、举报/对照 IDs |
| `selection_manifest.json` | 冻结选择及 payload SHA-256 |
| `model_review_groups.json` | 后置模型消费的冻结组摘要 |
| `hints/` | 条件式安全 hint 产物 |
| `model/` | 条件式个人 adapters、推理和 bootstrap |
| `non_model/loss-analysis.json` | 条件式子损、WLD 和两部分模型 |
| `non_model/time-analysis.json` | 条件式用时基线及组合比较 |
| `report.json` | 最终 Sentinel 报告 |

## 27. 明确禁止的行为

当前 Sentinel V1 明确不允许：

- 用时间或 session 参与候选选组。
- 枚举任意对局组合来寻找最显著子集。
- 在看到调查结果后调整 Reference 分箱、阈值、seed 或 k 范围。
- 在 Reference 重叠时只删除单方向记录。
- 跨颜色或跨 GE4 scope 借用 Reference。
- 用 WLD 重新选组。
- 用个人模型修改冻结 `reportedGameIds`。
- 对 `no_offbook` 局人工或模型生成一个虚构锚点。
- 把经验超越率、bootstrap 区间或模型概率解释成作弊概率。

## 28. 测试合同

当前核心回归测试覆盖：

- 零坐标落子局在最近 30 局限制前剔除；
- 同一 Reference 局双方独立运行算法；
- `offbook` 包含锚点、`no_offbook` 使用整局；
- pass 不形成 loss 节点；
- WLD 包含实际落子 ply 39；
- 调查重叠全局排除 Reference 双方向；
- 双方 Elo、颜色和 scope 匹配；
- 缺 cell fallback 和超界标记；
- 低 Elo 扩展不进入正式分母；
- leave-one 池不足时显式不可校准；
- 只扫描稳定强度前缀；
- 伪玩家完整扫描校正选择偏差；
- isolated 使用每名伪玩家自己的最大单局；
- plus-one 和 Wilson 边界；
- 固定 seed 精确复现；
- 模型不能进入或修改 selection manifest；
- 中文路径和 UTF-8 输出；
- completed 且 hash 匹配的阶段可恢复跳过；
- 脱谱阈值严格性和 post-fast 接受/拒绝规则。

主要测试文件：

- `tests/test_sentinel_v1.py`
- `tests/test_detect_offbook.py`
- `tests/test_sentinel_unified_analysis.py`
- `tests/test_player_investigation_orchestrator.py`
- `tests/test_offbook_phase_bootstrap.py`

## 29. 维护要求

以下任一变化都应视为 Sentinel 主功能版本或 Reference 合同变化，并同步更新本文档、测试、
配置版本和审计产物：

- 最近对局选择规则；
- Level22 参数或 WLD 起点；
- 脱谱阈值、post-fast 规则或 scope；
- GE4 主指标定义；
- Elo 边界、分箱中心、插值或 fallback；
- Reference 来源、正式分母或 overlap 排除；
- 候选 k 范围、排序、效应定义或 tie-break；
- 伪玩家抽样、leave-one、plus-one、Wilson 或扫描校正；
- bootstrap 单位或分类门槛；
- 举报组和模型对照门槛；
- selection manifest 字段或规范化 hash；
- 后置模型对主选择的关系政策。

新 Reference 必须全量生成新的派生目录，核对 build audit 的 `ok=true`，重新生成 SHA-256
manifest，再通过版本化配置切换。不得在原正式派生目录上原地覆盖。

## 30. 独立 estimated-Elo v2 接入（2026-08-28）

主 Sentinel 的异常扫描、举报组冻结和模型复核仍保持 V1；estimated-Elo v2 是统一单玩家
入口中的独立并行产物，不得反向改变 `selection_manifest.json` 或 `reportedGameIds`。

v2 核心只位于 `src/player_analysis_toolkit/sentinel_elo.py`。维护 CLI 负责 reference 派生、
条件缓存、calibration、覆盖率审计和调试估算；`sentinel_unified_analysis.py` 调用同一核心；
`run_player_investigation.py` 只编排并把 v2 `estimated_elo.json` 及 artifact 信息装入最终
`report.json`。

v2 不再使用 candidateZ 过零作为主评分。它按阶段拟合加权 Beta-Binomial，对每盘累加四个
`-log(PExact)`，再对目标棋做算术平均得到 `J(E)`。z1、z2、z3 仅依次作为下一阶段距离
坐标；z4 仅诊断。完整统计合同见 estimated-Elo 实施规格第 25 节。

统一入口必须同时验证：

- config schema 和算法版本；
- 原始 directed reference manifest；
- conditional reference manifest、records SHA 和 config SHA；
- v2 calibration schema、conditional manifest SHA 和 calibration SHA manifest。

任一版本或 SHA 不一致时直接失败。v1 calibration 不得用于 v2。正式 calibration 与独立
validation 完成前，v2 只能输出 `calibration_unavailable`，仓库主配置继续指向 v1。

## 31. 当前正式 estimated-Elo v4（2026-09-11）

主 Sentinel V1 的选组与冻结合同不变；统一单玩家分析中的 estimated-Elo 正式热路径已由
独立 v4 实现提供。v4 使用 Anscombe 方差稳定变换、局部 KNN 加权预测分布和相邻阶段 z
条件距离，完整数学与产物合同见
[`SENTINEL_ESTIMATED_ELO_V4_IMPLEMENTATION.md`](SENTINEL_ESTIMATED_ELO_V4_IMPLEMENTATION.md)。

v4 不读取 v1/v2/v3 calibration，不保留旧统计模型的正式失败字段或参数。当前正式配置为
`sentinel_elo_reference_config.json`（v4-matchup600-20260911；候选文件为
`sentinel_elo_reference_config_v4_matchup600_20260911.json`），其独立 config、
reference-z cache、calibration artifact 和 schema 通过各自 SHA-256 合同绑定；
`sentinel_unified_analysis.py` 是单玩家入口，`run_player_investigation.py` 只负责生命周期
编排。已有 Level22 的阶段 `x/n` 被复用，reference z 变化只生成新的 v4 缓存，不重新执行
Level22。
