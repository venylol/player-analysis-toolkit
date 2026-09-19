# 哨兵模式行棋质量等价 Elo 实施规格

状态：v1 历史规格；estimated-Elo v2 实施中，尚未完成正式 calibration  
规格版本：v1 + v2 补充合同（第 25 节）  
参考数据快照：2026-08-15 Level22、Elo 1600–2495  
字符编码：UTF-8

## 1. 文档目的

本文是后续实现 Agent 的交接规格。目标是在现有哨兵模式旁新增一个不依赖机器学习、可解释、可复现、会随参考数据库增大而改善的 Elo 预估功能。

该功能输出的是：

```text
目标用户最近一组有效对局的“行棋质量等价 Elo”
```

它不是账号当前显示 Elo、历史最高 Elo、作弊概率，也不是未来胜率预测。目标用户自己的显示 Elo 不得进入估计算法，只能在最终报告中作为外部对照。

本规格只新增功能，不得改变现有 sentinel 的异常评分、固定 Reference、scan、freeze 或旧输出语义。

## 2. 已冻结的用户决策

以下内容不得在实现时自行改写：

1. 不使用 TCN 或任何机器学习模型估计 Elo。
2. 四阶段固定，不自动学习或随数据库变化重新选边界。
3. 四阶段为：
   - 阶段 1：global placement ply 1–30；
   - 阶段 2：global placement ply 31–47；
   - 阶段 3：global placement ply 48–53；
   - 阶段 4：global placement ply 54–60。
4. global placement ply 只统计实际坐标落子；pass 不占用新的 ply。
5. 有脱谱锚点的目标局从锚点开始分析，包含锚点。
6. 无脱谱锚点的目标局纳入该局目标方的全部有效节点。
7. 阶段内部计算 GE4 比例；阶段之间严格等权，不按阶段长度或节点数加权。
8. 先在一局内部把四阶段等权合成一个整局指标，再对整局指标做数据库标准化。
9. 多局之间局等权；不先估计每局 Elo 再平均。
10. 不使用四阶段平方均方根作为最终主评分。
11. 不使用阶段相关性矩阵补偿。
12. 不对目标用户做普通 bootstrap 来决定权重或区间。
13. 正式估计至少需要 10 局，最多使用最近 30 局有效对局。
14. 第一版不考虑 session 划分。
15. 某局缺少任一阶段的有效数据时，整局排除；不得用剩余阶段重新归一化。
16. 每阶段第一版至少需要 1 个有效目标方子损节点。
17. 95% 数据库校准第一版不按对局数或 Elo 分组，使用一个全局阈值。
18. 隐藏答案测试的已知 Elo 使用数据库中该用户时间最新一局的赛后 `newR`，不使用当前 Player 字段，也不使用各局 `oldR` 平均值。
19. 参考邻居数使用 `K = ceil(N^(2/3))`。
20. Elo 搜索使用整数网格 1600–2495。

## 3. 术语

```text
目标用户
= 要估计 Elo 的账号

目标局
= 目标用户进入本次估计的一局棋

参考 directed game
= 参考库中从某一方选手视角形成的一条整局记录

试探 Elo E
= 算法正在检验的目标用户假设 Elo

目标局对手 Elo O_i
= 目标局开始前对手的 oldR

参考目标方 Elo RT_j
= 参考 directed game 目标方开始前的 oldR

参考对手 Elo RO_j
= 参考 directed game 对手开始前的 oldR
```

参考匹配必须使用对局开始前的 `oldR`，因为它描述该局发生时的对阵条件。`newR` 只用于已知用户隐藏答案校准。

## 4. 当前数据事实

冻结的源参考目录：

```text
research/offbook_detection/data/
  oq_elo_matchup400_reference_level22_1600plus_20260815
```

当前哨兵派生目录：

```text
research/offbook_detection/data/
  oq_sentinel_reference_level22_1600plus_v6_20260819
```

已经核对：

```text
源参考对局                         10,244 局
源参考玩家侧                       20,488 条
源参考唯一用户                      3,610 名
现有正式 sentinel directed records 20,256 条
现有正式唯一目标用户                 3,519 名
至少 10 局正式记录的用户               539 名
缺失玩家 ID                              0
缺失 newR 玩家侧                         0
重复 gameId                              0
```

`selected_account_bundle.json` 的 `details[].players[]` 同时保存 `oldR` 与 `newR`。现有 `directed_target_records.jsonl` 没有保存 `newR`，新派生文件必须补入。

每个 Level22 节点已经保存至少以下字段：

```text
ply
playerAccount
playerColor
lossPositive
lossClipped
```

因此构建新参考派生文件不需要重新运行 Level22 引擎。

## 5. 新派生参考文件

建议新增独立版本目录，不覆盖当前 v3 参考目录，例如：

```text
research/offbook_detection/data/
  oq_sentinel_elo_reference_level22_1600plus_v1_<date>/
```

主要文件建议为：

```text
directed_game_phase_records.jsonl
reference_build_audit.json
reference_source_manifest.json
reference_sha256_manifest.json
elo_calibration.json
elo_calibration_cases.jsonl
```

每条 `directed_game_phase_records.jsonl` 对应一个 `gameId + targetColor`，建议采用新 schema：

```text
player-sentinel-elo-directed-game-phase-v1
```

最低字段合同：

```text
schema
gameId
created
targetPlayerId
opponentPlayerId
targetColor
targetOldR
targetNewR
opponentOldR
opponentNewR
formalReferenceEligible
partitionScope
algorithmLabel
offBookPly
anchorSource
sourceLevel22File
sourceLevel22Sha256

metrics.full_game
metrics.post_offbook_inclusive
```

每套 `metrics` 至少保存：

```text
analysisStartPly
validLossNodeCount
phase1.validLossNodeCount
phase1.lossGe4Count
phase1.lossGe4Rate
phase2.validLossNodeCount
phase2.lossGe4Count
phase2.lossGe4Rate
phase3.validLossNodeCount
phase3.lossGe4Count
phase3.lossGe4Rate
phase4.validLossNodeCount
phase4.lossGe4Count
phase4.lossGe4Rate
completeFourPhase
equalPhaseGameGe4Rate
```

### 5.1 为什么每条参考记录要保存两套 metrics

目标局有两种口径：

```text
有锚点：post_offbook_inclusive
无锚点：full_game
```

不得把目标局的 `full_game` 指标与参考局的 `post_offbook_inclusive` 指标直接比较。为避免无锚点参考样本只有极少数，每个参考 directed game 都应从现有节点同时派生：

1. `full_game`：该目标方全部有效节点；
2. `post_offbook_inclusive`：仅当该参考方存在脱谱锚点时，从锚点开始并包含锚点。

运行时根据目标局口径选择参考记录中的同名 metrics。这样口径一致，又无需把无锚点目标局限制为只能匹配少量无锚点参考局。

### 5.2 阶段指标

对任一 metrics scope、任一阶段：

```text
phaseGe4Rate
= lossPositive >= 4 的有效目标方节点数
  / 该阶段有效目标方子损节点数
```

若某阶段没有有效节点：

```text
phaseGe4Rate = null
completeFourPhase = false
equalPhaseGameGe4Rate = null
```

若四阶段全部有效：

```text
equalPhaseGameGe4Rate
= (phase1Ge4Rate
   + phase2Ge4Rate
   + phase3Ge4Rate
   + phase4Ge4Rate) / 4
```

这里四阶段各占 25%。不得改成节点池化后的：

```text
sum(lossGe4Count) / sum(validLossNodeCount)
```

## 6. 目标对局选择

目标用户对局应按以下固定顺序处理：

1. 根据账号获取候选对局详情。
2. 确认账号恰好对应黑方或白方一侧。
3. 按现有流程获得 Level22 节点和算法脱谱标记。
4. 按第 5 节同一合同构建目标 directed game phase record。
5. 根据目标局 `algorithmLabel` 选择 metrics：
   - `offbook` 使用 `post_offbook_inclusive`；
   - `no_offbook` 使用 `full_game`。
6. 排除 `completeFourPhase = false` 的对局。
7. 排除对手 `oldR` 缺失、非有限数或位于正式参考 Elo 范围之外的对局；第一版不在对手 Elo 轴上外推。
8. 按 `created` 从新到旧排序。
9. 取最近最多 30 局有效对局。
10. 少于 10 局时不得给出正式估计；可以输出 `insufficient_target_games` 和诊断数据。

第一版不切 session。结果必须明确标注为“所选最近有效对局的平均行棋质量等价 Elo”，不得描述为用户永久棋力或严格的即时 Elo。

## 7. 参考泄漏排除

对真实目标用户和隐藏答案测试用户均必须执行：

1. 排除 `targetPlayerId` 或 `opponentPlayerId` 等于目标账号的所有参考 source games；
2. 排除 `gameId` 与任何目标局重叠的所有参考 directed records；
3. 一个 source game 的任一侧触发排除时，两侧 directed records 都排除；
4. 账号比较使用仓库现有 `account_key` 语义，忽略首尾空格和大小写差异。

不能只排除参考记录中目标用户作为 target 的一侧，否则同一用户或同一棋局仍可能通过 opponent 侧泄漏。

## 8. 单局参考邻居搜索

对每个试探 Elo `E` 和目标局 `i` 单独搜索参考邻居。

硬过滤条件：

```text
formalReferenceEligible = true
targetColor 与目标局相同
所选 metrics scope 与目标局相同
completeFourPhase = true
已通过第 7 节泄漏排除
RT_j 与 RO_j 均在正式参考 Elo 范围内
```

距离公式：

```text
targetGap   = RT_j - E
opponentGap = RO_j - O_i

distance_j
= sqrt(targetGap² + opponentGap²)
```

注意：

1. 距离中的 `E` 是试探 Elo，不是目标用户显示 Elo；
2. `O_i` 是该目标局对手的赛前 `oldR`；
3. 参考 ratings 使用该参考局赛前 `oldR`；
4. 两个 Elo 轴使用相同单位，不另加人为轴权重。

## 9. K 与距离权重

设硬过滤后的参考整局数为 `N`：

```text
K = ceil(N^(2/3))
```

要求至少存在 `K + 1` 条参考记录，以便取得核带宽。按以下稳定键排序：

```text
(distance, gameId, targetColor)
```

选择前 K 条，令第 K+1 条的距离为：

```text
boundaryDistance
```

前 K 条的权重：

```text
referenceWeight_j
= max(0, 1 - distance_j / boundaryDistance)
```

若 `boundaryDistance <= 0`、权重和为零或有效参考不足，应明确失败并输出 `insufficient_reference`；不得静默更换 K、回退到全库平均或添加任意常数。

K 由 N 决定，不得根据目标用户实际 GE4 或是否容易找到零点进行调参。

## 10. 单局参考均值、标准差与 gameZ

令参考邻居 `j` 在所选 metrics scope 下的：

```text
x_j = equalPhaseGameGe4Rate
```

距离加权参考均值：

```text
referenceMean_i(E)
= sum(referenceWeight_j * x_j)
  / sum(referenceWeight_j)
```

第一版采用可直接审计的加权总体方差：

```text
referenceVariance_i(E)
= sum(referenceWeight_j * (x_j - referenceMean_i(E))²)
  / sum(referenceWeight_j)

referenceSd_i(E)
= sqrt(referenceVariance_i(E))
```

令目标局的：

```text
a_i = equalPhaseGameGe4Rate
```

则：

```text
gameZ_i(E)
= (a_i - referenceMean_i(E))
  / referenceSd_i(E)
```

方向解释：

```text
gameZ > 0：目标局 GE4 更高，表现比试探 Elo 更差
gameZ < 0：目标局 GE4 更低，表现比试探 Elo 更好
gameZ = 0：目标局与试探 Elo 的局部参考均值相同
```

若参考标准差不是正有限数，当前目标局在该试探 Elo 下不可评分。不得使用 epsilon 把零标准差强行变成可评分结果。

`gameZ` 是数据库标准化差异，不得在报告中宣称它严格服从标准正态分布。

## 11. 多局局等权合并

目标用户有 `n` 局有效目标局时：

```text
candidateZ(E)
= (gameZ_1(E) + ... + gameZ_n(E)) / n
```

以及：

```text
score(E) = abs(candidateZ(E))
```

每局权重严格为 `1 / n`。不得按以下内容调整目标局权重：

```text
节点数量
对局输赢
目标局实际子损大小
目标局与参考均值的距离
对手知名度
时间先后
```

同一局的四阶段先合成为一个整局指标，因此四阶段共同变化会自然反映在参考整局分布的标准差中，不再使用额外相关性补偿。

## 12. Elo 网格搜索

逐整数计算：

```text
E = 1600, 1601, ..., 2495
```

每个 E 都必须重新：

1. 对每个目标局计算 Elo 距离；
2. 选择最近 K 条参考记录；
3. 计算距离权重；
4. 计算参考均值和标准差；
5. 计算各局 `gameZ_i(E)`；
6. 计算 `candidateZ(E)` 与 `score(E)`。

GE4 对 Elo 的变化梯度已经体现在 `referenceMean_i(E)` 随 E 的变化中。不得再把梯度乘入评分。可以在最终结果中记录局部梯度作为诊断，但它不参与主估计。

## 13. 点估计与状态

正常点估计：

```text
estimatedElo
= 使 score(E) 最小的整数 E
```

若两个相邻整数并列，使用较小整数并同时输出并列区间，不做未声明的小数插值。

至少实现以下状态：

```text
valid
insufficient_target_games
incomplete_phase_data
opponent_out_of_reference_range
insufficient_reference
above_reference_range
below_reference_range
multiple_crossings
low_resolution
calibration_unavailable
```

方向边界规则：

```text
若 candidateZ(E) 在整个范围内均小于 0：above_reference_range
若 candidateZ(E) 在整个范围内均大于 0：below_reference_range
```

边界状态不得强行输出 2495 或 1600 为正式预估 Elo。可以输出 `bestGridPoint` 作为诊断，但正式 `estimatedElo` 应为 null。

若存在多个彼此分离的过零区或相近最低区，应输出 `multiple_crossings`，不得只因其中一个低极小量就静默选择。

`low_resolution`、相近最低区和异常局部跳变的最终数值阈值必须由第 15 节的用户级校准确定，不得在实现时拍脑袋写死。

## 14. 不参与主估计的诊断

为保持结果可解释，应在最终 Elo 处额外输出：

```text
每局 gameZ
gameZ 的最小值、最大值、均值、中位数
mean(abs(gameZ))
四阶段目标用户局等权 GE4
四阶段参考期望 GE4
四阶段差异方向
candidateZ(E) 局部梯度
score 曲线的最低区域宽度
```

阶段诊断不得重新平方合并后影响主 Elo。`mean(abs(gameZ))` 仅用于识别“好局与差局强烈抵消”，其阈值也应从隐藏答案测试或已知用户分布中校准。

## 15. 用户级 95% 数据库校准

第一版不对目标用户做 bootstrap，也不按对局数或 Elo 分组。

### 15.1 校准对象

从当前正式参考用户中选择至少有 10 局有效目标记录的账号。现有数据库初步统计为 539 名，但实现后必须按新四阶段完整性合同重新统计，不能直接硬编码 539。

每个校准账号按照与真实目标完全相同的规则：

```text
选择最近最多 30 局有效对局
至少 10 局
不分 session
```

### 15.2 已知 Elo

对每个校准用户：

1. 在冻结源数据库的 `selected_account_bundle.json` 中找出该账号出现的全部详情；
2. 按 `created` 选择数据库内时间最新的一局；
3. 在该局 `players[]` 中按账号 ID 找到该用户；
4. 使用该玩家记录的 `newR` 作为 `knownElo`；
5. `knownElo` 仅在估计完成后核对，不得进入邻居距离、目标局选择、权重或评分。

该标签的准确表述是：

```text
冻结数据库中该用户最新一场已收录对局结束后的 Elo
```

它不保证是 Othello Quest 服务器完整生涯中的最后一局，因为当前参考数据库是筛选后的快照。

仅把 `knownElo` 位于正式参考范围内的用户纳入内部 95% 区间校准；范围外用户单独用于边界状态验证。

### 15.3 留出规则

估计某个校准用户时，必须从临时参考池中排除：

```text
该用户作为 target 的所有 source games
该用户作为 opponent 的所有 source games
全部对应的双侧 directed records
```

这是 leave-one-account-out，不是 leave-one-game-out。

### 15.4 阈值构建

对校准用户 `u` 保存完整曲线：

```text
score_u(E)
```

定义：

```text
minimumScore_u = min(score_u(E))

trueScoreIncrease_u
= score_u(knownElo_u) - minimumScore_u
```

若 `knownElo` 不是整数，可在两个相邻整数 score 之间做线性插值；必须在 schema 中记录插值规则。

全局阈值：

```text
T95
= trueScoreIncrease 的经验 95% 分位数
```

分位数算法必须明确、固定并写入 calibration artifact，建议复用仓库已有确定性 `weighted_quantile`/分位数语义或实现一个记录方法名的无权重经验分位数。不得使用目标用户调整 T95。

### 15.5 独立覆盖率验证

不得用同一批用户既确定 T95 又声称证明了 95% 覆盖率。按账号 ID 确定性拆分为：

```text
calibration users：确定 T95
validation users：只检查覆盖率、偏差和状态
```

拆分种子、比例和账号列表必须写入 manifest。具体比例可由实现者根据新合同下的有效用户数提出，但不得在没有独立 validation users 的情况下把范围宣传为“已验证 95%”。

如果独立验证覆盖率不足，状态应为 `calibration_unavailable` 或降低产品表述，不得继续标称 95%。

## 16. 真实目标用户的 95% 范围

若全局 T95 已通过独立验证：

```text
allowedScore
= minimumScore + T95
```

所有满足以下条件的整数 Elo 构成可接受集合：

```text
score(E) <= allowedScore
```

输出规则：

1. 若集合是一个连续区间，输出 `[lower, upper]`；
2. 若集合出现多个不连续区间，保留全部区间并输出 `multiple_crossings`；
3. 若区间触及 1600 或 2495，显式标记截断；
4. 不得把它称为理论正态置信区间；产品名称固定为“数据库校准 95% Elo 范围”。

## 17. 推荐 CLI 与输出

建议在现有 `scripts/analysis/sentinel_analysis.py` 增加独立子命令，或增加一个只负责此功能的新脚本。不要改变旧子命令参数和输出。

建议命令边界：

```text
build-elo-reference
calibrate-elo
estimate-elo
```

推荐输出文件：

```text
estimated_elo.json
estimated_elo_curve.csv
estimated_elo_games.csv
estimated_elo_phase_diagnostics.csv
```

`estimated_elo.json` 建议 schema：

```text
player-sentinel-estimated-elo-v1
```

最低字段：

```text
schema
account
referenceVersion
referenceManifestSha256
calibrationVersion
selectedGameIds
selectedGameCount
excludedGamesWithReasons
formalMinimumGameCount
formalMaximumGameCount
eloGridMinimum
eloGridMaximum
estimatedElo
bestGridPoint
minimumScore
candidateZAtBest
databaseCalibrated95Range
databaseCalibrated95Intervals
status
statusReasons
phaseDiagnostics
gameDiagnostics
curveFile
createdAt
```

CLI 和 JSON 中不得使用 `confidenceInterval` 这种容易暗示理论正态区间的字段名。

## 18. 配置与版本化

建议新增独立配置，例如：

```text
sentinel_elo_reference_config.json
```

而不是无版本地向旧 schema 添加字段。建议配置至少冻结：

```text
schema
version
sourceReferenceDirectory
derivedReferenceDirectory
directedPhaseRecords
referenceManifest
calibrationArtifact
formalEloMinimum = 1600
formalEloMaximum = 2495
minimumTargetGames = 10
maximumTargetGames = 30
phaseBoundaries = [30, 47, 53]
ge4Threshold = 4
neighborExponent = 2/3
distanceKernel = triangular_k_plus_one_boundary
eloGridStep = 1
calibrationCoverage = 0.95
calibrationGrouping = global
```

新参考库必须保存源文件 SHA-256、构建脚本 SHA-256、配置 SHA-256、Level22 合同和构建审计。构建过程不得修改或复制重写现有冻结 Level22 文件。

### 18.1 数据库扩展与校准缓存

`calibrate-elo` 生成的是当前参考数据库快照的校准缓存，不应在每次
`estimate-elo` 时重新计算。估计命令只读取该快照对应的
`elo_calibration.json`；`elo_calibration_cases.jsonl` 保存可复核的用户级校准明细。

当正式参考数据库新增对局或用户时，必须创建新的版本化参考目录并重新执行完整流水线：

```text
1. build-elo-reference：重新派生全部 directed game phase records、构建审计和 manifest
2. calibrate-elo：重新统计 >=10 局用户并重算 calibration/validation、T95 和诊断阈值
3. estimate-elo：让目标账号使用新的参考目录；已有估计若需反映新库必须重新运行
```

不得把新参考记录与旧 `elo_calibration.json` 混用。V1 不做自动扩展检测或增量校准；
数据库扩展不要求重跑旧 sentinel 的 `scan`、`freeze` 或旧输出流程。

### 18.2 单选手统一分析入口

与单个选手调查有关的指标统一由 `scripts/analysis/sentinel_unified_analysis.py` 暴露。
它读取旧 sentinel 已完成的 scoring/scan/freeze 产物，再读取 Elo Reference 与校准缓存，
输出包含两套结果的 `sentinel_unified_analysis.json`。`run_player_investigation.py` 只负责
调用这个阶段并把统一产物装入最终 `report.json`，不直接实现某一项指标。统一脚本同时
提供旧 sentinel 命令的兼容分发，因此旧脚本可以作为兼容接口保留，但不再是统一调查主入口。
后续新增单选手分析内容应扩展该统一脚本，不新增平行入口。

## 19. 推荐代码组织

现有 `src/player_analysis_toolkit/sentinel.py` 已较长，推荐把新算法放入：

```text
src/player_analysis_toolkit/sentinel_elo.py
```

建议职责拆分：

```text
phase_metrics_for_scope(...)
make_elo_directed_record(...)
build_elo_reference(...)
eligible_reference_records(...)
elo_distance(...)
neighbor_count(...)
nearest_weighted_neighbors(...)
weighted_reference_distribution(...)
score_game_at_elo(...)
score_candidate_curve(...)
classify_curve(...)
calibrate_global_interval(...)
estimate_database_calibrated_range(...)
```

可以复用现有模块的：

```text
read_json
write_json
write_jsonl
write_csv
sha256_file
account_key
disc_loss
_player_pair
```

不要复制粘贴并悄悄改变这些基础语义。

## 20. 必须实现的测试

### 20.1 阶段与节点合同

1. ply 30 属于阶段 1，31 属于阶段 2；
2. ply 47 属于阶段 2，48 属于阶段 3；
3. ply 53 属于阶段 3，54 属于阶段 4；
4. ply 60 属于阶段 4；
5. pass 不改变 global placement ply；
6. 有锚点时包含锚点；
7. 无锚点时 full_game 包含全部目标方有效节点；
8. 对手节点不能进入目标方指标；
9. 任一阶段空缺时整局不完整；
10. 四阶段综合率是四个比例等权平均，不是节点池化比例。

### 20.2 匹配与权重

1. 距离同时使用试探目标 Elo 与目标局实际对手 Elo；
2. 目标用户显示 Elo 不进入距离；
3. 同色硬过滤生效；
4. scope 使用目标局选择的同名 metrics；
5. K 严格等于 `ceil(N^(2/3))`；
6. 权重只由距离决定；
7. KNN 排序在并列距离下可复现；
8. 目标账号和目标 gameId 完整排除；
9. 数据库扩大时 K 根据新 N 更新。

### 20.3 gameZ 与局等权

1. 手算样例的加权均值、标准差和 gameZ 与实现一致；
2. 多局 candidateZ 是 gameZ 算术平均；
3. 增加节点不会直接改变一局在 candidateZ 中的权重；
4. 不计算或平均每局 Elo；
5. 正负 gameZ 可以在不同目标局之间平均；
6. 四阶段诊断不反向进入主评分。

### 20.4 曲线状态

1. 唯一内部过零得到 `valid`；
2. 全范围为负得到 `above_reference_range`；
3. 全范围为正得到 `below_reference_range`；
4. 多个分离过零得到 `multiple_crossings`；
5. 参考标准差为零时明确失败；
6. 少于 10 局不输出正式 Elo；
7. 边界状态不夹成 1600 或 2495。
8. 对手 oldR 超出正式参考范围的目标局被排除且记录原因。

### 20.5 校准

1. knownElo 取最新数据库详情的目标玩家 `newR`；
2. 该用户的所有 source games 均从参考池移除；
3. knownElo 不进入估计；
4. calibration 与 validation 用户不重叠；
5. 全局 T95 构建可复现；
6. 多区间、边界截断和无可用校准均有明确输出。

## 21. 实现验收报告

实现完成后必须生成一份 UTF-8 Markdown 验收报告，至少包含：

```text
新派生记录总数
full_game 四阶段完整记录数
post_offbook_inclusive 四阶段完整记录数
按颜色的正式参考数
重新统计的至少 10 局可校准用户数
calibration/validation 用户数
validation 实际覆盖率
Elo 误差的中位数、平均值、90% 与 95% 分位数
按真实 Elo 区间的误差方向诊断
above/below/multiple/low-resolution 状态数量
运行时间与峰值内存
所有新产物 SHA-256
```

在独立 validation coverage 未确认前，不得把功能标记为正式 95% 校准版本。

## 22. 明确不在 v1 范围内

```text
机器学习 Elo 模型
TCN Elo 回归
自动阶段边界搜索
session 自动切分
阶段节点数权重
参考节点暴露权重
阶段相关矩阵
candidate bootstrap
单局 Elo 后平均
小数 Elo 插值输出
超出参考范围后的外推
根据目标结果调 K 或调核宽度
```

## 23. 推荐实施顺序

1. 新增派生 record schema 和纯函数测试。
2. 从现有 10,244 个 Level22 文件构建双 scope 四阶段参考，不运行引擎。
3. 完成单目标局 KNN、距离权重、参考均值/标准差和 gameZ。
4. 完成多局局等权曲线与状态分类。
5. 完成 leave-one-account-out 用户级校准。
6. 冻结 T95，并在独立账号集验证覆盖率。
7. 接入 CLI，新旧 sentinel 路径并存。
8. 运行单元测试、集成测试、全库审计和验收报告。
9. 只有在审计通过后，才更新 README、METHOD 和正式配置入口。

## 24. 一页伪代码

```text
build reference:
    for each source game:
        load detail and existing Level22 nodes
        for targetColor in [black, white]:
            build full_game four-phase metrics
            build post_offbook_inclusive four-phase metrics when anchor exists
            store oldR, newR, IDs, color, created, provenance

select target games:
    build target directed phase records with the same contract
    offbook game -> use post_offbook_inclusive
    no_offbook game -> use full_game
    discard incomplete four-phase games
    sort newest first
    keep at most 30
    require at least 10

estimate:
    exclude every reference source game involving target account
    exclude every target gameId

    for E from 1600 to 2495:
        gameZs = []

        for target game i:
            choose the target game's metrics scope
            filter formal same-color complete reference records
            compute distance from (E, targetOpponentOldR)
            N = eligible reference count
            K = ceil(N^(2/3))
            select K nearest records
            set triangular weights using distance K+1 as boundary
            compute weighted mean and SD of equalPhaseGameGe4Rate
            gameZ = (targetGameRate - referenceMean) / referenceSd
            append gameZ

        candidateZ(E) = mean(gameZs)
        score(E) = abs(candidateZ(E))

    classify curve
    if valid:
        estimatedElo = integer E with minimum score
        calibrated range = E values with score <= minimumScore + T95
    output point estimate, range, curve, games, phases, status and provenance
```

本规格中的公式均为普通文本；实现和后续文档不得依赖 CLI 无法显示的 LaTeX。

## 25. estimated-Elo v2 补充合同（2026-08-28）

本节替代 v1 的“先平均四阶段比例，再计算 gameZ 和 candidateZ 过零”主评分。未被本节
替代的四阶段、目标棋选择、颜色/scope、账号整盘排除、Elo 网格和 UTF-8 合同继续有效。

### 25.1 版本隔离

v2 使用以下独立 schema，禁止读取 v1 calibration 作为 v2 calibration：

```text
player-sentinel-elo-reference-config-v2
player-sentinel-elo-conditional-reference-v1
player-sentinel-elo-calibration-v2
player-sentinel-estimated-elo-v2
```

正式配置候选为 `sentinel_elo_reference_config_v2_20260828.json`。在完整 calibration 和
独立 validation 完成前，仓库顶层主配置仍保持 v1，不得切换。

### 25.2 阶段计数与 Beta-Binomial

每阶段直接使用：

```text
x = lossPositive >= 4 的目标方着手数
n = 该阶段有效目标方子损着手数
```

近邻盘 `j` 保留 `(x_j,n_j,w_j)`。令 `alpha=m*kappa`、`beta=(1-m)*kappa`，最大化：

```text
sum_j w_j * log(
  C(n_j,x_j) * B(x_j+alpha,n_j-x_j+beta) / B(alpha,beta)
)
```

实现使用 SciPy `special` 和 `optimize`。缺少 SciPy、输入非法或优化不收敛时，该阶段明确
失败；不得用默认分布或 v1 正态化语义继续。相同 `(x,n)` 的似然项可先精确合并权重，
该变换与逐盘求和代数等价，不是近似拟合。确定性矩估计作为首个初值；只有首个解失败或
命中边界时才尝试固定第二初值，并保留尝试、边界和状态审计。

目标计数使用：

```text
PExact = Beta-Binomial P(X=x)
q = P(X<x) + 0.5*PExact
targetZ = inverse_standard_normal_cdf(q)
```

`q` 在求逆前固定裁剪到 `[1e-12,1-1e-12]`。`targetZ` 只进入下一阶段条件距离和诊断。

### 25.3 标准化距离和一阶相邻条件

每个颜色/scope 池缓存固定的 `selfEloSd`、`opponentEloSd`、`phase1ZSd`、
`phase2ZSd` 和 `phase3ZSd`。第一阶段距离使用标准化双方 Elo；第二阶段只加入 z1，
第三阶段只加入 z2，第四阶段只加入 z3。正式候选默认 `previousZWeight=1`。

每阶段都从该账号排除后的完整允许参考池重新计算 N、K、K+1 边界和三角权重；不得从上一
阶段的 K 盘子集继续筛选。该关系是预先编码的一阶相邻条件关系，不是因果结论。

### 25.4 统一冻结 reference-z 与直接账号隔离

根据 2026-08-28 的最终实施修订，v2 不强制五折 reference feature cache。它构建一套统一
冻结的 `reference_z1/reference_z2/reference_z3` 缓存。估算任一目标账号时：

1. 找出该账号作为任一方参与的全部 source game；
2. 删除这些 source game 的黑白两条 directed records；
3. 再从剩余记录重新计算 N、K、K+1 边界和权重。

Reference 自身的 z 准备仍对每条 R 使用 R 账号的整盘 source-game 排除。统一缓存不是“对
每个未来目标账号重新完全交叉拟合全部派生 reference z”，文档和报告不得作这种声明。
正式验收需在少量分层账号上比较统一缓存与“先排除该账号、再重建缓存”的 estimatedElo、
区间端点和近邻哈希变化。

`knownElo` 始终只在估算曲线完成后用于误差、方法对照和 T95；不得进入距离、目标棋选择、
Beta-Binomial 拟合或 reference-z 准备。

### 25.5 v2 主评分与曲线状态

每盘主评分为四阶段精确计数负对数概率之和：

```text
gameNll = -(log(P1)+log(P2)+log(P3)+log(P4))
J(E) = selected games 的 gameNll 算术平均
estimatedElo = 使 J(E) 最小的内部整数 Elo
```

阶段有效手数较多会自然贡献更多概率证据。这是相对 v1“四阶段先等权平均比例”的有意语义
变化。`meanConditionalZ` 仅为方向诊断，不参与主评分或过零分类。

曲线状态至少包括 `valid`、`above_reference_range`、`below_reference_range`、
`multiple_minima`、`multiple_intervals`、`low_resolution`、`unstable_curve`、
`insufficient_reference`、`beta_binomial_fit_failed` 和 `calibration_unavailable`。

### 25.6 可恢复命令和 calibration 边界

维护入口为：

```powershell
python scripts/analysis/sentinel_elo_analysis.py --config sentinel_elo_reference_config_v2_20260828.json prepare-conditional-elo-reference --output-dir <versioned-dir> [--resume]
python scripts/analysis/sentinel_elo_analysis.py --config sentinel_elo_reference_config_v2_20260828.json calibrate-elo --output-dir <versioned-dir> [--resume]
python scripts/analysis/sentinel_elo_analysis.py --config sentinel_elo_reference_config_v2_20260828.json audit-estimate-coverage --output-dir <new-dir>
python scripts/analysis/sentinel_elo_analysis.py --config sentinel_elo_reference_config_v2_20260828.json estimate-elo ...
python scripts/analysis/sentinel_elo_analysis.py --config sentinel_elo_reference_config_v2_20260828.json audit-reference-z-sensitivity --account <account> --bundle <bundle> --engine-dir <level22-dir> --offbook-records <json> --rebuild-dir <versioned-dir> --output-dir <new-dir> [--resume]
```

prepare 按账号分片原子提交并验证 config/input SHA。calibration 固定使用
`ProcessPoolExecutor(max_workers=16)`、一个 task 一个账号、`chunksize=1`、worker 内
`referenceQueryWorkers=1`，并逐账号原子写 case。

数据库级 calibration 只用 calibration 账号确定 T95、曲线诊断阈值和 A/B/C 方法对照；
validation 账号仅评估覆盖率、区间宽度和误差。若独立 validation 未确认至少 95% 覆盖，
产物状态必须是 `calibration_unavailable`。

敏感性命令执行真实的账号排除后重建：在计算任何尺度或 reference z 之前删除目标账号任一
方参与的整盘 source game。输出比较 unified/rebuilt 点估计和最佳点 phase 近邻集合 SHA。
重建 cache 的 manifest 与统一 calibration 不一致，因此不能直接沿用其正式 T95 区间；在
新的 calibration 合同建立前，重建侧区间端点明确记为不可用。

calibration 账号会在同一账号任务中记录三种方法：A 为历史四阶段等权 signed-z，B 为
Beta-Binomial 且 `previousZWeight=0`，C 为正式候选的 `previousZWeight=1`。validation 任务只
计算 C；B/C 的误差比较只从 calibration 账号得出，validation 不参与权重选择。为控制内存，
A 使用条件缓存中保存的四阶段 `(x,n)` 重建历史等权比例，不在每个 worker 额外加载完整 directed
JSONL；该比例与冻结源指标等价。
