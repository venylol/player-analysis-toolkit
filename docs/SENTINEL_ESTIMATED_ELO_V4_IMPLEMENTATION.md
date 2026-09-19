# Sentinel estimated-Elo v4

状态：当前正式算法实现；2026-09-11 的 600 Reference 已完成全链路校准。

v4 的版本标识为
`estimated-elo-v4-anscombe-local-gaussian-adaptive-grid-global-knn-v1`，使用独立的
`sentinel_elo_reference_config_v4_matchup600_20260911.json`、reference-z 缓存、calibration schema 和
产物目录。v1/v2/v3 产物只用于历史读取、兼容和复现实验，不参与 v4 估计。

## 数学合同

每个阶段从已有 Level22 记录读取整数 `x` 和 `n`，不重新运行 Level22，并计算：

```text
y = 2 asin(sqrt((x + 3/8) / (n + 3/4)))
v = 1 / (n + 0.5)
```

阶段 1 的 KNN 坐标是候选 self Elo、目标 opponent Elo；阶段 2、3、4 依次只增加
上一阶段的 reference z。坐标按冻结的 self/opponent Elo SD 和上一阶段 z SD 标准化。
每次账号查询先过滤账号任一方、目标 gameId 和同源另一方向记录，再按稳定 record key
排序；`N_allowed` 在过滤后计算，`K = ceil(N_allowed^(2/3))`，用第 K+1 个合格邻居定义
三角权重边界。

局部统计是闭式计算，不拟合参数：

```text
mu = sum(u*y)
C = 1 - sum(u^2)
observedVariance = sum(u*(y-mu)^2) / C
samplingVariance = sum(u*(1-u)*v) / C
tauSquared = max(0, observedVariance - samplingVariance)
muVariance = sum(u^2*(tauSquared+v))
predictiveVariance = tauSquared + targetV + muVariance
targetZ = (targetY-mu) / sqrt(predictiveVariance)
phaseScore = 0.5*(log(2*pi*predictiveVariance) + targetZ^2)
```

`J(E)` 是四阶段 `phaseScore` 之和再对固定目标 game 集合取平均；点估计为整数 Elo
搜索中的最小值。`phaseScore` 的正式名称是
`negativeLogPredictiveDensity`，不是离散精确计数概率。

## Reference 准备

`prepare-conditional-elo-reference` 从已有 directed phase records 生成单份冻结缓存，严格
按 phase1 → phase2 → phase3 → phase4 生成 reference z，并保存每个颜色/scope 池的
`y`、`v`、z 分布、均值、标准差、输入 SHA、模型合同 SHA 和阶段进度。账号分片、阶段文件、
最终 records 和 manifest 都是原子写入；`--resume` 只接受合同和 SHA 完全一致的进度。

缓存使用按 `targetColor × metricsScope × phase` 建立并复用的全局 cKDTree。账号路径只在
查询结果上过滤，不建立账号专属树；每次查询都可通过稳定暴力排序审计复核。

## Elo 搜索与校准

搜索顺序固定为 `40 → 20 → 10 → 5 → 2 → 1`。1600–2500 的 40 和 20 网格完整扫描并
显式包含 2500；后续保留多个低谷、平台和边界下降趋势。自适应成本不再低于剩余完整网格，
或低谷没有两侧保护时，按合同扫描完整 901 点网格。`knownElo` 只在搜索完成后探测。

calibration 第一阶段只用 calibration 账号计算
`trueScoreIncrease = J(knownElo) - minimumJ` 并冻结 `T95`；第二阶段只用冻结的
`minimumJ + T95` 评估 validation。每账号一个进程任务、16 个 worker、`chunksize=1`、
worker 内部查询线程为 1。任务、账号统计和模型输入失败分别计数，progress 和账号 case
均可恢复。

## 命令

```powershell
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v4_matchup600_20260911.json `
  build-elo-reference

python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v4_matchup600_20260911.json `
  prepare-conditional-elo-reference

python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v4_matchup600_20260911.json `
  calibrate-elo
```

单玩家正式入口仍是 `scripts/analysis/sentinel_unified_analysis.py`；其生命周期仍由
`scripts/analysis/run_player_investigation.py` 编排。调试 `estimate-elo` 可以在没有 v4
calibration artifact 时输出点估计，但状态必须保持 `calibration_unavailable`，不能把它
称为数据库校准 95% 区间。

v4 正式输出保存 `referenceModelContractSha256`、`searchContractSha256`、
`calibrationContractSha256`、输入/reference manifest SHA 和 calibration artifact SHA。
阶段诊断保存 `x`、`n`、`transformedY`、`samplingVariance`、`localMeanY`、
`observedVariance`、`betweenGameVariance`、`meanEstimateVariance`、`predictiveVariance`、
`targetZ`、`negativeLogPredictiveDensity`、`N_allowed`、`K`、边界距离和邻居集合 SHA。

## 单玩家估计的并行调度

当前单玩家 `estimate-elo` 热路径按“一个目标对局一个任务”分片，默认使用最多 4 个
worker 进程。每个 worker 在初始化时构建一次自己的 reference index，并在该进程收到的
后续候选 Elo 点任务中复用；候选 Elo 点之间仍按既有搜索合同串行。同一目标对局的
phase1 → phase4 保持顺序执行，因为后一个 phase 依赖前一个 phase 的 target z。进程结果
按冻结的 `target_records` 序号重新组装，因此不改变 `J(E)` 的算术平均、KNN 稳定排序、
搜索步骤或数值结果（只允许平台级极小浮点误差）。

这是用户要求的内存控制变更。`referenceQueryWorkers` 继续固定为 1，避免形成 4×N 的
嵌套并行；目标局进程上限为 4。正式 calibration 保留冻结合同：外层仍是 16 个
账号级进程，每个账号 worker 显式使用 `target_game_workers=1`，不会再创建目标局进程池。
估计输出的 `parallelization` 审计字段记录进程模式、配置/实际 worker 数、任务数、单局任务
单位、局内 phase 顺序、reference index 初始化策略以及聚合后的查询计数。
