# Sentinel estimated-Elo v2 验收报告（2026-08-29 更新）

状态：代码、可恢复数据准备、正式条件 Reference 缓存和测试已完成；按用户指示，尚未运行
calibration，不能验收为正式数据库校准版本，也未切换顶层主配置。

## 1. 算法摘要

每个阶段直接使用 GE4 次数 `x` 和有效目标方着手数 `n`。对同颜色、同 scope 的精确 K
近邻保留 `(x_j,n_j,w_j)`，通过 SciPy 对 Beta-Binomial 的 `m` 和 `kappa` 做加权最大似然
拟合，其中 `alpha=m*kappa`、`beta=(1-m)*kappa`。

目标阶段保存精确概率 `PExact`，并用 `P(X<x)+0.5*PExact` 的 mid-CDF 计算条件 z。phase2
距离只加入 z1，phase3 只加入 z2，phase4 只加入 z3；每阶段都从完整允许参考池重新取 K。

单盘评分是四阶段 `-log(PExact)` 之和。候选 Elo 评分 `J(E)` 是所有所选目标棋单盘评分的
算术平均。点估计是 1600–2500 整数网格中 `J(E)` 的可识别内部主最低点。z 不参与主评分。

## 2. Reference feature 隔离

本版采用统一冻结的 reference z1/z2/z3 缓存，不采用五折缓存。估算目标账号时删除该账号
作为任一方参与的全部 source games，并同时删除这些游戏的两条 directed records；随后从
剩余记录重新计算 N、K、K+1 边界和三角权重。

Reference 自身准备 z 时，每条 R 使用排除 R 账号全部 source games 的池。该设计是目标账号
原始 source-game 的直接 leave-one-account-out 隔离，不宣称对每个未来目标账号完全重建
全部派生 reference z。分层账号的“统一缓存 vs 账号排除后重建缓存”敏感性对照仍待正式
条件缓存完成后运行。

`knownElo` 只允许用于曲线完成后的误差、A/B/C 对照和 T95，不进入估算特征或拟合。

## 3. 数据与版本

| 产物 | SHA-256 |
|---|---|
| v8 directed phase records（37,770 条，102,799,153 bytes） | `971EE5754A28B8BDDB3A437E4664605C9E677D06C057E53810C19239A604E4F8` |
| v8 reference SHA manifest | `4573E292B41E7CB3DB0113A711A8B82FEA31BD8658216AEE5FCCBF4B7EBDC3ED` |
| v2 候选配置 | `D7CE518A9258F2C3BB3C8EE10A22A2E7A030845BEBC8B2F0947EDC45447AD70D` |

正式条件缓存基于 71,304 条颜色/scope 记录已完成，目录为：

```text
research/offbook_detection/data/oq_sentinel_elo_conditional_reference_v1_20260828
```

缓存 manifest SHA-256 为 `B25A6E5F0DF4EC1F078B1D60285052F50E0EA3033ED9BD1215B5B18569C5E3CC`，
记录文件 SHA-256 为 `DBACF9FAA5B99862B493880AAFE8D5CED45F3E38C835A625D7DCF7912629B1AF`。
目录共 894 个文件、449,696,014 bytes；根目录记录文件为 143,104,919 bytes。

四个颜色/scope 池的固定尺度和阶段 K 中位数如下（K 按每条记录的实际 N 计算，仍保留
精确 K+1 边界和三角权重）：

| pool | records | self SD | opponent SD | z1 SD | z2 SD | z3 SD | K1/K2/K3/K4 median |
|---|---:|---:|---:|---:|---:|---:|---|
| black / full_game | 18,715 | 188.373 | 189.350 | 0.9113 | 0.9496 | 0.8090 | 705/704/703/701 |
| black / post_offbook_inclusive | 16,777 | 187.568 | 189.198 | 0.8677 | 0.9506 | 0.8096 | 655/655/654/653 |
| white / full_game | 18,723 | 189.394 | 188.342 | 0.9193 | 0.9471 | 0.8177 | 705/705/704/702 |
| white / post_offbook_inclusive | 17,089 | 187.999 | 188.238 | 0.8941 | 0.9458 | 0.8176 | 664/663/662/661 |

四个池合计的阶段有效/失败记录为 phase1 `71,201/103`、phase2 `71,030/274`、phase3
`70,826/478`、phase4 `70,557/747`。失败记录保留 `fitStatus`，不会静默替换分布。
缓存准备使用阶段依赖波次：phase2 等待全部 phase1 z/SD，phase3 等待 phase2，phase4
等待 phase3；每个波次的待处理账号分片通过 `ProcessPoolExecutor(max_workers=16)`（尾部
不足 16 个分片时按待处理数缩小）并行计算，单分片原子提交。正式目录记录了 16 个阶段波次
和 872 个分片，progress 状态为 `completed`；累计阶段计时为 2,463.4 秒（含此前中断后的
续跑，不能当作单次墙钟时间）。最后一次续跑对待处理分片使用 16 个 worker，同时保留此前
已提交的分片；再次 `--resume` 只做合同/哈希校验并复用全部已提交分片。

## 4. 已完成的正确性与性能验证

小缓存包含四个“颜色 × scope”池，每池 40 条，共 160 条。prepare 首次用时约 12.9 秒；
第二次 `--resume` 约 1.4 秒，只校验并复用已提交分片。各池 z1/z2/z3 标准差均为正；拟合
失败记录保留为无效记录，没有默认分布回退。

已知账号 `00hamosuke00` 使用 29 盘有效目标棋运行完整 1600–2500 网格：

```text
901 candidate Elo × 29 games × 4 phases = 104,516 次阶段查询/拟合
elapsed = 199.9 seconds
smoke diagnostic best grid point = 1631
status = calibration_unavailable
```

该点估计只证明完整控制流、性能和输出合同，不是正式 Elo 结果。smoke estimate SHA-256：
`87B86A94A78F7CD2B92F0377FE375177925876210ECF6D46316738F523871054`。

性能优化只采用代数等价变换：相同 `(x,n)` 的似然项先合并权重，以及沿 Elo 升序使用前一
候选的 MLE 作为确定性 warm start；首解失败或命中边界时仍运行固定第二初值。没有用矩估计
替代正式 MLE。

当前相关测试 49 项通过；官方 `python -m unittest discover -s tests -v` 共 155 项通过。覆盖手算概率、`alpha=4,beta=16,n=3,x=1` 概率
`0.35324675324675325`、mid-CDF、相同比例不同证据、加权 MLE、二项极限、极端参数、CDF
裁剪、优化器失败、标准化距离、相邻 z、完整池重新取 K、账号双方向排除、NLL 曲线状态、
16 进程合同、schema 拒绝、统一入口共享核心、账号排除重建敏感性合同和最终报告编排。

calibration 实现已包含 calibration 账号内的 A/B/C 记录：A 为历史四阶段等权 signed-z，B
为 `previousZWeight=0`，C 为正式 `previousZWeight=1`；validation 任务只运行 C。A 使用
条件缓存的四阶段 `(x,n)` 恢复等权比例，worker 不重复加载完整 directed JSONL。

## 5. v3 Calibration / validation（2026-08-29 已运行）

按后续指示，使用 `sentinel_elo_reference_config_v3_20260829.json` 运行正式命令：

```powershell
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v3_20260829.json `
  calibrate-elo
```

运行在 UTC 04:10:07 创建 progress，UTC 04:43:24 完成；中途一次 progress 原子替换遇到
Windows `WinError 5`，保留并隔离了 40 个已验证 case 到
`recovery_orphan_cases_20260829`，随后用 `--resume` 完成，未删除任何数据。最终结果为：

- calibration accounts：918；validation accounts：178；两集合按 SHA-256 账号排序确定性拆分且互不重叠；
- calibration/validation case：918/178，总计 1,096；失败任务 0，重试 0；
- worker 合同：`parallelWorkers=16`、`taskUnit=one_player_account`、`processPoolChunksize=1`、
  `referenceQueryWorkersPerWorker=1`；
- 所有 1,096 条 case 的 `curveStatus` 为 `beta_binomial_fit_failed`，因此没有可纳入 T95 的
  有效 calibration 曲线；这是显式失败状态，不是默认分布回退；
- `t95=null`、validation 覆盖率为 0、最终状态为 `calibration_unavailable`，不得宣称数据库
  校准 95% 区间，也不得切换顶层 v1 主配置；
- 共执行 2,198,058 次 Beta-Binomial 拟合和 cKDTree 查询；仅 1 个账号触发完整网格 fallback，
  平均每账号评估 25.18 个 Elo 点；
- 单账号运行时间 median/mean/p90/p95 为 23.245/26.197/38.188/46.332 秒；本次续跑墙钟
  801.5 秒，监控到的 Python 进程峰值工作集约 12.8 GB。

validation 误差（由于没有 T95，区间宽度均为空）为：

| 分组 | case 数 | 绝对 Elo 误差 median | mean | p90 | p95 |
|---|---:|---:|---:|---:|---:|
| overall validation | 178 | 85.09 | 100.45 | 204.50 | 238.94 |
| 目标棋 10–14 | 430（全 case） | 98.35 | 121.02 | 242.28 | 303.04 |
| 目标棋 15–19 | 229（全 case） | 98.72 | 107.53 | 208.58 | 248.44 |
| 目标棋 20–24 | 137（全 case） | 77.21 | 101.48 | 208.45 | 250.29 |
| 目标棋 25–30 | 300（全 case） | 76.07 | 91.53 | 192.94 | 238.75 |

按 known Elo 区间的完整分组指标已写入 calibration artifact 的 `groupMetrics`；按
`full_game/post_offbook_inclusive` 和黑白颜色的审计字段仍保持在 case 输出中。由于所有曲线
均显式拟合失败，A/B/C 实证比较、T95 区间端点和 reference-z 敏感性对照均没有可发布结果。

正式产物 SHA-256：

| 产物 | SHA-256 |
|---|---|
| `elo_calibration_v3.json` | `53E8BAFC2A0D78415513F413B9B87F8A40E014597EEE9D2DC441DE59BF0F1BFD` |
| `elo_calibration_cases_v3.jsonl` | `632C1DCD5D5D3A5E0943D9F46F64E161EE1F716826BBAC448D1AF161F5AB53F3` |
| `calibration_sha256_manifest_v3.json` | `EECAEB240051CE32CB43FBF82B52EF3977F25279B27960A6803D26CAF5A8BB35` |
| `estimate_coverage_audit_v3.json` | `ED93DA02A8ADCAC187B6DAB0457AE7DD4E8EB8CA09818467DC406B4F02DE4231` |
| `estimate_coverage_cases_v3.csv` | `78755BE828551BC861EF2D1955458B48ADC4F0583ED108C29768AB1E6CCC234D` |

审计目录为 `research/offbook_detection/data/oq_sentinel_elo_coverage_audit_v3_20260829`。

## 6. 已知限制

1. 统一 reference-z 缓存不是每个目标账号的完全派生特征交叉拟合。
2. Beta-Binomial MLE 计算量较高；small-cache 单账号完整网格约 200 秒，完整缓存需重新基准。
3. 当前 v3 adaptive search 在本数据上所有账号均出现 Beta-Binomial 拟合失败，因而不能产出
   T95；需要先修复或重新验证拟合/搜索合同，再考虑第二次 calibration。
4. 当前报告是完成运行但未通过正式覆盖验收的记录；`calibration_unavailable` 是有意保留的
   安全状态，不是成功的数据库校准结论。
