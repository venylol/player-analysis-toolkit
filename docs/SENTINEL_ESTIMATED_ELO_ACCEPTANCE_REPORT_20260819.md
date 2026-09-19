# 哨兵模式行棋质量等价 Elo V1 验收报告

日期：2026-08-19  
参考版本：`oq_sentinel_elo_reference_level22_1600plus_v1_20260819`  
状态：独立 validation 已确认数据库校准覆盖率达到 95%，校准 artifact 状态为 `validated`

## 1. 实施范围

已新增独立的 `sentinel_elo` 模块、配置和 CLI：

- `src/player_analysis_toolkit/sentinel_elo.py`
- `scripts/analysis/sentinel_elo_analysis.py`
- `sentinel_elo_reference_config.json`
- `tests/test_sentinel_elo.py`

旧 `sentinel.py`、旧 sentinel CLI、固定 Reference、scan、freeze 和旧输出语义未改动。
新实现不使用机器学习或 TCN。

## 2. 参考派生审计

| 项目 | 数值 |
|---|---:|
| 源参考对局 | 10,244 |
| 新 directed game phase records | 20,488 |
| `full_game` 四阶段完整记录 | 20,440 |
| `post_offbook_inclusive` 四阶段完整记录 | 20,401 |
| 正式参考记录 | 20,256 |
| 新记录中双方 `newR` 完整 | 是 |
| Level22 源文件复制数 | 0 |
| 源文件哈希复核 | 通过 |

正式完整记录按颜色和 scope：

| 颜色 | `full_game` | `post_offbook_inclusive` |
|---|---:|---:|
| black | 10,101 | 10,080 |
| white | 10,107 | 10,089 |

构建审计 `reference_build_audit.json` 为 `ok=true`。新 JSONL 的 20,488 个
`(gameId, targetColor)` 键唯一；每阶段比例、四阶段等权综合率和缺阶段 `null`
合同已对全量记录复核。

## 3. 用户级校准

按新四阶段完整性合同重新统计，至少有 10 局有效目标记录的用户数为 536，未使用
规格中的旧初步数 539。按 `sha256(seed|accountKey)`、seed `20260819` 确定性拆分：

| 项目 | 数值 |
|---|---:|
| calibration 用户 | 428 |
| validation 用户 | 108 |
| calibration 有效案例 | 361 |
| validation 有效案例 | 108 |
| `T95` | 0.47849639827259294 |
| validation 覆盖 | 106 / 108 |
| validation 实际覆盖率 | 98.1481% |
| 校准状态 | `validated` |

已知 Elo 只取冻结源 bundle 中该账号按 `created` 最新详情的赛后 `newR`，并在
估计完成后核对。calibration 和 validation 账号集合不重叠。校准使用 16 个账号
worker；每个 worker 内部使用 1 个树查询线程，避免 16×16 线程过度竞争。

本次校准结果是参考数据库快照级缓存：`estimate-elo` 只读取
`elo_calibration.json`，不会每次运行重新计算满足至少 10 局的用户。若后续新增正式
参考对局，需要重新构建新的版本化参考目录，并重新生成 directed phase records、构建
审计、`reference_sha256_manifest.json`、`elo_calibration.json` 和
`elo_calibration_cases.jsonl`；随后已有账号也应重新估计，才能使用扩展后的 Reference
分布和新的 `K`。当前流程不自动检测数据库扩展，也不做增量校准；旧 sentinel 的
`scan`、`freeze` 和旧输出不受影响。

validation 的绝对 Elo 误差：

| 指标 | 数值 |
|---|---:|
| 中位数 | 98.7850 |
| 平均值 | 108.2882 |
| 90% 分位数 | 204.6312 |
| 95% 分位数 | 263.2627 |

按 validation 已知 Elo 区间的有符号误差方向（正值表示估计偏高）：

| 已知 Elo | 样本数 | 平均有符号误差 | 偏高 | 偏低 | 相等 |
|---|---:|---:|---:|---:|---:|
| 1600–1799 | 14 | -6.6457 | 6 | 8 | 0 |
| 1800–1999 | 27 | -0.3043 | 12 | 15 | 0 |
| 2000–2199 | 48 | 6.3560 | 23 | 25 | 0 |
| 2200–2399 | 16 | 46.5522 | 9 | 7 | 0 |
| 2400–2495 | 2 | 48.2850 | 2 | 0 | 0 |

## 4. 曲线状态审计

下表统计 536 个 calibration/validation 案例的内部曲线状态；边界状态不夹成
1600 或 2495。

| 状态 | 全部案例 | validation |
|---|---:|---:|
| `valid` | 452 | 91 |
| `above_reference_range` | 34 | 10 |
| `below_reference_range` | 50 | 7 |
| `multiple_crossings` | 0 | 0 |
| `low_resolution` | 0 | 0 |
| `calibration_unavailable` | 0 | 0 |

## 5. CLI 集成 smoke test

由于本次没有指定最终目标账号，未把任何账号当作用户最终报告对象。使用已有的
`Dentist_` 调查目录进行了明确标注的集成 smoke test：

| 项目 | 结果 |
|---|---|
| 选择有效目标局 | 18 局 |
| 状态 | `valid` |
| 点估计 | 1804 |
| 数据库校准 95% 范围 | [1600, 2005] |
| 下界截断 | 是 |

Smoke 输出位于 `tmp/estimated_elo_smoke_dentist_20260819/`，不代表用户最终结论。

正式单选手调查由 `scripts/analysis/run_player_investigation.py` 编排，在旧 sentinel 的
scoring、pseudo scan 和 freeze 完成后调用统一入口
`scripts/analysis/sentinel_unified_analysis.py`。该入口将旧 sentinel 摘要与本次
`estimated_elo` 结果写入 `sentinel_unified_analysis.json`；最终 `report.json` 同时保留
旧指标，并包含 `estimatedElo` 和 `unifiedSentinelAnalysis`。后续单选手指标应继续扩展
这个统一入口。统一入口也兼容旧 sentinel 的 acquire/score/scan/freeze 命令，旧脚本仅
作为兼容接口保留。

## 6. 测试与运行资源

- 新 Elo、统一入口、编排器和旧 sentinel 测试：49 passed。
- 根目录 `tests/`：117 passed，6 个 subtests passed。
- 全仓库 `pytest -q`：未作为通过标准；仓库内既有归档/研究副本包含同名测试模块
  和已不存在的旧 `src.*` 依赖，导致 43 个 collection errors。这些错误发生在
  `research/**/archive`、旧 TCN 研究目录及重复测试树，不涉及本次新增路径。
- 参考双 scope 派生运行时间：约 14.5 秒。
- 16-worker 全量校准运行时间：约 4 分钟。
- 校准期间观测峰值工作集：约 5.33 GB。
- 估计 smoke 运行时间：约 2.7 秒。

## 7. 新产物 SHA-256

`reference_sha256_manifest.json` 自身不纳入自己的文件列表；下表列出目录内全部
新参考产物：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `directed_game_phase_records.jsonl` | 37,121,266 | `ff66acb8e5b51829e70e7045f74dd897f82109a2a6fed191bdd137a538d71316` |
| `elo_calibration.json` | 356,010 | `9574ccab35cee24d0ec695d82ff5ade7cea396180666b25abad45031e4cda418` |
| `elo_calibration_cases.jsonl` | 35,556,877 | `6d1446e32c3fb2218134ba9a596b2d7a6b91d60dd4250d5cd9472251c6b2093c` |
| `reference_build_audit.json` | 1,036 | `fe8abefe015bf6a775144d83b19895c0efd59d06378bd70796aa2a303946c33e` |
| `reference_source_manifest.json` | 3,533 | `39637b196b4256aff40372c5b418f4435b1520f82278d93899095e66f2428f12` |
| `reference_sha256_manifest.json` | 1,225 | `f0df6b88c7fba3101b7174ee91444020b13572b793b06f81a17dffe602ee872c` |

实现、配置和 smoke 产物：

| 文件 | SHA-256 |
|---|---|
| `sentinel_elo_reference_config.json` | `044af2319e1acca02b3f5c42ef5fdc799d7094b834626b37e5eb5fe7a5c8d909` |
| `src/player_analysis_toolkit/sentinel_elo.py` | `d38bc33c39a4df65624be24272484bd58f529bd350e7cc62256eb9bcf106d7fb` |
| `scripts/analysis/sentinel_elo_analysis.py` | `c6f06813f4a11cb85599fd6bfc8808e7b71fa109f276b6eb7fecbb4116b21a75` |
| `tests/test_sentinel_elo.py` | `ce6b5801419249932d39cd52bd6e736656e3f151a25f331f66e1dbf47029239e` |
| `scripts/analysis/sentinel_unified_analysis.py` | `4a4afb6d131e04c60508ee928defdbf1ea3913a9585e8234ded942bbeb6dc149` |
| `scripts/analysis/run_player_investigation.py` | `7103c14c6d128b632ba570e42731b2e29d1d416938ddbb08b1e8ae0290649719` |
| `tests/test_sentinel_unified_analysis.py` | `960c24d2d0f70e43efd3ff36c68fbe2046a2f07e7dbb8234b58832dc894721cf` |
| `tests/test_sentinel_v1.py` | `8f31f8d60ce2ecd30b08dd13430b6ac44a076babe7197c55826478fcafbcfe5f` |
| `AGENTS.md` | `8a09fe1ca4ccdaf0f397ae2d2c819c7564782bd068764867b5b46e59ddf106dc` |
| `tmp/estimated_elo_smoke_dentist_20260819/estimated_elo.json` | `67dd2d493e8164c24541ecdc7dd140eff33c96334c72c852ac82288f9f2c7e97` |
| `tmp/estimated_elo_smoke_dentist_20260819/estimated_elo_curve.csv` | `d2c0fb8100c9efe4a9113db9a91f2e253be1feabd8049c5847c48d9be54f7c82` |
| `tmp/estimated_elo_smoke_dentist_20260819/estimated_elo_games.csv` | `080dc01606054329f441f34ebdf8815f472d3e210da9869c7e2fcd8139458b01` |
| `tmp/estimated_elo_smoke_dentist_20260819/estimated_elo_phase_diagnostics.csv` | `635685d844b2102e18b13bf19284a1ff17dbf353e2ef186fe6497fa46ce71985` |
| `tmp/sentinel_unified_smoke_dentist_20260819/sentinel_unified_analysis.json` | `4699300009e90c7c603c04e0ede591ecfd5086a74021bc09ab535fcd411c3a2d` |
| `tmp/sentinel_unified_smoke_dentist_20260819/estimated_elo/estimated_elo.json` | `40ec97fb62a3e7d95e8e7c64f4aac385448155db18b851e7cb1862896601b4a1` |

验收结论：V1 的独立数据库校准已具备可复现产物和独立覆盖率证据；真实用户估计仍
应由调用者明确提供账号及其目标 Level22 输入，不应把本报告的 smoke 账号结果当作
默认用户结论。
