# 选手调查流水线 Agent 运行手册

本文面向后续负责运行 `scripts/analysis/run_player_investigation.py` 的主 Agent，记录实际操作中最容易影响结果或造成不必要中断的经验。本文只讨论运行方法，不记录历史代码缺陷。

## 1. 开始前固定调查口径

开始运行前，先在会话中明确并复述以下信息：

- 账号，包括大小写、下划线等完整字符。
- 举报时间使用哪个时区。
- 起止时间是否为闭区间。
- 用户给到分钟时，结束时间是精确到 `HH:MM:00`，还是包含整个 `HH:MM` 分钟。
- 是否使用区间外全部当前可查询对局作为对照组。

时间必须写成带偏移的 ISO 8601。中国标准时间示例：

```text
2026-08-08T21:00:00+08:00
```

如果用户说“到 22:51 分”且确认包含整个 22:51 分钟，闭区间结束值应明确写为：

```text
2026-08-08T22:51:59.999999+08:00
```

不要默认为整分钟或整点。边界附近可能恰好存在对局，差几十秒就会改变举报组与对照组。

## 2. 每次调查使用独立目录

为每个账号和调查日期创建唯一输出目录，例如：

```powershell
python scripts/analysis/run_player_investigation.py start `
  --account "example_player" `
  --output-dir "investigations\example_player_investigation"
```

不要复用其他选手的目录，也不要在已冻结分组的目录中改换时间区间。已完成阶段同时校验命令摘要和输出 SHA-256；手工修改阶段产物会使续跑校验失败。

所有 PowerShell 会话建议先设置 UTF-8：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 3. 启动后先核对棋谱目录

`start` 完成后，顶层状态应为 `awaiting_group_selection`。先读取：

- `game_catalog.json`
- `account_bundle.json`
- `progress.json`

核对账号、总局数、`created` 时间和时区换算，再选择分组。不要仅凭列表顺序推断“最后几局”。

时间区间选择示例：

```powershell
python scripts/analysis/run_player_investigation.py select-groups `
  --run-dir "investigations\example_player_investigation" `
  --reported-from "2026-08-08T21:00:00+08:00" `
  --reported-to "2026-08-08T22:51:00+08:00"
```

分组完成后，必须从 `progress.json` 再确认：

- `reportedGameIds`
- `controlGameIds`
- `excludedGameIds`
- 转换为 UTC 后的 `reportedFrom` 和 `reportedTo`

最好同时检查区间上下界前后的相邻对局，避免分钟边界理解错误。

## 4. `select-groups` 是长命令

`select-groups` 会执行 Player 画像、Level22、算法脱谱登记、hint1、hint6、安全装配以及后续模型与统计阶段，直到生成最终报告。

个人调查使用固定并行契约：

- Level22：12 个独立 Console worker，每个 16 线程，hash25，逐局原子提交，完成后全量 audit。
- hint6：12 个独立 Console worker，每个 16 线程，Level18、book、hash25、batch 128、timeout 900、最多 2 次，完成后全量 audit。
- Tournament 直接调用 Level22 runner 时默认只有 2 个 worker，避免比赛期间资源抢占；调查编排器会显式传入 12。

因此：

- 不要给外层命令设置两分钟之类的短超时。
- 允许底层命令长时间运行，在另一终端或工具调用中轮询 JSON。
- 每次轮询间隔建议 30–60 秒；不要用长时间阻塞的 `sleep`。
- 命令暂时没有标准输出不等于卡死，应以进程和嵌套进度为准。

主要嵌套进度文件：

```text
progress.json
engine_level22/progress.json
hints/hint1/progress.json
hints/hint6/progress.json
model/adapters/personal_ensemble_progress.json
```

hint 阶段的总实着位置数来自：

```text
hint_source/source_manifest.json -> shape.placements
```

将它与 hint `progress.json` 中的提交计数比较，可以得到真实完成比例。

## 5. Level22 后的算法脱谱登记

Level22 完成并通过 audit 后，`offbook_detection` 自动生成 `offbook_records.json`。
不需要 Agent 审查包或手工提交标记。每局必须具有 `algorithmLabel`，取值为
`offbook` 或 `no_offbook`，并记录锚点来源与计算证据。

规则固定为：只分析目标棋手实着；候选闭区间 ply 5–38；当前用时严格大于此前
本人全部实着用时中位数的1.75倍时形成用时候选；首次严格满足
`abs(bestEval) > 6` 的本人节点形成估值截断点。用时规则仅搜索截断点之前的
节点；优先取更早的用时候选，否则取截断点，两者都没有则标记 `no_offbook`。

## 6. 算法标签与模型口径

模型评估口径为：

- 有算法锚点的举报局：从锚点手开始，包含锚点手。
- 算法标签为 `no_offbook` 的举报局：使用目标选手整局全部着手。
- 无锚点局会出现在 `fullGameFallbackNoOffbookGameIds`，不会获得合成锚点。

算法标签生成后会继续执行 hint、模型物化、Profile 物化、12成员个人适配、举报局推理、Bootstrap、常规子损/WLD/时间统计和最终报告。

## 7. 状态驱动的断点恢复

先读取顶层 `progress.json`，再决定动作：

| 顶层状态或条件 | 操作 |
| --- | --- |
| `awaiting_group_selection` | 执行 `select-groups` |
| 预审阶段 `running` | 继续轮询，不重复启动 |
| 任一阶段 `failed` | 确认没有残留计算进程，再执行 `resume` |
| `offbook_detection` 或后续阶段未完成 | 执行 `resume` |
| `completed` | 校验并读取 `report.json` |

续跑命令：

```powershell
python scripts/analysis/run_player_investigation.py resume `
  --run-dir "investigations\example_player_investigation"
```

查看状态：

```powershell
python scripts/analysis/run_player_investigation.py status `
  --run-dir "investigations\example_player_investigation"
```

外层工具超时后，不要立即再启动一个相同任务。先检查 Python 和 Egaroucid 进程以及嵌套进度，确认旧进程确实已经结束，避免两个进程同时写同一运行目录。

顶层 `lastError` 可能保留已恢复的历史错误。最终判断应同时查看顶层 `status`、各阶段状态、`completedStages/totalStages` 和最终产物哈希。

## 8. 完成标准

只有满足以下条件才算完成：

- 顶层 `status` 为 `completed`。
- `completedStages == totalStages`，通常为 `16/16`。
- `final_report.status` 为 `completed`。
- `report.json` 存在，且实际 SHA-256 与 `progress.json` 一致。
- `report.json.status` 为 `completed`。
- 最终 JSON 中的举报局、对照局和时间选择仍与用户口径一致。

流水线不生成 Markdown。完成后直接从 `report.json` 向当前会话汇报：

- 举报组和对照组局数。
- Level22 子损点估计、Bootstrap 区间和精确组合位置。
- WLD 差值及区间。
- 12 模型实际减预期残差及区间。
- 思考时间经验位置和显著异常项。
- 算法锚点、算法标签及无锚点整局回退情况。
- 样本量、时间控制归一化和“统计结果不是作弊概率”等限制。

不要只看点估计下结论。区间跨零时应明确说明证据不确定；思考时间单项异常也不能脱离其他指标直接解释为作弊。

## 9. 推荐的最短操作清单

```text
1. 复述账号、时区、精确秒级闭区间。
2. start，核对 game_catalog.json。
3. select-groups，核对边界相邻对局和冻结后的两组 ID。
4. 每 30–60 秒轮询顶层和当前 nestedProgress。
5. 确认 Level22 后的 `offbook_detection` 生成每局算法标签。
6. 失败时先查残留进程，再 resume。
7. 校验 16/16、report.json 状态和 SHA-256。
8. 不生成 Markdown，直接在会话总结结果与限制。
```

## 10. 哨兵模式

没有预先举报局、需要自动筛查最近最多30局有坐标落子的棋局时，使用：

```powershell
python scripts/analysis/run_player_investigation.py start-sentinel `
  --account "example_player" `
  --output-dir "investigations\example_player_sentinel" `
  --reference-config sentinel_reference_config.json
```

该模式不等待人工选组。依次完成 acquisition、profile、Level22、双向兼容的算法
脱谱、二维 Reference 评分、10,000名伪玩家完整扫描和名单冻结。只有冻结清单具有
非空 `reportedGameIds` 且模型对照至少8局时才继续 hint 与12成员个人适配。

恢复前先看 `progress.json`；已完成阶段的命令摘要和输出 SHA-256 一致时，`resume`
不会重跑。正式判断以 `selection_manifest.json` 为准，模型报告不得更改该文件。
`external_uniform_anomaly`、`internal_variation_only` 和 `no_clear_signal` 的
`reportedGameIds` 都为空，不应为了进入旧模型流程而人工补组。

## 11. 哨兵 estimated-Elo v2/v3 运行门槛

单玩家正式入口仍是 `start-sentinel`，不得新增平行调查脚本。统一阶段会读取配置指定的
directed reference、conditional reference 和 calibration，并在任一 schema/SHA 不匹配时
停止。最终 `report.json.estimatedElo` 必须与
`sentinel_unified_analysis.json.estimatedElo` 来自同一个共享核心。

在主配置切到 v2 或 v3 前，Agent 必须确认：

1. `conditional_reference_audit.json.ok=true`，全部 pool 的 Elo/z 标准差为正；
2. 正式 calibration 记录 `parallelWorkers=16`、`taskUnit=one_player_account`、
   `processPoolChunksize=1`、`referenceQueryWorkersPerWorker=1`；
3. calibration 与 validation 账号不重叠；
4. validation coverage 至少为 0.95；
5. A/B/C 对照和统一缓存敏感性对照已写入验收报告；
6. 对应版本 calibration、cases、conditional records 和 manifest SHA 全部通过。

2026-08-29 的 v3 calibration 已生成 918 个 calibration case 和 178 个 validation case，
但 1,096 条曲线均为显式 `beta_binomial_fit_failed`，因此 `t95=null`、覆盖率未确认，
artifact 状态为 `calibration_unavailable`。在重新验证拟合合同前不得将 v3 配置作为主配置。

任一条件未满足时，保持顶层正式 v1 配置不变。调试运行允许产生 v2 点估计，但状态必须是
`calibration_unavailable`，不得把空区间描述为“数据库校准 95% 范围”。

### 11.1 当前正式 estimated-Elo v4

当前单玩家统一流程使用 `sentinel_elo_reference_config_v4_matchup600_20260911.json`。v4 的
reference cache、calibration 和输出 schema 均独立于 v1/v2/v3；统一入口只消费 v4
artifact，不把历史版本作为回退路径。v4 使用 Anscombe 方差稳定变换、过滤后 `N_allowed`
的 ceil K、相邻阶段 z 条件 KNN 和闭式局部高斯预测密度。完整合同见
[`SENTINEL_ESTIMATED_ELO_V4_IMPLEMENTATION.md`](SENTINEL_ESTIMATED_ELO_V4_IMPLEMENTATION.md)。

本轮 600 Reference 的 reference-z 缓存、完整 calibration 和独立 validation 已完成；后续复核使用：

```powershell
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v4_matchup600_20260911.json calibrate-elo
```

该命令固定 16 个进程、每任务一个账号、`chunksize=1`、worker 内部查询线程为 1，先由
calibration 账号冻结 T95，再使用冻结阈值计算 validation。`knownElo` 只在曲线搜索结束后
探测。reference 扩展或新缓存准备仍须同时保留黑棋 Elo 桶 × 白棋 Elo 桶的完整 9×9 有向表
和无向 45 桶兼容汇总。
