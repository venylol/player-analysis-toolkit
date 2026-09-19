# OQ 哨兵 Reference 黑白有向 600 扩充任务 Prompt

工作目录：
`C:\Users\MeroAF\Desktop\比赛编排\player_analysis_toolkit`

这是实际实施任务。请先以 UTF-8 完整阅读根目录和项目目录的 `AGENTS.md`，再检查仓库现状、配置、manifest、进程和审计。不得只依据本 prompt 中的历史数字执行。

## 核心目标

以现有、已验证的 550 局黑白有向 Reference 作为唯一基线，执行增量扩充，使每个正式黑棋 oldR 桶 × 白棋 oldR 桶单元格的目标上限调整为 600 局。已有 550 链中的棋谱、Level22 结果、Sentinel records、Player Elo phase records、Anscombe/reference-z 缓存和已验证 calibration 结果必须优先复用并核验 SHA；不得把已有 550 局重新分析或重复运行。

基线源库：
`research/offbook_detection/data/oq_elo_matchup550_blackwhite_reference_level22_1600plus_20260829`

基线 Sentinel、Player Elo、Anscombe 和 calibration 路径必须从实际配置和 manifest 读取，不得硬编码猜测。新批次必须使用独立目录和唯一 batch ID，不能覆盖 550 目录。

## 分桶与选择合同

正式维度保持最低 Elo 1600、桶宽 100、9×9 有向矩阵。行是黑棋 oldR，列是白棋 oldR，左上角标为“黑棋＼白棋”。最后桶上界由本轮冻结排行榜动态推导，且不得低于历史正式上界；不得新增独立 2500 桶。

每格选择公式固定为：

```text
final_count = max(
    existing_550_black_white_count,
    min(600, frozen_valid_capacity)
)
```

只选择缺口数量 `max(0, 600-existing_550_count)`。已有棋谱全部保留；容量不足时必须记录 `remainingGap` 和 `capacityExhaustedBelowTarget=true`，不得接纳不合格棋谱或把容量不足报告为流程失败。理论上限为 `81 × 600 = 48,600`，不是强制最终总数。

反向单元格独立计数。源库每局只属于一个实际黑白方向；Sentinel 派生层每局生成黑方和白方两条 directed target records。45 个无向桶只能由 81 格派生：`unordered(A,B)=directed(A,B)+directed(B,A)`，不得参与新增选择。

## 增量来源和验证

来源优先级为：现有 550 Reference；本地已完整验证缓存和旧冻结快照候选；本轮唯一新快照。所有来源按 `gameId` 全局去重。新增候选必须重新验证 gameId、reversi 模式、5 分钟合同、黑白 oldR、账号顺序、完整合法落子、时钟、provenance、重复和失败状态。缺失或冲突数据不得进入新源库。

selection seed、calibration split seed 必须彼此不同且不同于 550 批次，并写入配置、manifest、审计和报告。稳定排序至少包含 canonical black-white cell key、gameId、sourcePriority、stableSha256。

网络最多执行一轮冻结排行榜、一轮相关选手列表和按缺口请求的详情；必须 direct=true、禁用代理、有限重试，成功项恢复时不得重复请求。

## 新批次产物

建议目录名：
`oq_elo_matchup600_blackwhite_reference_level22_1600plus_<RUN_DATE>`

对应 Sentinel、Player Elo、Anscombe、calibration 和配置文件均使用 `matchup600_<RUN_DATE>` 独立命名。必须包含有向/无向分区表、选择审计、扩充 manifest、容量审计、provenance、详情验证、Level22/WLD/off-book 输出、engine index、完成审计和最终 SHA-256 manifest。每格必须同时记录 550 基线数量、本地缓存容量、新快照容量、最终选择数、最终数、缺口、容量耗尽、无效、重复、合同失败和 merged gameIds。

## Level22 与下游

550 基线中合同和 SHA 匹配的 Level22 文件只允许核验后复制到新目录；新增棋谱每个 gameId 只运行一次 Egaroucid Level 22，保持现有 worker、thread、hash、book 和 WLD 合同。新目录运行时不得依赖旧目录。

完成源库审计后，再增量构建新 Sentinel、Player Elo phase records、Anscombe/reference-z、KNN 审计和 calibration。新阶段必须证明输入来自 600 批次；已有 550 结果只在 manifest/SHA 和算法合同完全匹配时复用。不得使用旧 550 calibration case、progress 或旧 validation 结果冒充新批次结果；若校准输入发生变化，应按新 600 records 重新生成并完整验证。

## 生命周期与正式切换

复用现有配置驱动生命周期入口，提供 `run`、`resume`、`status`，记录 batch ID、配置 SHA、阶段输入/输出 SHA、状态、时间、进程、恢复命令和下一阶段。目标目录已存在时，同 batch 且合同一致只能 resume；batch 或合同不同必须明确失败，不得覆盖。

只有新 600 源库、Sentinel、Player Elo、Anscombe、calibration、validation、所有审计和测试全部通过后，才生成候选正式配置并原子切换。切换前保存精确配置快照；切换后最小测试失败时恢复该快照。不得出现新 records 搭配旧缓存或新 calibration 搭配旧 source manifest。

最终报告必须包含真实完整 9×9“黑棋＼白棋”矩阵、45 格无向兼容汇总位置、基线局数、新增局数、复用 Level22 数、新增 Level22 数、Sentinel records、Player phase records、Anscombe records、calibration/validation 人数、T95、validation coverage 及所有失败/容量耗尽说明。

旧 550 目录只有在新 600 链完成正式切换并证明不再被运行时引用后，才可按既有回收站流程处理；默认保留，未经明确审计不得回收。
