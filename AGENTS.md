# Repository workflow constraints

## OQ Reference 扩展的黑白有向 Elo 分桶约束

- 任何 OQ 哨兵 Reference 的新增、扩容或重建任务开始前，Agent 必须提醒用户：正式的对局拉取、容量评估和后续分析应优先按“黑棋 Elo 桶 × 白棋 Elo 桶”的有向二维分区执行，不能只查看无向双方 Elo 配对桶。
- 有向分区表的行必须是黑棋 `oldR` Elo 桶，列必须是白棋 `oldR` Elo 桶；左上角明确标为“黑棋＼白棋”。表格必须填充完整 9×9 正式矩阵，不能用镜像标记代替实际单局方向计数。
- 正式矩阵的每个单元格按唯一 `gameId` 的实际黑方与白方 Elo 桶计数。跨档棋局在源 Reference 中只存储一次、只运行一次 Level22；其实际黑白方向只占一个有向单元格，但在 Sentinel 派生层仍生成黑方和白方两条 directed target records。
- 扩展选择和审计必须同时保存有向单元格与无向桶视图：有向表用于发现黑白颜色方向的覆盖缺口、安排抓取和分析；无向桶用于维持既有 45 桶容量合同、目标数量和兼容性校验。两种视图的总量和逐局映射必须能够相互核对。
- 每次扩展至少记录有向单元格的当前数量、本地缓存合格容量、新快照候选容量、最终选择数量、最终数量、剩余缺口、失败/重复/合同失败数量及容量耗尽状态；不能只报告全库总局数或无向桶汇总。
- 有向表的 Elo 上界必须从本轮冻结排行榜动态推导；最低 Elo、桶宽和正式矩阵结构按当前配置执行，不得把历史最高分或历史矩阵数量硬编码为本轮结果。
- Agent 输出扩展计划、阶段状态或最终体量时，应同时向用户展示或明确提供这张“黑棋 Elo × 白棋 Elo”有向表，并说明是否还提供无向 45 桶兼容汇总。

## Unified single-player investigation entrypoint

- Any new metric, diagnostic, or analysis that is specifically about investigating one player must be added to `scripts/analysis/sentinel_unified_analysis.py` and exposed through its unified per-player command surface.
- `scripts/analysis/run_player_investigation.py` is the lifecycle orchestrator: it prepares inputs, invokes the unified analysis stage, and assembles the final report. Do not put the analysis implementation directly into this orchestrator.
- The legacy `sentinel_analysis.py` and the database-maintenance `sentinel_elo_analysis.py` CLIs remain compatibility/maintenance interfaces. Their per-player results must be consumable by the unified analysis output and final report.
- Do not add another standalone per-player investigation script when the capability can be implemented as a stage or metric in the unified analysis script.
