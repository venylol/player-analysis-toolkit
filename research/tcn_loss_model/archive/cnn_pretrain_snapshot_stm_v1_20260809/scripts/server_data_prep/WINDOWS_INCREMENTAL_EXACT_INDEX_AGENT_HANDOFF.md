# Windows 9950X 增量合并与数据装配交接

## 目标

在不丢失服务器当前三小时以上已提交进度的前提下，合并可复用 hint6，只计算缺失节点，随后完成分阶段审计、10,000 局装配以及 362 数值特征/23 棋盘通道物化。服务器无 CUDA；严禁训练、微调或评估正式模型。

## 第一步：先停止旧运行

在旧 v3 运行窗口按一次 `Ctrl+C`，等待 Python 和 12 个 Egaroucid Console 正常退出。不得删除、清空、改名或覆盖旧 v3 目录中的 `work\hint6\batches`。增量入口若发现任何目标 Egaroucid 进程仍存在，会拒绝启动。

增量脚本不会改写旧批次；所有新结果写入旧 v3 目录下新的：

```text
incremental_attempts\oq_hint6_incremental_exact_index_20260804_v4
```

## 不可更改的索引合同

合并前必须先配对索引：

1. 对局只按完全相同的 `game_id` 配对。
2. 节点只按完全相同的 `(game_id, move_index)` 合并。
3. `move_index` 是 OQ 原始索引，明确的 `-` pass 占一个索引。
4. `source_ply_including_pass` 必须一致；`global_placement_ply` 排除 pass，但绝不作为 merge key。
5. 完整 `board_setboard`、`side_to_move`、`actual_move` 也必须一致。
6. 禁止按相邻 ply、只按棋盘、或把 pass 压缩后重新编号来补配。

任何一项不一致必须停止，不得让 Agent 猜测或手工改 CSV。

## 数据优先级

同一精确键只保留最高优先级：

1. 服务器当前已提交且重新审计通过的记录；
2. 本包本机 12-worker 原生 Console 记录；
3. 本包本机早期原子化原生 Console 记录；
4. 381,437 条精确索引、完整棋盘、候选合法性筛查通过的旧记录；
5. 只对剩余键重新计算。

服务器记录永远不被本包覆盖。worker 数只是并行度，不影响单请求原子化记录的复用资格。

## 预检（不会启动 Console）

在增量包解压目录运行，替换旧 v3 解压目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_incremental_merge_and_data_prep.ps1 `
  -ServerRoot "D:\path\to\oq_tcn_windows_9950x_data_prep_20260804_v3" `
  -PreflightOnly
```

预检必须生成旧 v3 根目录的 `INCREMENTAL_DELIVERY_STATUS.json`，状态应为 `preflight-complete`。检查 `prepare_manifest.json`：

- `ok: true`；
- `indexContract.boardOnlyRemappingUsed: false`；
- `selectedRows + missingRows = 599112`；
- `priority` 顺序未改变。

## 正式继续

```powershell
.\run_incremental_merge_and_data_prep.cmd "D:\path\to\oq_tcn_windows_9950x_data_prep_20260804_v3"
```

固定 hint6 参数：12 workers、每 worker 独占一个 Console、每 Console 16 threads、level18+book、hash25、batch 128、timeout 900、最多 2 次。不得使用 `-q` 或 `-noboard`。

重跑同一命令会从新计算阶段已提交批次继续。不要同时开第二份。

## 分阶段硬门

1. 校验本增量包全部 SHA-256。
2. 审计服务器现有批次及两个本机原生批次。
3. 执行上述 pass-aware 精确索引配对。
4. 再检查请求棋盘、Console 响应棋盘、合法候选与候选数量。
5. 只计算 `missing_hint6_source.csv` 中的节点并做完整原生响应审计。
6. 按优先级装配恰好 609,124 行：599,112 实着、10,012 pass、10,000 局。
7. 旧筛查记录明确标记为 `legacy-exact-key-board-and-legality-screened`；不得伪装成原生响应棋盘记录。
8. 物化 362 数值特征和 23 棋盘通道并进行 CPU checkpoint 合同验证。

只有旧 v3 根目录下 `INCREMENTAL_DELIVERY_STATUS.json` 同时出现：

```json
{"status":"complete","stage":"data-ready","trainingStarted":false,"cudaUsed":false}
```

并且增量 attempt 中 `results\model_ready\server_final_validation.json` 的 `ok` 为 `true`，才算完成。

## 禁止事项

- 不删除、清理、覆盖旧 v3 或本增量包的任何证据。
- 不修改 worker/thread/level/book/hash/sample/源棋谱。
- 不进行网络拉取或替换对局。
- 不启动 TCN 训练、微调、预测或正式评估。
