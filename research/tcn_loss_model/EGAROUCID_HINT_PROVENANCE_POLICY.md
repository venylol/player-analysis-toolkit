# Egaroucid hint 计算与中间产物留存合同

本合同适用于今后所有生成正式训练数据的 Egaroucid hint1/hint6 脚本。

## 已确认事故原因

2026-08-04 的服务器脚本把多个 future 通过 `index % worker_count` 复用到同一
持久引擎。引擎锁只保护单条 `command()`；`setboard()` 和 `hint()` 分别加锁，
不是同一个事务。因此可能出现：

`setboard(A) -> setboard(B) -> hint(A)`

此时 A 的结果实际来自 B。响应还可能重复，原本属于 A 的搜索可能根本没有
执行，所以不能假设错误数据只是可逆置换。批次流水线层面的锁不能修复同一
hint6 引擎内部的命令交错。

## 强制计算合同

1. 每个引擎实例只能通过原子的 `setboard + hint` 接口执行局面搜索。锁必须
   覆盖两条命令及完整响应读取；不得让调用者分别调用这两条命令。
   Console 启动参数不得包含 `-q` 或 `-noboard`，因为本机 7.8.1 只有在两者
   都不存在时才会在 hint 响应中回显原生棋盘。
2. 多 worker 时，每个 worker 最好独占一个引擎进程。即使调度器会把多个任务
   分配给同一引擎，原子事务仍是强制要求。
3. 每条实着节点必须分别保存：
   - 来源棋盘 `board_setboard`；
   - 发给引擎的 `hint1_request_board_setboard`；
   - 从 hint1 响应原生 8×8 棋盘及 `BLACK/WHITE to move` 解析得到的
     `hint1_board_setboard`；
   - 发给引擎的 `hint6_request_board_setboard`；
   - 从 hint6 响应原生回显解析得到的 `hint6_board_setboard`；
   - hint 参数、引擎 SHA-256、book 配置、线程数、hash、请求/worker/batch 标识；
   - 原始 hint1 与 hint6 响应字段。
4. `hint1_board_setboard` 和 `hint6_board_setboard` 不得使用请求值或最终共享
   字段回填；必须从各自 hint 响应的 Console 原生棋盘文本现场解析，并与完整
   原始响应一起写入中间产物。响应缺少完整 8 行棋盘或先手时必须立即失败。
5. pass 节点保留在时序原始表中，但不伪造 hint 搜索或分数。

## 正式验收门槛

只有以下条件全部通过，才允许生成或发布正式完整训练数据：

- game_id、节点数、实着及 pass 数量符合冻结样本 manifest；
- `(game_id, move_index)` 唯一，节点未跨 game_id 划分；
- 每个实着节点满足来源棋盘、两个请求棋盘、两个 Console 响应棋盘五者相等；
- hint1 走法合法；hint6 恰有 `min(6, n_legal_moves)` 个完整、合法候选；
- hint 参数、引擎及 book 哈希符合冻结配置；
- 标签、pass 跨越、TCN 时序、当前着手隐藏、特征名称和顺序通过正式合同；
- 数值输入为 362 项、棋盘输入为 23 通道；
- 正式 model-ready 文件、合同报告、manifest 与 SHA-256 均已落盘。

任何一项失败都必须保留原始产物并明确失败，不得通过改写服务器/引擎产物
掩盖错误。

## 中间产物留存与回收

以下内容在最终验收前必须保留：原始棋谱、局面任务表、独立 hint1/hint6
结果、引擎日志、批次/请求/worker 元数据、进度文件、失败记录、manifest、
合同报告与哈希记录。

脚本不得自动删除、清理、覆盖或自动移动这些文件。只有完整正式训练数据已经
生成、全部最终验收通过并留下可核验的验收记录之后，中间产物才“具备进入
Windows 回收站的资格”。实际移动仍需用户明确指示，并且只能进入回收站，
不得永久删除或清空回收站。
