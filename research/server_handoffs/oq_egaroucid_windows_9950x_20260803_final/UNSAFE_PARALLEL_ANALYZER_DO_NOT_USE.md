# 禁止使用当前并行 hint6 分析脚本

`research/analyze_oq_reversi_5min_hints.py` 已被硬性禁用，仅保留作来源证据。

原因：同一个持久 Egaroucid 引擎可能被多个 future 交错执行 `setboard` 和
`hint`。单条命令虽然有锁，但这两条命令不是同一个原子操作，导致 hint6
响应写入错误局面。2026-08-04 的 10,000 局交付审计已证明该问题同时影响
原始保留的 9,790 局和新增替补的 210 局。

具体交错点是：调度器把多个 future 用 `index % worker_count` 分配给同一个
持久引擎，而引擎锁只覆盖一条 `command()`。`setboard()` 和 `hint()` 是两次
独立加锁，所以同一引擎可能按以下顺序执行：

`setboard(A) -> setboard(B) -> hint(A)`

服务器后加的批次流水线锁只协调批次及 hint1 阶段，没有覆盖同一个 hint6
引擎的整次 `setboard + hint` 事务，因此没有消除这个问题。

未来实现必须同时满足：

- 单个引擎上的 `setboard + hint` 必须由同一把锁作为不可分割事务执行；
- 不得使用 `-q` 或 `-noboard` 抑制 Console 的原生棋盘回显；
- hint1、hint6 分别保存请求 board state，以及从各自 hint 响应原生棋盘文本解析的
  `hint1_board_setboard`、`hint6_board_setboard`；不得用请求值回填响应字段；
- 正式写出前逐行验证来源棋盘、两个请求棋盘、两个 Console 响应棋盘五者相等，
  并验证全部候选合法且完整；
- 原始棋谱、任务表、hint1/hint6 响应、日志、进度及 manifest 都是验收证据；
- 只有完整正式训练数据生成且最终合同验收全部通过后，中间产物才允许进入回收站；
- 任何脚本都不得自动删除或自动回收这些中间产物。

服务器最终 ZIP 内的脚本保持原样，以便校验交付哈希；不得再次运行。
