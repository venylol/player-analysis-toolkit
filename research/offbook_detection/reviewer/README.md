# 脱谱候选实时回放审查器

## 直接使用

1. 双击打开 `offbook_replay_reviewer.html`。
2. 点击“导入审查数据”。
3. 选择 `data/monotonic_evidence_log_time_ratio_multiscale_v1_replay_bundle_20260811.json`。
4. 从左侧选择对局，保持 `1×（一比一）` 后点击“播放”。
5. 停在目标棋手某一手落子前，点击右侧“标记当前锚点”；若本局没有明确锚点，可点击“本局标为 no_offbook”。
6. 点击顶部“导出复核 JSON”，保存包含全部导入对局及复核结果的新批量包。

播放器在每手落子前按原始 `thinkingTimeMs` 等待。进入算法候选手的思考阶段时，棋盘外圈为红色；进入原 Agent 锚点时为黄色；两者重合时为紫色。时间线中的 `A` 和 `M` 分别表示算法和 Agent 节点，点击该行会跳到对应落子前。

页面是单文件离线工具。导入的数据和复核状态只保留在当前页面内，不上传；只有点击“导出复核 JSON”时才由浏览器下载新文件，不会覆盖原文件。时间线中的 `R` 表示你的锚点。

## 批量 JSON

推荐格式为 `offbook-replay-review-bundle-v1`。一个文件可以包含多位选手和多局对局，每局保留：

- 目标棋手、执色、对手和对局 ID；
- 完整实际落子序列；
- 每手原始思考毫秒数；
- 黑白双方初始总时限；
- 黑白双方实际落子思考时间合计；
- 算法候选 ply、目标决策号、置信等级和支持尺度；
- 原 Agent 判断和锚点 ply。
- 页面复核结论 `reviewerMark`：`offbook` 时记录锚点 strict ply、目标决策号和标记时间；`no_offbook` 时锚点字段为 `null`。尚未复核的对局为 `null`。

`summedThinkingTimeMs` 只合计实际坐标落子的原始时间；pass 不属于严格落子 ply。`initialTimeLimitMs` 是该方开局总时限。

页面也能同时导入多个 `player-offbook-agent-review-packet-v1` JSON 和多尺度 `cross_validated_game_summary.csv`，但这种方式不会自动补充初始总时限。

## 重新生成当前批量包

```powershell
python research/offbook_detection/reviewer/build_offbook_replay_bundle.py `
  --review-packet investigations/bambooL_investigation_20260809/offbook_packet.json `
  --review-packet investigations/lianmian0v0_investigation_20260809/offbook_packet.json `
  --review-packet investigations/SilverSiro_investigation_20260809/offbook_packet.json `
  --review-packet investigations/xiaoqi_investigation_20260809/offbook_packet.json `
  --review-packet investigations/z779_profile_tcn_20260805/z779-offbook-packet.json `
  --review-packet investigations/zertion_intake_20260808/zertion-offbook-packet.json `
  --algorithm-summary research/offbook_detection/outputs/monotonic_evidence_log_time_ratio_multiscale_convergence_v1_20260811_v2/cross_validated_game_summary.csv `
  --cohort-manifest research/tcn_loss_model/config/manual_offbook_time_baseline_cohort_20260811.json `
  --output research/offbook_detection/reviewer/data/choose-a-new-output-name.json
```

生成脚本拒绝覆盖已有输出。它会校验严格 ply 连续性、着手坐标、思考时间、算法候选字段和输入键的唯一性。

## First Long Think v1 审查包

`first_long_think_v1/build_review_sample.py` 会在 strict ply 5–38 内寻找第一个用时严格大于此前本人非 pass 节点用时中位数 1.75 倍的节点。它不读取旧 Agent 或模型标签。生成的 `review_bundle_stratified30_plus2_no_anchor.json` 可按本文开头的步骤直接导入。
