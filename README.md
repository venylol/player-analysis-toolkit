# Player Analysis Toolkit

本目录提供可复用的 Othello Quest 选手调查工具。它不会启动或修改 EG
引擎，只读取已有的 OQ bundle 和 EG `game_*.json` 分析结果。

项目以 [GNU GPL v3](LICENSE) 发布。仓库只收录通用工具、方法文档、研究脚本和
可公开复现的技术材料；针对具体玩家生成的调查包、举报分组、异常结论和逐人报告
不会公开提交。公开数据集、Reference 与模型通过 GitHub Releases 分发。

所有文本输入输出均使用 UTF-8。脚本不会删除输入或既有结果文件；输出路径
如果需要覆盖，应由使用者先明确处理。

大型公开训练数据和模型不提交到 Git；版本化下载与校验方式见
[`DATASETS.md`](DATASETS.md)。

完整选手调查与玩家异常哨兵模式所依赖的数据、参照库、引擎和模型，需按
[`LONG_TERM_MAINTENANCE.md`](LONG_TERM_MAINTENANCE.md) 的清单持续检查和版本化更新。

## 文件

- `scripts/analysis/player_analysis.py`：数据汇总、子损、用时、个人开局库、留一法开局比较、
  外部参照组比较及统一运行入口。
- `src/player_analysis_toolkit/analysis_core.py`：棋盘回放、8种对称归一化和全部统计函数。
- `assets/standard_openings.json`：可手工维护的标准定式名称、别名和着手序列表。
- `scripts/analysis/detect_offbook.py`：在 Level22 审计完成后生成逐局算法脱谱标签。
- `scripts/analysis/offbook_analysis.py`：计算包含脱谱手在内的脱谱后统计。
- `scripts/reporting/render_github_pdf.py`：把 Markdown 渲染成 GitHub Issue 风格 HTML/PDF。
- `scripts/review/generate_review_checklist.py`：从既有调查 JSON 和固定模板生成审核信息速览表。
- `templates/review_checklist_template.md`：审核信息速览表的 Markdown 模板。
- `METHOD.md`：当前分析方法及各项结果的含义。
- `TOOLKIT_THEORETICAL_FOUNDATIONS.md`：完整理论基础、统计推断与等价 Elo 说明。
- `DATASETS.md`：Release 数据包、校验清单与隐私边界。

## 环境

- Python 3.11 或更高版本
- `numpy`
- `scikit-learn`
- Microsoft Edge（仅 PDF 生成需要）

## 1. 汇总 OQ 数据

```powershell
python scripts/analysis/player_analysis.py summary `
  --bundle "C:\path\account-bundle.json" `
  --account "player_id" `
  --output-dir "C:\path\output\summary"
```

输出：

- `summary.json`
- `events.json/csv`
- `games.json/csv`

## 2. 建立特定选手 Human Frequency Book

```powershell
python scripts/analysis/player_analysis.py build-book `
  --bundle "C:\path\account-bundle.json" `
  --account "player_id" `
  --color white `
  --exclude-game-id "reported_game_1" `
  --exclude-game-id "reported_game_2" `
  --max-ply 20 `
  --output "C:\path\output\player-white-book.json"
```

个人库的每个节点为：

```text
[board_string, occurrence_count, ply]
```

`board_string` 是当前行动方视角的完整棋盘，经过8种旋转、镜像归一化。
对手着手不会被算作被调查选手的选择，但会改变下一次选择前的完整棋盘。

## 3. 分析被报告局与留一法对照局的开局

```powershell
python scripts/analysis/player_analysis.py opening `
  --bundle "C:\path\account-bundle.json" `
  --account "player_id" `
  --reported-game-id "reported_game_1" `
  --reported-game-id "reported_game_2" `
  --max-ply 20 `
  --output "C:\path\output\opening-analysis.json"
```

被报告局不会进入个人库。检查每一盘对照局时，该局自身也会从个人库排除。

## 4. 查询或筛选标准开局定式

查询目标选手的指定对局；省略 `--game-id` 时查询该 bundle 中该选手的全部
对局：

```powershell
python scripts/analysis/player_analysis.py standard-opening `
  --bundle "C:\path\account-bundle.json" `
  --account "player_id" `
  --game-id "game_id" `
  --output "C:\path\output\standard-opening.json"
```

筛选所有包含指定定式的对局：

```powershell
python scripts/analysis/player_analysis.py standard-opening `
  --bundle "C:\path\account-bundle.json" `
  --account "player_id" `
  --opening "tanida" `
  --output "C:\path\output\tanida-games.json" `
  --csv-output "C:\path\output\tanida-games.csv"
```

`--opening` 可使用定式 `id`、名称、别名或坐标序列。坐标序列经过8种旋转、
镜像归一化后比较；只要定式序列是对局序列的前缀就算匹配。因此 Tanida 对局
也会被 `diagonal` 查询选中。每局的 `opening` 是最长、最具体的已登记定式，
`openingMatches` 保留全部上级定式。

维护新定式时，在 `standard_openings.json` 的 `openings` 数组增加：

```json
{
  "id": "unique-id",
  "name": "Opening Name",
  "sequence": "f5f6e6d6",
  "aliases": ["optional alias"]
}
```

脚本会拒绝非法着手序列、重复名称/别名，以及经过对称变换后重复的序列。若
输入的是目录中尚未登记的合法坐标序列，仍可临时筛选，但不会获得定式名称。

## 5. 算法脱谱登记与脱谱后分析

正式调查在 Level22 完成并通过全量 audit 后，立即运行固定规则脱谱算法。每局
都会在 `offbook_records.json` 中获得 `algorithmLabel`：找到锚点为 `offbook`，
未找到为 `no_offbook`，不再生成审查包或等待 Agent 手工标注。

固定规则只分析目标棋手从 ply 5 开始的实着，不设最大 ply。每局按规定时限计算
动态用时阈值：`5500ms × ln(1 + 时限秒数) ÷ ln(301)`。Level22 落子前局面的
`abs(bestEval) <= 6` 是时间锚点的可搜索范围；首次 `abs(bestEval) > 6` 的目标方
节点形成估值截断点，等于正负6仍可参与时间搜索。若截断点之前存在用时严格超过
动态阈值的目标方实着，依次检查这些时间候选；否则检查估值截断点。每个候选都
查看其后最多4个目标方实着，快棋阈值为
`2000ms × ln(1 + 时限秒数) ÷ ln(301)`；若出现连续3步用时小于等于快棋阈值，
该候选被否决。后续不足3个目标方实着时保留候选并登记证据不足。第一个通过检查
的候选作为脱谱锚点；所有候选均被否决或根本没有候选时标记 `no_offbook`。

单独对已有 Level22 目录生成标签：

```powershell
python scripts/analysis/detect_offbook.py `
  --engine-directory "C:\path\engine_level22" `
  --bundle "C:\path\selected_account_bundle.json" `
  --account "sample_player" `
  --output "C:\path\offbook_records.json"
```

### 5.1 脱谱后指标及排行榜参照比较

统计配置示例：

```json
{
  "target": {
    "name": "sample_player",
    "account": "sample_player",
    "marks": "C:\\path\\sample_player-offbook-records.json",
    "engineDirectory": "C:\\path\\sample_player-existing-eg-json"
  },
  "references": [
    {
      "name": "leaderboard-page-5",
      "members": [
        {
          "name": "leader-1",
          "account": "leader_1",
          "marks": "C:\\path\\leader-1-offbook-records.json",
          "engineDirectory": "C:\\path\\leader-existing-eg-json"
        }
      ]
    }
  ]
}
```

如果只统计某个已校验记录文件中的指定对局，可在对应 target 或 reference
member 中加入 `"gameIds": ["game_1", "game_2"]`。脚本会拒绝空数组、重复
gameId 或不属于该脱谱记录文件的 gameId。输出的 `postOffBookInclusive.loss.games`
会为每盘同时保留 `offBookPly`、`postOffBookStartsAtPly` 和
`excludedPreOffBookMoveCount`，明确证明该统计已经排除锚点之前的目标选手着手。

已经有合并记录文件时，reference group 可用
`membersFromConsolidated` 自动展开成员，避免手工重复列出每个账号：

```json
{
  "name": "diagonal-reference",
  "membersFromConsolidated": {
    "path": "C:\\path\\40-offbook-records-consolidated.json",
    "opening": "Diagonal Opening",
    "engineDirectory": "C:\\path\\diagonal-eg-json"
  }
}
```

`membersFromConsolidated` 也可为对象数组，用于把多个开局及各自 EG 目录合并成
一个参照组。脚本只接受合并文件中已经关联到原始已校验记录文件的记录。

比较输出同时给出整局和脱谱后差值。如果 target 只包含一盘，脚本还会分别在
`individualFullGameEmpiricalPosition` 和
`individualPostOffBookEmpiricalPosition` 中列出该盘相对参照逐局平均子损的
小于、等于、含等号经验位置及参照范围。

```powershell
python scripts/analysis/offbook_analysis.py offbook-stats `
  --config "C:\path\offbook-stats-config.json" `
  --output "C:\path\offbook-stats.json"
```

同一目标账号的被报告局与个人对照还可以在算法锚点切分后重新计算差值、整局
聚类 bootstrap、精确组合位置和两部分模型：

```json
{
  "dataset": {
    "account": "sample_player",
    "marks": "C:\\path\\sample_player-offbook-records.json",
    "engineDirectory": "C:\\path\\sample_player-existing-eg-json",
    "gameIds": ["reported_1", "reported_2", "control_1", "control_2"]
  },
  "reportedGameIds": ["reported_1", "reported_2"],
  "bootstrap": 10000,
  "modelBootstrap": 1000,
  "seed": 20260801
}
```

```powershell
python scripts/analysis/offbook_analysis.py offbook-model `
  --config "C:\path\offbook-model-config.json" `
  --output "C:\path\offbook-model.json"
```

输出的 `fullGame` 与 `postOffBookInclusive` 使用完全相同的比较和模型字段；后一
单元只读取各盘脱谱锚点及其后的目标选手着手，并包含锚点手本身。

`offbook-model` 还在不改变以上两个旧单元的前提下新增固定四阶段输出。配置仍
使用上例的 `bootstrap` 和 `seed`；阶段 bootstrap 使用 `seed + 200`，无需也不
允许从当前样本配置或重新估计边界/权重：

```text
postOffBookByPhase
  phaseDefinitions
  bootstrapPolicy
  ply1To30
  ply31To47
  ply48To53
  ply54To60
postOffBookPhaseCombined
  primaryWeightScheme
  combinationOrder
  missingPhasePolicy
  weightSchemes
    referenceHumanExposure
    equalPhase
  weightSources
  coverageDiagnostics
plyCoordinateAudit
```

固定闭区间为 `1–30 / 31–47 / 48–53 / 54–60`，等价半开区间为
`[1,31) / [31,48) / [48,54) / [54,61)`。每局先执行算法脱谱硬切分，再按全局
实际落子 ply 放入阶段；例如 `offBookPly=35` 的对局不会给前一阶段或 ply 31–34
提供节点。每个阶段的 `reported` 和 `control` 分别给出原组局数、目标节点数、
有效子损节点数、实际贡献局数/独立局数，以及局等权平均子损、着手等权平均
子损、零子损率、正子损平均值、子损≥4比例和子损≥10比例。两项阈值还分别
给出节点数；`reportedMinusControl` 为六项
举报组减对照组点估计及整局聚类区间。`targetNodeCount` 包括该阶段所有目标
选手实际落子节点，`lossNodeCount` 只包括具有 `lossClipped` 的统计节点。子损
统一定义为 `disc_loss = max(0, raw_loss)`；两个阈值都使用包含边界的 `>=`。
≥4 保持主要敏感指标，紧邻的 ≥10 表示更稀少的“大失误”指标；≥10 区间即使
更宽，也不因此获得更强的证据权重。

每次阶段 bootstrap 在举报组和对照组内分别抽取与原组相同数量的整局。同一
次抽到的两组对局名单同时供四阶段、≥4、≥10 和全部其他指标使用；一局跨阶段时仍只有一个
聚类身份。阶段合成也在该次重复内部先完成，再对合成重复值取 2.5% 和 97.5%
分位，绝不平均四个阶段区间端点。

合成结果同时输出两种预先固定的命名口径：

- 主口径 `referenceHumanExposure`：来自归档独立人类参考样本
  `investigations/oq_loss_phase_boundaries_20260803/ply_statistics_postbook.csv`
  的有效子损节点数 `48631 / 51032 / 18040 / 17921`，总计 `135624`；固定权重
  约为 `0.358572229104 / 0.376275585442 / 0.133014805639 /
  0.132137379815`。
- 敏感性口径 `equalPhase`：四阶段各 `0.25`，强调阶段同等重要，因此会相对
  提高两个较短尾盘阶段的影响。

权重不从当前举报/对照样本估计，也不会因某次重复缺阶段而重新归一化。某阶段
在任一组完全无数据时，对应点估计/阶段结果允许为带原因的 `null`；某次重复
缺少某固定阶段或某指标时，该指标的合成重复记为失败，不重抽。只有成功率至少
达到固定的 `0.95` 才发布该项 95% 区间，否则区间为带 `nullReason` 的 `null`。
成功/失败次数和阶段覆盖在输出中显式保留。

`plyCoordinateAudit` 强制检查 EG `ply` 是范围 1–60 的、剔除显式 pass 后的全局
实际落子序号；人工 `offBookPly` 必须恰好对应目标选手 EG 节点。pass 只会造成
`sourceMoveIndex` 跳号，不占用 `ply`。不符合该语义、非坐标节点或超过 60 的
输入会明确失败，不做静默转换。

边界 `30|47|53` 是用户选择的固定细分方案，用于提高解释区分度，不是稳定识别
出的普适自然边界。归档调查中它相对三阶段的样本内拟合改善 28.3%，但完整
bootstrap 复现率只有 38.4%，后两个边界还会随最小阶段宽度改变；不得在每份
数据或每次重复中重估。ply 31/32 附近还存在 Egaroucid 搜索制度变化，阶段差异
不能全部解释为人类行为的自然转折。

阶段测试可复现运行：

```powershell
python -m unittest discover -s tests -v
```

当被报告对局分别属于不同开局或其他不同层时，可用 `offbook-stratified` 从每个
参照层各取一盘，枚举笛卡尔积组合。输出同样同时包含整局和脱谱后组合位置，
避免把不同层的任意同层组合混入对照。

结果同时保留整局和 `postOffBookInclusive` 单元。脱谱后单元从 Agent 标记的
目标选手着手开始，包含该手，计算局等权/着手等权子损和原始毫秒 mean 用时，
并输出目标选手减排行榜参照组的差值。标为 `no_offbook` 的对局保留在整局统计，
但不进入脱谱后单元。

## 6. 子损分析

```powershell
python scripts/analysis/player_analysis.py loss `
  --engine-dir "C:\path\ega-analysis" `
  --account "player_id" `
  --reported-game-id "reported_game_1" `
  --reported-game-id "reported_game_2" `
  --bootstrap 10000 `
  --model-bootstrap 1000 `
  --wld-from-ply 39 `
  --output "C:\path\output\loss-analysis.json"
```

脚本读取既有 EG `game_*.json`，不会重跑引擎。结果包含按局、按着手、整局
聚类 bootstrap、精确组合位置和单步两部分模型。每局、举报组、对照组、差值和
脱谱后输出都将 `lossAtLeast10Count` / `lossAtLeast10Rate` 放在原有
`lossAtLeast4Count` / `lossAtLeast4Rate` 旁边；≥4 与 ≥10 的区间复用同一次
整局抽样。

节点可选携带 `probability_loss_ge4` 和 `probability_loss_ge10`。只有对应阈值的
每个有效子损节点都有合法 `[0,1]` 概率时，JSON/报告才输出预计节点数、实际减
预计节点数，以及按节点或整局归一的实际发生减预测概率；否则明确输出
“模型概率不可用”，不会用 0、空字符串或虚构值代替。

`--wld-from-ply 39` 是可选项；省略时输出与原有 schema 和字段完全一致。启用后，
以排除 pass 的全局实际落子序号为边界，包含第 39 手，只新增每局及当前汇总棋手的
`engine_wld_loss_total_from_ply39`。不会新增逐着 WLD、变化前后等级、平均值或
逆转次数。Egaroucid 的 book、level、threads、hash 等现有参数保持不变。

直接运行正常引擎包装器时，也可在原命令后加 `--wld-from-ply 39`。引擎完成后会
在新的输出目录写入 `engine_wld_loss_totals_from_ply39.json`、按局/棋手 CSV 和
按棋手 CSV；参数不会传给 Egaroucid，也不会改变其搜索配置。

模型预测 CSV/JSON 可用通用消费入口汇总。启用 WLD 时，输入必须包含
`expected_wld_loss`、`wld_applicable`、`global_placement_ply`、`game_id`，并至少
包含 `player_id` 或 `side`；缺字段会直接报错：

```powershell
python scripts/analysis/player_analysis.py model-wld `
  --predictions "C:\path\node-predictions.csv" `
  --wld-from-ply 39 `
  --output "C:\path\predicted-wld.json" `
  --csv-output "C:\path\predicted-wld-by-game-player.csv" `
  --player-csv-output "C:\path\predicted-wld-by-player.csv" `
  --markdown-output "C:\path\predicted-wld.md"
```

该入口只对 `wld_applicable=true` 且 `global_placement_ply >= 39` 的行求和，字段为
`predicted_expected_wld_loss_total_from_ply39`。生成的 Markdown 可继续交给第 10
节的 PDF 渲染命令。

阶段调查脚本的诊断图默认同时包含 `loss_ge4_rate` 与 `loss_ge10_rate`，也可
重复使用 `--plot-metric loss_ge4_rate --plot-metric loss_ge10_rate` 选择要绘制
的指标。该选项只控制图中展示；固定输出统计仍同时保留两个阈值，边界拟合特征
继续沿用归档方案，不因新增 ≥10 描述指标而改变。

## 7. 用时趋势

```powershell
python scripts/analysis/player_analysis.py time `
  --engine-dir "C:\path\ega-analysis" `
  --account "player_id" `
  --reported-game-id "reported_game_1" `
  --reported-game-id "reported_game_2" `
  --output "C:\path\output\time-analysis.json"
```

基线使用同色对照局的原始毫秒用时，以算术平均为中心拟合 ply 曲线。长考阈值
为相同 ply 对照用时的第90百分位。

## 8. 外部参照组

参照组通过 JSON 配置。排行榜 bundle 的示例：

```json
{
  "target": {
    "engineDirectory": "C:\\path\\target-ega",
    "account": "player_id",
    "gameIds": ["reported_game_1", "reported_game_2"]
  },
  "groups": [
    {
      "name": "page5",
      "engineDirectory": "C:\\path\\reference-ega",
      "bundle": "C:\\path\\reference-bundle.json",
      "where": {"page": 5},
      "accountField": "leaderboardAccount"
    }
  ]
}
```

运行：

```powershell
python scripts/analysis/player_analysis.py reference --config reference-config.json --output reference-analysis.json
```

## 9. 一次运行主要分析

配置：

```json
{
  "bundle": "C:\\path\\account-bundle.json",
  "engineDirectory": "C:\\path\\ega-analysis",
  "account": "player_id",
  "reportedGameIds": ["reported_game_1", "reported_game_2"],
  "outputDirectory": "C:\\path\\analysis-output",
  "maxPly": 20,
  "bootstrap": 10000,
  "modelBootstrap": 1000,
  "seed": 20260801,
  "wldFromPly": 39,
  "referenceConfig": "C:\\path\\reference-config.json"
}
```

```powershell
python scripts/analysis/player_analysis.py run-all --config run-config.json
```

`referenceConfig` 和 `wldFromPly` 都可以省略。`reference`、`offbook-stats`、
`offbook-model` 的配置也可使用 `"wldFromPly": 39`；后两者还接受同名 CLI
`--wld-from-ply 39` 覆盖/启用该选项。

全量 hint 阶段调查可使用：

```powershell
python scripts/analysis/analyze_oq_loss_phase_boundaries.py `
  --wld-from-ply 39 `
  --output-dir "C:\path\new-investigation-output"
```

该脚本使用已有 level-18 hint-6 分数和原有 book 状态，不改变阶段拟合或搜索参数；
只额外生成按局/棋手与按棋手的 WLD 总和 CSV/JSON。

## 10. 生成 GitHub 风格 PDF

```powershell
python scripts/reporting/render_github_pdf.py `
  --markdown "C:\path\report.md" `
  --pdf "C:\path\report-github.pdf" `
  --title "对局调查报告" `
  --subtitle "Othello Quest 对局分析" `
  --reporter "瑞瑞（使用 AI 协助）" `
  --role "无差别网赛裁判" `
  --report-date "2026年8月2日"
```

附录图片使用 JSON manifest：

```json
{
  "heading": "附录：截图依据",
  "intro": "以下截图作为报告的图像依据。",
  "images": [
    {"path": "C:\\path\\image1.jpg", "caption": "图 A-1 说明"},
    {"path": "C:\\path\\image2.jpg", "caption": "图 A-2 说明"}
  ]
}
```

将 manifest 路径传给 `--appendix-manifest`。脚本同时保存 HTML，默认与 PDF
同名；可以用 `--html-output` 指定位置。

## 11. 生成审核信息速览表

该流程只读取已有 OQ bundle、EG 汇总/单局 JSON、算法脱谱记录和高分参照统计，
不会抓取网络数据或重跑引擎。OQ 公开对局需要更新时，先复用
`scripts/data/oq_account_bundle.py` 生成新的 account bundle，再把该文件写入配置。

```powershell
python scripts/review/generate_review_checklist.py --config "C:\path\checklist-config.json"
```

配置示例：

```json
{
  "schema": "review-checklist-config-v1",
  "playerDisplayName": "Player Name",
  "account": "oq_account",
  "reportedGameIds": ["reported_game_1", "reported_game_2"],
  "accountBundle": "..\\source-investigation\\account-bundle.json",
  "engineSummary": "..\\source-investigation\\ega-analysis\\summary.json",
  "sameColorModel": "..\\report-investigation\\same-color-model.json",
  "allControlModel": "..\\report-investigation\\all-control-model.json",
  "comparisonStats": "..\\report-investigation\\highscore-reference.json",
  "comparisonReference": "reference-name-when-the-file-has-more-than-one",
  "profileImage": "C:\\path\\profile.jpg",
  "openingImage": "C:\\path\\oq-opening-ranking.jpg",
  "template": "..\\..\\review_checklist_template.md",
  "outputMarkdown": "player-review-checklist.md",
  "outputData": "player-review-checklist-data.json"
}
```

相对路径一律相对于配置文件所在目录解析。`comparisonReference` 仅在
`comparisonStats.references` 有多组时必填；`outputData` 可省略。脚本把资料截图
和 OQ 开局排行截图复制到 Markdown 同目录的 `assets` 子目录，并使用相对路径
嵌入。开局排行完全来自 OQ 截图；checklist 不读取 Human Frequency Book，
也不计算“最常用开局”。原图不会被修改。

`engineSummary` 所在目录必须保留对应的 `game_*.json`。脚本复用这些已有 EG
单局结果和 `src/player_analysis_toolkit/analysis_core.py` 的既有用时基线函数，生成举报局逐局用时以及举报局
组合与同色个人非举报对照的汇总对比。相对基线平均残差使用同色非举报局拟合的
固定四节点三次回归样条；长考阈值为相同 ply 对照用时的第 90 百分位。脚本不
重跑 EG 引擎。

Rating 表只使用目标选手具有有效 `averageLoss`、有效 `totalLoss` 且
`nodeCount > 0` 的 EG 整局结果，并强制排除 `reportedGameIds`。对手 Rating 取
对应 OQ detail 的赛前 `players[*].oldR`。分组边界固定为：

- `<1200`
- `[1200,1500)`
- `[1500,1700)`
- `[1700,2000)`
- `[2000,2200)`
- `>=2200`

Rating 表按组输出样本局数、组内总子损、每盘整局总子损的平均数/中位数，
以及每盘整局平均子损的平均数/中位数。其他子损表同时保留总子损、局等权平均
和着手等权平均等口径。

输入 schema、账号、举报局、对手 `oldR`、模型控制组排除关系或模板占位符有误
时，脚本会明确报错。所有文本输出为 UTF-8 无 BOM；任一目标 Markdown、数据
JSON 或目标图片已经存在时，脚本拒绝覆盖。

## 12. 主 Agent 驱动的完整调查流水线

`run_player_investigation.py` 串联棋谱、Player 画像、Level22、hint1/hint6、
脱谱登记、TCN 数据装配、12 成员个人适配、举报局推理、整局 bootstrap 和经验
思考时间分析。流水线只生成 JSON/CSV/NPZ 等机器产物，最终报告为
`report.json`，不生成 Markdown。

主 Agent 首次执行或处理断点恢复时，建议先阅读
[`docs/PLAYER_INVESTIGATION_AGENT_RUNBOOK.md`](docs/PLAYER_INVESTIGATION_AGENT_RUNBOOK.md)。

先按账号创建调查并拉取所有当前可查询棋谱：

```powershell
python scripts/analysis/run_player_investigation.py start `
  --account example_player `
  --output-dir "C:\path\example-player-investigation"
```

命令完成后，`progress.json` 状态为 `awaiting_group_selection`，游戏清单位于
`game_catalog.json`。主 Agent 可以明确列出两组 game ID：

```powershell
python scripts/analysis/run_player_investigation.py select-groups `
  --run-dir "C:\path\example-player-investigation" `
  --reported-game-id game_a `
  --reported-game-id game_b `
  --control-game-id game_c `
  --control-game-id game_d
```

也可以用 `created` 时间的闭区间选择举报组；区间外所有当前可查询对局自动成为
对照组。时间必须带 `Z` 或明确 UTC 偏移：

```powershell
python scripts/analysis/run_player_investigation.py select-groups `
  --run-dir "C:\path\example-player-investigation" `
  --reported-from "2026-08-08T21:40:00+08:00" `
  --reported-to "2026-08-08T23:10:00+08:00"
```

脚本随后依次执行 Player 画像和 Level22。Level22 完成并通过 audit 后，
`offbook_detection` 立即为全部选中对局生成 `offbook_records.json`；每局都带有
`offbook` 或 `no_offbook` 算法标签、锚点来源和计算证据。流程不等待人工脱谱审核，
并继续运行 hint、模型适配、统计与最终报告。

中断后使用 `resume`；查看顶层阶段状态使用 `status`。Level22、hint 和12模型适配
还会在 `progress.json` 的 `nestedProgress` 字段中暴露原阶段进度文件。已完成
阶段只有在命令摘要和输出 SHA-256 均一致时才会跳过。举报局若没有可信脱谱锚点，
不会被强行赋予虚假锚点；该局仍进入常规子损/WLD/时间统计，并使用目标选手
整局着手参加模型评估。模型报告会列入 `fullGameFallbackNoOffbookGameIds`。

hint 数据走 `safe_recompute_egaroucid_hints.py` 的原子事务入口，不使用 legacy
并行入口。个人调查的 Level22 固定采用12个独立 worker、每个 Console 16线程、
hash25；Tournament 直接调用同一 runner 时默认只启用2个 worker。个人调查默认
hint1 为12个 worker，hint6 为12个独占 worker；hint1 固定1线程且无 book，hint6
固定 Level18、每引擎16线程并启用 book，两者 hash level 均固定为25。hint6 使用
batch 128、timeout 900、最多2次尝试，并在完成后执行全量 audit。每个实着节点保留
请求棋盘、setboard 回显、hint 回显、引擎哈希、request/worker/batch ID；Level22
逐局原子写入并执行完整棋谱/引擎契约 audit，两类审计通过后才能进入后续阶段。

## 13. 玩家异常哨兵模式 V1

哨兵模式从冻结的双方 Elo 二维 Reference 自动筛查最近最多30局有坐标落子的棋局，不需要预先提供
举报局。默认配置为仓库顶层 `sentinel_reference_config.json`；配置只用相对路径引用
冻结 Reference 和派生缓存，不复制或重跑其中1,495局 Level22。

```powershell
python scripts/analysis/run_player_investigation.py start-sentinel `
  --account example_player `
  --output-dir "C:\path\example-player-sentinel" `
  --reference-config sentinel_reference_config.json
```

`status` 和 `resume` 与普通调查共用。哨兵先剔除零坐标落子棋局（例如开局即掉线，
不占30局名额），再自动选取最近最多30局并运行固定参数
Level22和确定性脱谱算法，然后按目标方精确 oldR、对手精确 oldR、颜色及算法 scope
进行二维 Reference 插值。调查 gameId 若已在 Reference 中，会在全部格、bootstrap池
和伪玩家池中全局剔除其黑白两个方向。

主指标是每局 `loss_ge4_rate`：`offbook` 从算法锚点手开始并包含该手；
`no_offbook` 使用目标玩家整局实着。扫描只检查最强单局、全部对局，以及按外部
强度残差稳定排序后的前缀 k，不枚举任意组合。默认10,000名伪玩家执行相同完整
k扫描并校正选择偏差；固定候选再以10,000次整局 bootstrap 给出效应区间。
WLD 固定包含 `global_placement_ply >= 39`，只评价 GE4 冻结组，不能另选举报局。

只有 `concentrated_external_internal_anomaly` 和 `isolated_external_anomaly` 会产生
非空 `reportedGameIds`。Reference门槛未通过时不运行个人模型；模型对照少于8局时
也跳过个人适配。模型结果仅标记为支持、未支持或冲突，不能修改
`selection_manifest.json`，也不输出作弊概率。

主要输出为 `per_game_reference_scores.json/csv`、`pseudo_scan_replicates.csv`、
`pseudo_scan_summary.json`、`sentinel_scan_results.json`、`selection_manifest.json`、
`model_review_groups.json` 和 `report.json`。

Reference 派生缓存可复现构建，但目标目录必须为空：

```powershell
python scripts/analysis/sentinel_analysis.py build-reference `
  --reference-dir research/offbook_detection/data/oq_elo_matchup30_reference_level22_1600plus_20260814 `
  --output-dir research/offbook_detection/data/oq_sentinel_reference_level22_1600plus_v6_20260819
```

该命令只读取既有 Level22 JSON，并直接调用 `detect_offbook.py` 的确定性算法。
当前冻结缓存的正式 `no_offbook` 池为黑方1条、白方0条，不足以执行强制留一局
伪玩家校准；相关调查局会保留整局算法记录并显式标为 `not_calibratable`，不会跨
颜色或 scope 借用分布。

### 哨兵模式行棋质量等价 Elo（v1）

这是与上述异常扫描并存的独立功能，不改变旧 sentinel 的 `scan`、`freeze`、Reference
或输出语义。实现规格、冻结边界和验收记录见
[`docs/SENTINEL_ESTIMATED_ELO_IMPLEMENTATION_SPEC.md`](docs/SENTINEL_ESTIMATED_ELO_IMPLEMENTATION_SPEC.md)
和 [`docs/SENTINEL_ESTIMATED_ELO_ACCEPTANCE_REPORT_20260819.md`](docs/SENTINEL_ESTIMATED_ELO_ACCEPTANCE_REPORT_20260819.md)。

固定四阶段为 global placement ply 1–30、31–47、48–53、54–60；每局四阶段等权，
再按整局对参考库做距离加权标准化，最后对最近最多30局做局等权整数网格搜索。
该功能不使用机器学习，目标账号自己的显示 Elo 不进入估计。

构建冻结 Level22 的双 scope 参考派生库（目录必须为空）：

```powershell
python scripts/analysis/sentinel_elo_analysis.py build-elo-reference
```

构建 leave-one-account-out 校准并验证数据库校准 95% 范围：

```powershell
python scripts/analysis/sentinel_elo_analysis.py calibrate-elo
```

单选手调查的统一分析入口是
`scripts/analysis/sentinel_unified_analysis.py`。`run_player_investigation.py
start-sentinel` 会在旧 sentinel 的 Reference scoring、pseudo scan 和 freeze 完成后自动调用
它，将旧 sentinel 摘要与本次预估 Elo 合并到 `sentinel_unified_analysis.json`，再写入最终
`report.json` 的 `estimatedElo` 和 `unifiedSentinelAnalysis` 字段。直接调用
`sentinel_elo_analysis.py estimate-elo` 只适合单独调试或复算 Elo，不是完整调查入口。

统一脚本同时保留旧 sentinel 和 Elo 的兼容命令族；以后新增与单选手调查有关的指标，应扩展
该统一脚本，不应再新增平行的单选手分析入口。统一调查流程已通过该脚本调用旧 sentinel
的 acquire/score/scan/freeze 兼容命令，因此旧脚本可作为兼容接口保留，不再是统一调查的
主入口。具体仓库约束见 [`AGENTS.md`](AGENTS.md)。

对一个已有目标账号 bundle、Level22 目录和脱谱记录运行估计：

```powershell
python scripts/analysis/sentinel_elo_analysis.py estimate-elo `
  --account example_player `
  --bundle "C:\path\selected_account_bundle.json" `
  --engine-dir "C:\path\engine_level22" `
  --offbook-records "C:\path\offbook_records.json" `
  --output-dir "C:\path\estimated-elo"
```

输出包括 `estimated_elo.json`、`estimated_elo_curve.csv`、`estimated_elo_games.csv` 和
`estimated_elo_phase_diagnostics.csv`。结果名称固定为“所选最近有效对局的平均行棋质量
等价 Elo”；数据库校准范围不是理论正态置信区间。

#### 校准缓存与数据库扩展

`calibrate-elo` 是一次性的数据库级校准步骤，结果缓存为参考目录中的
`elo_calibration.json`，并同时保存 `elo_calibration_cases.jsonl` 作为校准审计明细。
`estimate-elo` 只读取已有校准缓存，不会在每次目标账号估计时重新计算。

当正式参考数据库扩展时，不能继续沿用旧校准缓存。应使用新的版本化输出目录依次：

1. 重新运行 `build-elo-reference`，生成包含新增对局的 directed phase records、构建审计和
   `reference_sha256_manifest.json`；
2. 对新的参考目录运行 `calibrate-elo`，重新计算满足至少 10 局的用户集合、
   calibration/validation 划分、`T95`、曲线诊断阈值和校准案例；
3. 让后续 `estimate-elo` 指向新的参考目录。已有目标账号如需使用扩展后的数据库，也应
   重新运行估计，以纳入新的 Reference 分布和新的 `K`；
4. 不要把新 directed records 与旧 `elo_calibration.json` 混用。该流程目前不自动检测数据库
   扩展，也不做增量校准。

数据库扩展只影响这套独立的 estimated-Elo 参考/校准流水线，不要求重跑旧 sentinel 的
`scan`、`freeze` 或旧输出流程。

### estimated-Elo v2（实施候选，尚未切换正式配置）

v2 直接使用每阶段的 GE4 次数 `x` 和有效着手数 `n`，对精确 K 近邻执行加权
Beta-Binomial 最大似然拟合。每盘累加四阶段精确计数的负对数概率，再对所选目标棋取平均，
以 1600–2500 整数网格上最小的 `J(E)` 作为点估计。有效着手较多的阶段会自然提供更多
概率证据；这有意替代 v1 的“四阶段比例先等权平均”语义。

新配置候选是 `sentinel_elo_reference_config_v2_20260828.json`。它使用独立 schema 和新目录，
不会覆盖 v8 Reference 或旧 calibration。当前顶层 `sentinel_elo_reference_config.json` 仍是
正式 v1 指针；只有完整 calibration 和独立 validation 通过后才能切换。

从现有 Level22 派生记录准备统一冻结的条件 reference 缓存：

```powershell
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v2_20260828.json `
  prepare-conditional-elo-reference `
  --output-dir research/offbook_detection/data/oq_sentinel_elo_conditional_reference_v1_20260828
```

中断后在相同命令末尾增加 `--resume`。程序验证 config/input SHA，并复用已原子提交的账号
分片。缓存按颜色和 `full_game/post_offbook_inclusive` 保存双方 Elo 标准差、z1/z2/z3
标准差、逐阶段 K/权重/拟合状态及 artifact SHA。

正式 calibration 命令固定 16 个独立进程、每 task 一个账号、`chunksize=1`、每 worker
内部近邻查询线程为 1：

```powershell
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v2_20260828.json `
  calibrate-elo `
  --output-dir research/offbook_detection/data/oq_sentinel_elo_calibration_v2_20260828
```

这一步生成新的 v2 T95 和独立 validation 结果，不能复用 v1 `elo_calibration.json`。若
validation 未确认 95% 覆盖，状态固定为 `calibration_unavailable`。

每个 calibration 账号任务还记录 A/B/C 对照：A 是 v1 四阶段等权 signed-z，B 是
`previousZWeight=0`，C 是正式候选 `previousZWeight=1`。只有 calibration 账号参与 B/C 的
方法选择；validation 只评估冻结的 C，避免用 validation 调参。

v2 使用统一冻结的 reference z 缓存。估算目标账号时仍会删除该账号作为任一方参与的全部
source games 及其两条 directed records，并重新计算 N、K、K+1 边界和三角权重；但不得把
该设计描述为对每个未来目标账号完全重建全部派生 z 的逐账号交叉拟合。`knownElo` 只用于
估算后的误差和 T95。

统一缓存的间接影响用可恢复的精确重建命令审计。它先删除指定账号参与的全部 source
games，再重新准备尺度和 reference z；`--rebuild-dir` 与 `--output-dir` 都必须使用新目录：

```powershell
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v2_20260828.json `
  audit-reference-z-sensitivity --account <account> `
  --bundle <selected_account_bundle.json> --engine-dir <engine_level22> `
  --offbook-records <offbook_records.json> `
  --rebuild-dir <versioned-account-rebuild-dir> --output-dir <new-audit-dir> [--resume]
```

报告比较点估计与最佳点近邻集合 SHA；只有存在与统一 conditional manifest 匹配的已验证
v2 calibration 时才显示统一缓存的正式区间。重建缓存因 manifest 不同，不借用统一缓存的
正式 calibration，避免把反事实敏感性结果伪装为已校准区间。

调试估算仍使用 `estimate-elo`，正式单玩家调查仍由 `sentinel_unified_analysis.py` 暴露并由
`run_player_investigation.py` 编排。v2 曲线 CSV 使用 `meanNegativeLogLikelihood`、
`meanConditionalZ`、有效目标棋数和失败摘要；阶段诊断保存 `x/n`、`PExact`、target z、K、
边界距离、拟合 `m/kappa`、scope 和颜色。

#### estimated-Elo v3 calibration（2026-08-29）

v3 使用独立的 `sentinel_elo_reference_config_v3_20260829.json`、adaptive
multi-basin Elo 搜索和 global cKDTree；不会读取或覆盖 v1/v2 calibration。正式命令为：

```powershell
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v3_20260829.json `
  calibrate-elo
```

本次已完成 918 个 calibration 账号和 178 个 validation 账号，严格使用 16 进程、每 task
一个账号、`chunksize=1`、每 worker 一个查询线程。由于 1,096 条曲线均报告显式
`beta_binomial_fit_failed`，无法形成 frozen T95；artifact 状态为
`calibration_unavailable`，顶层主配置继续保持 v1。输出位于
`research/offbook_detection/data/oq_sentinel_elo_calibration_v3_20260829/`，覆盖率审计位于
`research/offbook_detection/data/oq_sentinel_elo_coverage_audit_v3_20260829/`。完整 SHA、分组误差、
运行时间和已知限制见
[`docs/SENTINEL_ESTIMATED_ELO_V2_ACCEPTANCE_REPORT_20260828.md`](docs/SENTINEL_ESTIMATED_ELO_V2_ACCEPTANCE_REPORT_20260828.md)。

#### estimated-Elo v4（当前正式算法，2026-08-29）

当前正式 estimated-Elo 已切换为独立的 v4 合同：
`estimated-elo-v4-anscombe-local-gaussian-adaptive-grid-global-knn-v1`。它使用固定
Anscombe `y/v`、过滤后 `N_allowed` 的 `ceil(N_allowed^(2/3))`、局部 KNN 三角权重、
相邻阶段 reference z 条件距离，以及闭式局部高斯预测密度。正式阶段字段是
`transformedY`、`samplingVariance`、`localMeanY`、`betweenGameVariance`、
`meanEstimateVariance`、`predictiveVariance`、`targetZ` 和
`negativeLogPredictiveDensity`；不再把分数描述为离散精确计数概率。

v4 使用独立的 `sentinel_elo_reference_config_v4_matchup600_20260911.json`、派生 Reference、单份
reference-z 缓存和 calibration 目录，不覆盖 v1/v2/v3 产物；旧版本只为历史读取和复现实验
保留。准备缓存时复用已有 Level22 的 `x/n`，不重新运行 Level22：

```powershell
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v4_matchup600_20260911.json `
  build-elo-reference
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v4_matchup600_20260911.json `
  prepare-conditional-elo-reference
```

测试和小样本验证通过后，正式 calibration 使用 16 个 worker、每任务一个账号、
`chunksize=1`、worker 内部查询线程 1，并以 calibration 账号单独冻结 T95，再评估 validation：

```powershell
python scripts/analysis/sentinel_elo_analysis.py `
  --config sentinel_elo_reference_config_v4_matchup600_20260911.json `
  calibrate-elo
```

单玩家正式入口仍是 `scripts/analysis/sentinel_unified_analysis.py`，生命周期由
`scripts/analysis/run_player_investigation.py` 编排。完整数学、字段、恢复和哈希合同见
[`docs/SENTINEL_ESTIMATED_ELO_V4_IMPLEMENTATION.md`](docs/SENTINEL_ESTIMATED_ELO_V4_IMPLEMENTATION.md)。
