# 调查与哨兵模式长期维护清单

本文只登记以下两条正式运行链路长期依赖的**信息、参照数据和模型资产**：

- `scripts/analysis/run_player_investigation.py start`：完整选手调查；
- `scripts/analysis/run_player_investigation.py start-sentinel`：玩家异常哨兵模式。

历史调查目录、单次运行输入/输出、实验模型、归档脚本和缓存不属于本清单。本文于
2026-08-14 按仓库当前实现核对；“当前”只表示代码指向，不表示外部数据仍然新鲜。

## 维护优先级

| 级别 | 含义 | 检查时机 |
| --- | --- | --- |
| P0 | 会直接改变正式结论或导致流水线不能运行 | 每次正式调查前；相关上游变化后立即检查 |
| P1 | 会逐渐失去代表性，但短期内通常仍可运行 | 每季度检查；发现分布漂移时提前检查 |
| P2 | 复现、解释和发布所需的配套信息 | 每次替换 P0/P1 资产时同步检查 |

## P0：正式运行入口和判定基准

| 维护对象 | 适用链路 | 当前入口或资产 | 何时必须更新 | 更新时必须同步 |
| --- | --- | --- | --- | --- |
| 主模型指针 | 完整调查；哨兵的条件模型阶段 | `research/tcn_loss_model/PRIMARY_MODEL.json` | 正式主模型换代、输入策略或模型 schema 改变 | 新模型目录、ensemble manifest 及其 SHA-256、基础 checkpoint、选择理由和 `supersedes` |
| 12 成员主 ensemble | 完整调查；哨兵的条件模型阶段 | `research/tcn_loss_model/models/primary_wld_ensemble12/ensemble_manifest.json` | 新训练集或新模型在冻结验证集上通过正式验收 | 每个成员 checkpoint、文件大小和 SHA-256；保持 manifest 为 `completed`，或同步修改运行校验合同 |
| 基础 TCN checkpoint | 完整调查；哨兵的条件模型阶段 | `research/tcn_loss_model/checkpoints/base/tcn_board_cnn_time_model_best.pt` | backbone、特征顺序、预处理或 checkpoint 格式改变 | `PRIMARY_MODEL.json`、预处理快照、兼容性测试和公开发布资产 |
| 哨兵参照入口 | 哨兵 | `sentinel_reference_config.json` | 启用新参照总体、新版本分箱/范围、判定口径或随机复现参数 | `version`、源/派生目录、directed records、SHA-256 manifest、Elo 范围、分箱宽度、WLD 起始 ply、bootstrap 次数和 seed |
| 哨兵源参照总体 | 哨兵 | `research/offbook_detection/data/oq_elo_matchup400_reference_level22_1600plus_20260815/` | 当前 OQ 玩家/等级/对局分布不再具有代表性，样本覆盖不足，或 Level22 引擎合同改变 | 新建带日期/版本的源目录；重新抓取 bundle、完成全部 Level22 结果和 audit；不要覆盖旧总体 |
| 哨兵派生参照 | 哨兵 | `research/offbook_detection/data/oq_sentinel_reference_level22_1600plus_v6_20260819/` | 源参照、脱谱算法、指标定义、分箱或 Level22 合同任一改变 | 用 `sentinel_analysis.py build-reference` 全量重建；确认 `reference_build_audit.json` 的 `ok=true`，再生成并核对 `reference_sha256_manifest.json` |
| estimated-Elo 条件 Reference | 哨兵 estimated-Elo v2 | 候选配置 `sentinel_elo_reference_config_v2_20260828.json` 指向的版本化 conditional directory；正式指针尚未切换 | directed phase records、标准化尺度、Beta-Binomial/相邻 z 合同或统一缓存隔离政策改变 | 新目录运行 `prepare-conditional-elo-reference`；核对 progress、audit、config/input/records SHA；不得覆盖旧缓存 |
| estimated-Elo reference-z 敏感性 | 哨兵 estimated-Elo v2 | `audit-reference-z-sensitivity` 的账号级版本化 rebuild/audit 目录 | 条件 Reference、隔离策略或近邻实现改变 | 分层选择少量账号，逐账号重建；比较点估计和近邻 SHA；重建 manifest 不得冒用统一 cache calibration |
| estimated-Elo v2 calibration | 哨兵 estimated-Elo v2 | 候选配置中的独立 calibration directory；正式 calibration 尚未完成 | 条件 Reference、算法版本、A/B/C 选择、T95 或诊断阈值改变 | 固定 16 进程全量重算；validation 不参与选择或 T95；覆盖不足时保持 `calibration_unavailable`，不得切换主配置 |
| estimated-Elo v3 calibration | 哨兵 estimated-Elo v3 | `research/offbook_detection/data/oq_sentinel_elo_calibration_v3_20260829/`；本次运行已完成但状态为 `calibration_unavailable` | 条件 Reference、adaptive search、Beta-Binomial 拟合、T95 或诊断合同改变 | 固定 16 进程、每 task 一个账号、`chunksize=1`、每 worker 查询线程 1；validation 不参与选择或 T95；所有曲线拟合失败时必须保留 unavailable，不得切换主配置 |
| Egaroucid 引擎及 Level22 合同 | 两条链路 | 仓库同级的 `Egaroucid_for_Console_7_8_1_Windows_AVX512_AMD/`，以及 `wechat-decrypt/agent_egaroucid_analysis.py` | 引擎版本、默认 book、输出 JSON、评估尺度、命令行参数或运行器格式改变 | 重新做 Level22 合同审计；重建哨兵源/派生参照；评估是否需要重算训练 hint、归一化数据和模型 |

哨兵的统计扫描本身不读取 TCN 模型。只有扫描得到正式报告局，且至少有 8 盘模型
控制局时，才进入与完整调查相同的模型阶段。因此，模型资产损坏不应被误判为哨兵
参照扫描失效，但会阻止后续模型复核。

## P1：会随时间或上游变化的输入信息

| 维护对象 | 适用链路 | 当前依赖 | 主要失效信号 | 维护动作 |
| --- | --- | --- | --- | --- |
| OQ 对局公开接口及字段合同 | 两条链路 | `scripts/data/oq_account_bundle.py` 当前默认读取 `http://questgames.net` 的账户对局与对局详情 | URL、模式路径、响应字段、玩家标识、用时单位或公开范围改变；抓取/校验开始失败 | 先保存原始响应样本，再更新解析合同和测试；确认历史 bundle 仍可读取 |
| OQ Player 档案接口及 31 项 profile 合同 | 两条链路 | `research/tcn_loss_model/src/oq_player_profile.py` 与 `scripts/data/fetch_oq_player_profiles.py` | Socket.IO 端点/协议或字段改变；rating/战绩不变量失败；缺失率明显升高 | 更新抓取与归一化合同；若 31 项输入定义、缺失值策略或分布改变，重建 profile 数据、归一化参照并重新评估/训练主模型 |
| Human Frequency Book | 完整调查；哨兵的条件模型阶段 | `research/tcn_loss_model/data/oq_elo2000_5min_bilateral_10000_source_only_20260804/source_snapshot/othelloquest_human_frequency_nodes_ply1_30_min5.runtime.json` | 新对局累积后开局频率明显漂移，或棋谱解析/对称归一化规则改变 | 用版本化新快照替换默认指向；记录来源范围、截止时间、过滤条件、节点数和 SHA-256；用旧调查做回归比较 |
| profile 归一化参照 | 完整调查；哨兵的条件模型阶段 | `research/tcn_loss_model/outputs/oq_tcn_model_ready_11200_oq_profile_wld_ply39_20260808/model_ready_11200_oq_profile_wld_ply39.npz` | 新 profile 总体出现明显分布漂移；profile 字段、缩放或缺失值策略改变 | 重建 NPZ 和 provenance；同步主模型、personal materialization 及测试，不能只换 NPZ |
| 模型预处理与特征顺序 | 完整调查；哨兵的条件模型阶段 | `research/tcn_loss_model/provenance/source_snapshot/preprocessing.json` | 输入列、棋盘通道、时钟语义、hint 字段或标准化方式改变 | 将其与 checkpoint 作为不可拆分版本更新；运行 strict-load、data-contract 和预测回归测试 |
| 个人适配超参数 | 完整调查；哨兵的条件模型阶段 | `research/tcn_loss_model/config/personal_finetune.json` | 新验证显示校准、正则或优化过程不再合适 | 以验证结果支持变更；版本化配置，并记录对历史调查结论的敏感性 |
| 训练/推理所需原始研究快照 | 完整调查；哨兵的条件模型阶段 | `research/tcn_loss_model/data/oq_elo2000_5min_bilateral_10000_source_only_20260804/source_snapshot/official_research/` | 上游特征代码或数据合同改变，或该快照不再能复现当前模型输入 | 新建 provenance 快照并记录来源版本/提交；同步 preprocessing、模型和 `default_paths()` |

## P2：策略合同和发布信息

以下内容不是可单独替换的数据文件，但它们定义了上述资产的含义。修改时必须进行
版本迁移，不能让旧名称继续指向新语义。

- 哨兵范围与统计合同：Elo 上下限、100 分带宽、匹配规则、正式/低分扩展范围、
  ply 39 WLD 边界、伪选手抽样与 bootstrap 规则。入口在
  `sentinel_reference_config.json`，实现位于 `src/player_analysis_toolkit/sentinel.py`。
- estimated-Elo v2 合同：1600–2500 整数网格、四阶段 `(x,n)`、加权 Beta-Binomial、
  标准化双方 Elo、一阶相邻 z、统一冻结 reference-z 与目标账号整盘双方向排除。实现位于
  `src/player_analysis_toolkit/sentinel_elo.py`，维护入口位于
  `scripts/analysis/sentinel_elo_analysis.py`。v1/v2 schema 和 calibration 不得混用。
- estimated-Elo v3 在 v2 条件 Reference 模型合同之上增加独立的 adaptive
  multi-basin search（40/20/10/5/2/1）和 global cKDTree；其 search/calibration schema、
  config SHA 与 artifact 必须独立校验。2026-08-29 运行结果为
  `calibration_unavailable`，不得把 v3 artifact 当作已验证的 95% 校准范围。
- 脱谱算法合同：`scripts/analysis/detect_offbook.py`。算法变化会改变哨兵参照记录，
  因此必须重建参照，而不能继续复用旧 `directed_target_records.jsonl`。
- 模型数据合同：`research/tcn_loss_model/DATA_CONTRACT.md`、
  `ACTIVE_BOARD_STATE_CONTRACT.md`、`MODEL_PLAN.md`。
- 方法解释：`METHOD.md` 与 `README.md`。正式指标或默认资产变化时应同步更新。
- 大型资产发布清单：`DATASETS.md` 与 `tools/repository/public-release-layout.json`。
  更新正式模型或其必需数据后，应生成新的发布资产 manifest 和 `SHA256SUMS.txt`。

## 不作为长期维护基准的内容

- `investigations/` 下的账户 bundle、profile 快照、模型适配器和报告：它们是单次调查证据，应冻结保存。
- `research/**/outputs/` 中未被 `PRIMARY_MODEL.json` 或正式默认路径引用的实验产物。
- `research/tcn_loss_model/archive/`、历史 server handoff、smoke run、日志、缓存和 `tmp/`。
- 每次调查由 Agent/用户冻结的 reported/control game IDs：它们属于该次案件，不应滚动更新。
- `assets/standard_openings.json`：当前完整调查的非模型统计不依赖标准定式命名目录；除非以后正式流水线接入该目录，否则不列为本清单的运行依赖。

## 一次正式更新的完成条件

1. 新数据、参照或模型写入新的带版本/日期目录，不覆盖旧资产。
2. 记录数据来源、采集截止时间、筛选规则、schema、软件/引擎版本和随机 seed。
3. 为正式文件生成 SHA-256 manifest；模型 manifest 同时记录成员文件大小和哈希。
4. 先完成数据合同、checkpoint strict-load、参照 build audit 和历史样例回归，再修改
   `PRIMARY_MODEL.json` 或 `sentinel_reference_config.json` 指针。
5. 检查 `run_player_investigation.py::default_paths()` 中的固定路径，并同步 README、METHOD、
   DATASETS 和发布清单。
6. 用一份已有完整调查和一份已有哨兵调查做只读回归比较，记录差异及是否会改变结论。
7. 保留旧入口信息以便复现和回滚；不要改写历史调查目录中的 manifest 或报告。
8. estimated-Elo 更新还必须先完成小参考 smoke、一个账号完整 901 点网格、正式条件缓存、
   16 进程 calibration、独立 validation 和 A/B/C 对照；validation 未确认覆盖时不得切换
   `sentinel_elo_reference_config.json`。

## 每次正式运行前的快速检查

- OQ 对局和 profile 的实时 smoke/字段不变量通过。
- 引擎、运行器、book、Level22、worker/thread/hash 参数与参照合同一致。
- `PRIMARY_MODEL.json` 指向的基础 checkpoint、ensemble manifest 和 12 个成员哈希一致。
- profile 归一化 NPZ、preprocessing、Human Frequency Book 和个人适配配置均存在且属于同一数据/模型合同。
- 哨兵配置指向的 derived reference audit 为成功状态，manifest 中所有文件哈希一致。
- 本次运行配置保存了上述入口的绝对路径、版本或哈希，能够在未来复现。
