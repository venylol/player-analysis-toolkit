# 双向时间 Transformer 第二阶段冻结实施规格

## 一、交付边界

第二阶段只完成以下内容：

1. 校验并读取冻结的 61,145 局数据；
2. 执行整局单方向规范化；
3. 使用冻结空间编码器生成 FP32 `board_cache`；
4. 训练一个随机种子的双向时间 Transformer；
5. 完成无标签技术验证；
6. 冻结时间 Transformer 并生成 `sequence_cache`。

本阶段不训练单变点模型，不生成脱谱候选，也不查看旧锚点或人工标记。

## 二、冻结数据

- 数据目录：`research/offbook_detection/data/oq_transformer_61145_20260813`
- 原始完整棋局：61,145
- 原始节点：3,725,188
- 目标视角序列：122,290
- 划分：train 48,922；validation 6,118；test 6,105
- 划分单位：完整原始棋局
- 同一原始棋局的两个目标视角必须在同一 split
- 不要求棋手身份在不同 split 之间隔离
- `oq_transformer_100000_20260813` 是未完成采集中间目录，不属于正式输入

## 三、整局单方向规范化

读取第一步黑棋落子，并选择将其映射到 `f5` 的固定 D4 变换：

| 原始首着 | transform_id | 规范首着 |
|---|---:|---|
| `d3` | 7 | `f5` |
| `c4` | 2 | `f5` |
| `f5` | 0 | `f5` |
| `e6` | 5 | `f5` |

一个原始棋局只选择一次变换。所有后续 `board_before`、着法坐标、合法步坐标及回放核验都沿用该变换。禁止节点级重新定向。

模型只处理规范方向：不随机 D4、不缓存 8 个方向、不做多方向推理平均、不加入 D4 一致性损失。

## 四、目标棋手视角

每局构造黑方目标和白方目标两条序列。整局定向先于目标视角编码：

```text
原始棋盘与着法
→ 首着 f5 整局定向
→ 目标棋手/对手占用编码
→ 冻结空间 Transformer
```

占用编码固定为：空位 0、目标棋手棋子 1、对手棋子 2。`actor_is_target` 独立输入空间编码器。

## 五、board_cache

- 来源模型：`spatial_board_encoder_epoch3_forced.pt`
- embedding 维数：96
- dtype：FP32
- 每个原始节点保存两个目标视角 embedding
- 每个目标视角节点只保存一个规范方向
- 按 shard 写入，支持可验证的断点续跑
- manifest 必须记录数据文件哈希、编码器哈希、定向契约、目标视角、dtype、shape、shard 范围和 shard 哈希
- manifest 不匹配时必须明确失败，不能静默复用缓存

## 六、节点输入

时间 Transformer读取完整已结束棋局的双方节点。节点输入至少包含：

- FP32 board embedding
- `log1p(thinking_time_ms)`
- `log1p(target_remaining_time_ms_after)`
- `log1p(opponent_remaining_time_ms_after)`
- `strict_ply`
- 完整节点位置
- `actor_is_target`
- 当前行动方执色
- pass 标识
- 时间控制标识

时间稳健中心和尺度只在 train split 上拟合。当前数据只有五分钟无加秒时间控制。剩余时间口径固定为本次决策完成后。

## 七、首版模型与训练

首版冻结使用双向时间 Transformer：hidden size 128、3 层、4 heads、FFN 512、dropout 0.1。若 8GB 显存实测无法运行，只允许调整 batch size 或梯度累积；结构变化必须先重新确认。

随机种子固定为 42。空间 Transformer 全程冻结。

正式训练参数固定为 batch size 1024、AdamW、weight decay `0.01` 和 FP16 混合精度。首轮以学习率 `3e-4` 训练至 24 epochs；固定 validation 的最佳点为 epoch 22。最终从 epoch 22 checkpoint 连同优化器状态恢复，将学习率降至 `1e-4`，训练至总计最多 32 epochs。validation 对每条序列固定同一个遮盖节点，并以三类逐元素 MSE 之和选模；连续 3 个完整 epoch 未改善时早停。

每条训练样本随机选择一个有效节点，并只执行以下一种遮盖任务：

1. 遮盖该节点的实际思考时间；
2. 遮盖该节点的目标方和对手方剩余时间；
3. 遮盖该节点的 board embedding。

首轮不遮盖连续片段、不同时遮盖多个节点、不建立预测不确定度。本轮先观察单节点恢复结果，再决定是否扩展任务。

三种任务等概率抽样，各占 `1/3`。每种任务先对自身目标元素的损失取平均，再以权重 `1.0` 计入训练目标。该设置已经在查看人工脱谱信息前冻结。

## 八、验收与交付

至少交付：

- 数据契约和整局定向测试
- `board_cache` 与 manifest
- train-only 时间标准化统计量
- 时间 Transformer 配置、checkpoint 和训练日志
- validation 选模记录
- 固定 test 的单节点三类恢复指标
- 按棋局长度、执色、目标/对手节点等切片的技术报告
- 冻结的 `sequence_cache` 与 manifest

第二阶段完成标志是时间 Transformer 和 `sequence_cache` 已冻结并可复现。单变点 Student-t/高斯建模另行启动。
