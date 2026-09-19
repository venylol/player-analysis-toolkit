# First Long Think v2

v2 保留 strict ply 5–38 的 clip/cap 与 v1 的 1.75 倍历史中位数时间规则，并增加一个只使用当前局面绝对估值的截止点：

```text
evaluation_cutoff = strict ply 5–38 内，目标棋手本人第一个 abs(current_score) > 6 的非 pass 节点

time_anchor = evaluation_cutoff 之前，第一个
              current_time > 1.75 × median(all prior same-player times) 的节点

final_anchor = time_anchor（如果存在）
               否则 evaluation_cutoff（如果存在）
               否则 no_anchor
```

`raw_loss`、`disc_loss` 和任何相对子损都不参与规则。估值截止节点本身不再进入时间规则。只分析源 `games.csv` 的 `recorded_sides` 中明确记录了真实用时的棋手视角。

```powershell
python research/offbook_detection/first_long_think_v2/build_review_sample.py `
  --output-dir research/offbook_detection/first_long_think_v2/outputs/oq_11200_seed42_20260814
```
