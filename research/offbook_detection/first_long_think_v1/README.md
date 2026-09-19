# First Long Think v1

这是一个不使用人工或模型脱谱标签的确定性锚点规则。对每局的黑、白两个目标视角分别计算：

```text
在 strict ply 5–38（包含两端）内，
当前本人非 pass 着手用时 > 1.75 × 此前本人所有非 pass 着手用时的中位数，
取第一个满足条件的节点。
```

ply 5 之前的本人着手参与历史中位数，但不能成为锚点。如果 ply 5–38 内都没有满足条件的节点，输出 `no_anchor`。

## 运行

```powershell
python research/offbook_detection/first_long_think_v1/build_review_sample.py `
  --output-dir research/offbook_detection/first_long_think_v1/outputs/full_61145_seed42_20260814
```

输出包含：

- `anchor_records.jsonl`：全部棋手视角的检测事实；
- `summary.json`：整体数量、逐 ply 分布和抽样清单；
- `review_bundle_stratified30_plus2_no_anchor.json`：可直接导入现有回放审查器的 30 个分层锚点局加 2 个无锚点局。
