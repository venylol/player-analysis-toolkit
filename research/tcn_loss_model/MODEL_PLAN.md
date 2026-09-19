# Model plan

## Fixed architecture

Only one model is supported: all 362 official numeric/missing-indicator inputs,
23 ordered board-CNN planes, FiLM conditioning, and the six-level causal TCN.
The original thinking-time head remains unchanged. The loss head receives the TCN
latent state plus `log1p(actual_thinking_time_ms / 1000)` and a missing flag.
Current actual time is never passed into the backbone or time head.

The OQ profile model schema is
`time-plus-four-class-severity-plus-three-class-wld-oq-profile-v1`. The official 362 numeric fields,
23 board planes, board CNN, causal TCN, and thinking-time head remain unchanged and
strict-load from the official checkpoint. A small MLP reads the selected normalized
Player fields concatenated with their separate missing masks. Its final projection
produces severity-only FiLM gamma/beta and is initialized to exact zero:

`conditioned = severity_hidden * (1 + gamma) + beta`.

Consequently a newly initialized profile model is numerically identical to the
baseline severity model when their ordinary severity weights are the same. Player
context cannot enter or alter the thinking-time head. A profile checkpoint has schema
`tcn-loss-profile-wld-checkpoint-v2` and records the 31-field order, train-only profile
preprocessing hash, temporal policy, temporal-leakage authorization, ablation name,
and ablation hash. Profile training and inference fail rather than falling back when
any required array or identity field is absent or different.

The fixed cumulative ablations are:

1. baseline: no Player branch;
2. `overall-both-10-plus-five-differences`: 15 selected fields;
3. `with-color`: 23 selected fields (the two color differences were already among
   the five differences in step 2 and are not duplicated);
4. `with-strong-weak`: all 31 fields;
5. `full-31`: all 31 fields.

Steps 4 and 5 are deliberately equivalent under the user-fixed 31-field list. Both
names remain in manifests so the requested comparison is represented without adding
unauthorized features.

For the current retrospective dataset only, cumulative snapshots queried after the
games are trusted by explicit user authorization. This is documented temporal
leakage, not a claim of historical point-in-time features.

## Uniform loss-history input policy

There is exactly one input policy for training, validation, test, personal-control,
and reported-game inference:
`uniform-no-current-player-loss-history-v1`. There is no full/masked mode, random
context masking, cohort-dependent input, dual branch, or masking pretraining.

The current player's prior `raw_loss`, `disc_loss`, zero/ge4/ge10 outcomes, counts,
rates, cumulative/mean/recent values, sequences, predicted severity probabilities,
and prediction residuals are never constructed as model features. They are omitted
from `X` entirely; no zero-filled column or missing-indicator placeholder is retained.
Targets and audit arrays remain outside `X` and are used only by the loss function,
metrics, and exported audit rows. The official checkpoint's ordered 362 features have
been audited and contain zero loss-history fields. Validation rejects loss-, severity-,
blunder-, mistake-, or residual-named inputs (including `__missing` variants), requires
the exact checkpoint feature order, and requires the policy identifier in every
model-ready bundle and run manifest.

The single severity head emits four Softmax probabilities in this fixed order:

1. `class_zero`: loss = 0;
2. `class_1_3`: loss in 1–3;
3. `class_4_9`: loss in 4–9;
4. `class_ge10`: loss >= 10.

External probabilities are derived from that same distribution:

- `P(zero) = p0`;
- `P(positive) = 1 - p0`;
- `P(ge4) = p2 + p3`;
- `P(ge10) = p3`.

Consequently `P(ge10) <= P(ge4) <= 1 - P(zero)` and the four raw class
probabilities sum to one by construction. Training uses one weighted four-class
cross-entropy; no concrete loss value is regressed and no independent binary heads
can contradict each other. Class weights are a single finite four-entry config.

The same conditioned hidden representation feeds an independent WLD linear head in
the fixed order `class_no_wld_loss`, `class_half_wld_loss`,
`class_full_wld_loss`. Its public scalar is
`expected_wld_loss = 0.5 * p_half + p_full`, always in `[0,1]`. WLD supervision uses
three-class cross-entropy only at pass-excluded `global_placement_ply >= 39`; a batch
without valid WLD nodes contributes a graph-connected zero WLD term. Legacy
checkpoints may omit only `wld_head.weight` and `wld_head.bias`; any other missing or
unexpected state key fails before a final strict load.

## Transfer sequence

1. Strict-load the official best checkpoint into the unchanged backbone/time head.
2. Train only the randomly initialized four-class severity branch for a short head stage.
3. Unfreeze and fine-tune the backbone and both tasks at the smaller configured LR.
4. Retain the thinking-time MSE as an auxiliary objective.
5. Select one checkpoint on the fixed validation games and evaluate the test games once.

An optional `wld-head-only` extension freezes the backbone, profile conditioning,
thinking-time head, and severity head. It selects on validation WLD cross-entropy and
retains the incoming checkpoint as the epoch-zero baseline. Test data never selects
the extension epoch.

No no-time, no-CNN, Transformer, CatBoost, stage-specific model, or broad search is
implemented.

## Pass finding and migration risk

The source trainer did not filter `actual_move == "-"`; it grouped all rows into
sequences. The official checkpoint reports `max_seq_len=66`, proving that its legacy
sequence position/`ply` semantics can exceed the 60 actual placements. The new flow:

1. sorts the unmodified source rows by `game_id, move_index`;
2. computes adjacent-child labels while pass rows remain available;
3. excludes pass rows as decision targets;
4. creates `global_placement_ply` by cumulative non-pass moves;
5. rejects any decision ply outside 1–60; and
6. remaps the legacy numeric feature named `ply` to this placement index.

This preserves tensor/order checkpoint compatibility but intentionally corrects the
semantic meaning of `ply`. The change must be stated in the final data manifest and
checked during fine-tuning; it must not be described as identical preprocessing.
