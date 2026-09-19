# TCN human disc-loss model

This directory contains the official full-feature 8×8 board-CNN causal TCN, its
retained thinking-time head, one current-time-conditioned four-class severity head,
and an independent three-class WLD head on the same conditioned representation.
Formal training artifacts are retained under `outputs/`. The 12 selected public
checkpoints are also published under `models/primary_wld_ensemble12/`; no personal
adapter or profile-specific fine-tuned checkpoint is included there.

Every split and future cohort uses the single
`uniform-no-current-player-loss-history-v1` input policy. Current-player prior loss
labels, aggregates, model probabilities, and residuals are absent from the model
feature tensor rather than masked with zeros. There is no full/masked dual mode or
random context masking.

## Primary model

`PRIMARY_MODEL.json` is the repository's authoritative model pointer. The current
primary model is the accepted 12-member, 11,200-game full-31 OQ Player profile
ensemble with the ply-39 WLD head:

```text
models/primary_wld_ensemble12/ensemble_manifest.json
```

Inference and player investigations should resolve the ensemble through the primary
model registry and must use profile-aware materialization. The superseded 3,531-game
baseline ensemble without Player profile inputs is retained under `archive/models/`
for historical reproduction only.

Safe checks:

```powershell
python train.py inspect
python train.py check-checkpoint --base-checkpoint checkpoints/base/tcn_board_cnn_time_model_best.pt --preprocessing provenance/source_snapshot/preprocessing.json
python train.py smoke-test --base-checkpoint checkpoints/base/tcn_board_cnn_time_model_best.pt
python train.py status --output-dir outputs/waiting-for-data
```

## Public OQ Player profile snapshots

The reusable read-only Player-page client speaks the public Socket.IO 0.9
`d4b6e7ef` protocol and does not read credentials. It preserves the complete raw
response separately from the normalized profile. Existing output directories require
the explicit resumable mode and existing snapshot files are never overwritten.

```powershell
# one account
python scripts/data/fetch_oq_player_profiles.py fetch `
  --account hero9 --output-dir outputs/oq_profiles_20260804

# UTF-8 CSV/JSON/JSONL/text list, resuming successful accounts
python scripts/data/fetch_oq_player_profiles.py fetch `
  --input accounts.csv --output-dir outputs/oq_profiles_20260804 --resume

# explicit network-only smoke; numeric ratings/records are not frozen assertions
python scripts/data/fetch_oq_player_profiles.py live-smoke
```

Each normalized `oq-player-profile-snapshot-v1` record contains the requested and
case-normalized account, original returned `id`/`name`, query gtype, interface/script
versions, rating/high, overall win/loss/draw/played, the sente/gote/strong/weak
records, `last`, `profile_fetched_at_utc`, explicit missing categories, and the
canonical raw-response SHA-256. The raw envelope retains the entire server response,
including fields intentionally excluded from model inputs.

The explicitly authorized retrospective materialization command is:

```powershell
python scripts/data/materialize_oq_profile_context.py `
  --input-npz data/oq_elo2000_5min_bilateral_10000_model_ready_20260803_final/model_ready/model_ready_10000.npz `
  --games data/oq_elo2000_5min_bilateral_10000_model_ready_20260803_final/handoff/games.csv `
  --snapshots-dir outputs/oq_player_profiles_2339_retrospective_20260804 `
  --output-dir data/oq_elo2000_5min_bilateral_10000_oq_profile_retrospective_20260804 `
  --policy retrospective-current-profile-trusted-temporal-leakage-v1 `
  --allow-temporal-leakage
```

This command deliberately backfills cumulative Player-page statistics queried after
the historical games. The user explicitly chose to trust this rough retrospective
context for training. It is not a leak-free temporal design: overall/color/strong/
weak records include win, draw, game-count, and rating information accumulated after
some or all target games. The output remains isolated under its own profile schema,
policy, hashes, masks, preprocessing, coverage report, and warning-bearing manifest.
The original 362-feature NPZ is never modified.

After the user confirms the completed new dataset, the only formal training command is:

```powershell
python train.py train --data <model-ready.npz> --context-metadata <context.csv> --output-dir outputs/<run-id> --base-checkpoint checkpoints/base/tcn_board_cnn_time_model_best.pt --config config/default.json --run-name <run-id> --confirm-new-data-ready
```

Inference is also explicit:

```powershell
python train.py predict --data <model-ready.npz> --checkpoint outputs/<run-id>/best.pt --base-checkpoint checkpoints/base/tcn_board_cnn_time_model_best.pt --output outputs/<run-id>/node_predictions.csv --device cuda:0
```

Profile-conditioned training and inference use separate commands and refuse missing
profile arrays or mismatched feature/preprocessing/policy hashes:

```powershell
python train.py smoke-test-profile `
  --base-checkpoint checkpoints/base/tcn_board_cnn_time_model_best.pt `
  --ablation full-31 --device cuda:0

python train.py train-profile `
  --data data/oq_elo2000_5min_bilateral_10000_oq_profile_retrospective_20260804/model_ready_10000_oq_profile_retrospective.npz `
  --output-dir outputs/<profile-run-id> `
  --base-checkpoint checkpoints/base/tcn_board_cnn_time_model_best.pt `
  --config config/profile_full.json --run-name <profile-run-id> `
  --confirm-new-data-ready

python train.py predict-profile `
  --data <profile-model-ready.npz> --checkpoint outputs/<profile-run-id>/best.pt `
  --base-checkpoint checkpoints/base/tcn_board_cnn_time_model_best.pt `
  --output outputs/<profile-run-id>/node_predictions.csv --device cuda:0
```

Create a fixed-theme PNG diagnostic from the prediction export:

```powershell
python scripts/reporting/create_tcn_loss_diagnostic_png.py `
  --predictions outputs/<run-id>/node_predictions.csv `
  --output outputs/<run-id>/tcn_loss_diagnostics.png `
  --split test `
  --sample-size 8000 `
  --dpi 170
```

The reusable plotter accepts `train`, `validation`, or `test`. It renders the original
disc-loss scatter and three calibration panels plus a separate WLD scatter of actual
`wld_loss` against the primary public metric `expected_wld_loss`. The WLD panel uses
only label-available nodes at pass-excluded `global_placement_ply >= 39`; earlier plies
never enter the plot. Group means include 95% mean intervals. Calibration, group means,
and metrics use every labelled node in the selected split; only dense scatter layers
are sampled.

Formal training is deliberately gated by the `train` subcommand and
`--confirm-new-data-ready`. It requires CUDA and a validated model-ready NPZ with
game-isolated train/validation/test splits. It never searches for old data or old
checkpoints, never silently falls back to CPU, and never resumes without `--resume`.

The seed-42 epoch probe is the sole exception to normal test evaluation: it must use
`--skip-test-evaluation`. That mode writes `validation_only_completion.json` and never
opens the fixed test split. `scripts/modeling/select_seed42_epoch_schedule.py` refuses to freeze
a schedule if a probe `test_metrics.json` exists or if too little validation curve
remains after the selected minimum.

The raw-data validator accepts an explicit CSV, Parquet, or JSONL path. It links each
placement to the next actual placement while retaining intervening consecutive pass
rows as continuity evidence, then excludes pass records. The source `ply` is retained
as `source_ply_including_pass`; model/output `ply` is remapped to
`global_placement_ply`, the 1–60 actual-placement index.

The severity head never regresses a disc-loss value. Its mutually exclusive Softmax
classes are `0`, `1–3`, `4–9`, and `>=10`. One distribution supplies
`probability_loss_zero`, `probability_loss_ge4`, and `probability_loss_ge10`, so their
logical ordering is structural rather than post-hoc.

The WLD head emits `class_no_wld_loss`, `class_half_wld_loss`, and
`class_full_wld_loss`. Public inference uses
`expected_wld_loss = 0.5 * p_half + p_full`; `probability_wld_any = p_half + p_full`
is diagnostic only. Both are exported only for label-applicable placements at
pass-excluded `global_placement_ply >= 39`; earlier placements are blank/not
applicable. Training uses three-class cross-entropy, not WLD-value regression.

`training_mode: "wld-head-only"` freezes every parameter except `wld_head` and
selects checkpoints using validation WLD cross-entropy. The incoming warm-start
checkpoint remains the epoch-zero candidate, so an extension cannot replace it
unless validation WLD loss improves. `--warm-start-checkpoint-name latest.pt`
explicitly starts a new ensemble stage from each completed member's latest checkpoint.

## Fixed-test ensemble and personal residual adapters

Train deterministic members sequentially while preserving the original test games:

```powershell
python train.py ensemble --data <model-ready.npz> --context-metadata <context.csv> `
  --output-dir outputs/<ensemble-id> --base-checkpoint checkpoints/base/tcn_board_cnn_time_model_best.pt `
  --config config/default.json --seeds 42 43 44 45 46 47 48 49 50 51 52 53 `
  --confirm-new-data-ready
```

The ensemble entrypoint writes one deterministic whole-game split bundle per seed,
keeps the original test membership fixed, trains members on one CUDA device in order,
and resumes incomplete members from `latest.pt`. A completed member is skipped only
when its data and official checkpoint hashes match.

The reusable personal materializer accepts an account bundle, reported-game IDs, a
target player, and an explicit effective clock. It computes only that bundle's
official hint1/hint6 inputs and reuses the same 362-feature/23-channel pipeline and
checkpoint preprocessing. Source and effective clock fields remain in the resulting
NPZ for audit.

Personal calibration freezes every parameter in each trained member. It trains two
independent zero-initialized residual-logit adapters: 64-by-4 plus bias for severity,
and 64-by-3 plus bias for WLD. Both use the fixed game-equal cross-entropy,
`KL(base || personal)`, and L2 objective. The WLD adapter uses only nodes with an
available WLD label at pass-excluded `global_placement_ply >= 39`; games with no
eligible WLD node remain in the fixed control-game denominator with zero contribution:

```powershell
python train.py personalize-ensemble --data <personal-model-ready.npz> `
  --ensemble-manifest outputs/<ensemble-id>/ensemble_manifest.json `
  --base-checkpoint checkpoints/base/tcn_board_cnn_time_model_best.pt `
  --config config/personal_finetune.json --output-dir outputs/<personal-id>/adapters
```

No personal validation split, reported-game stopping rule, or backbone update is used.
