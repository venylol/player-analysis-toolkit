# Board perspective contract

## Root cause and new contract

The retained official board-context snapshot encodes raw OQ boards as fixed
black=`2` and white=`3`. Its context search correctly selects current,
previous-opponent, and previous-own rows, but it never applies the selected row's
`side_to_move`. That conflicts with LV17 pretraining, where X always denotes the
snapshot mover.

All newly generated formal, personal, control, and ablation data therefore uses:

`board_perspective = snapshot_side_to_move_v1`

Each selected snapshot is normalized independently. Black-to-move retains `2/3`;
white-to-move swaps only occupied tokens `2/3`. Padding `0` and empty `1` are
unchanged. Missing historical contexts remain all-zero. Coordinates, historical
move tokens, hint move tokens, hint values, numeric features, and labels are not
spatially transformed or sign-flipped. Existing score labels remain under their
documented acting-row perspective; this repair found no evidence requiring a score
sign change.

The 23 planes still enter one shared CNN. The stem's first three planes now mean
`current_empty`, `current_local_mover_X`, and `current_local_opponent_O`.

## Active data-preparation entries

- `src/board_perspective.py`: canonical normalization and three-context builder.
- `scripts/data/materialize_oq_tcn_model_ready.py`: formal and new-control base
  materialization; directly calls the canonical builder.
- `scripts/data/materialize_personal_oq_tcn_model_ready.py`: personal materializer;
  delegates to the same formal array builder.
- `scripts/data/merge_model_ready_npz.py`: merges only cohorts with equal explicit
  perspective and feature-revision arrays.
- `scripts/data/materialize_oq_profile_context.py` and
  `scripts/data/materialize_wld_labels.py`: preserve and report the input perspective.
- `scripts/data/subset_model_ready_npz.py`, `resplit_model_ready_npz.py`, and
  `reassign_personal_inference_split.py`: derive new files without changing board
  arrays or scalar contract arrays.
- `src/training.py`, `src/inference.py`, and `src/personal_adapter.py`: require data,
  configuration, and checkpoint perspective identity before use.
- `scripts/server_data_prep/*.ps1`, `validate_server_model_ready.py`, and the Windows
  package builders: use versioned snapshot-perspective output locations and carry
  the maintained `src` implementation.

## Intentionally retained legacy/history

- `data/oq_elo2000_5min_bilateral_10000_source_only_20260804/source_snapshot/official_research/train_causal_transformer_board_model.py`
  is an immutable source snapshot. It remains dynamically imported for retained
  numeric/metadata compatibility, but its board-context builder is no longer called.
- `data/.../source_snapshot/materialize_oq_tcn_model_ready.py` and
  `provenance/source_snapshot/*` are historical evidence and are unchanged.
- Existing server package directories/ZIPs, old model-ready NPZs, Stage A outputs,
  and fixed-color checkpoints are immutable legacy artifacts. They were not
  overwritten or relabelled.
- Existing formal configs are explicitly marked `legacy_fixed_color`; the new CNN
  Stage A/joint/control drafts are explicitly `snapshot_side_to_move_v1`.

## Cache and checkpoint compatibility

Old board-context/model-ready products do not contain the new feature revision and
must not be used for new training. Regeneration must target a new directory whose
name contains `snapshot_stm_v1`. Do not copy the scalar metadata onto an old NPZ:
the board tokens themselves must be regenerated from raw snapshot rows.

The independent LV17 CNN checkpoint is unchanged. The transferred 23-channel
checkpoint is annotated by a separate SHA-bound
`.pt.board-perspective.json` manifest; the binary checkpoint was not modified.
Newly trained checkpoints embed `board_perspective`. A missing or mismatched value
is a hard error. Old checkpoints may only be reproduced with explicitly annotated
`legacy_fixed_color` data and configuration.

## Required sequence before formal joint training

Do not run these commands until the user authorizes full data regeneration/training.
First rebuild the 10,000- and 1,200-game base NPZs from the audited raw handoffs by
using the updated Windows entrypoints; their output names are respectively
`model_ready_10000_snapshot_stm_v1.npz` and
`model_ready_1200_snapshot_stm_v1.npz`. Then merge the returned cohorts:

```powershell
python scripts/data/merge_model_ready_npz.py `
  --base-data <returned-10000>/model_ready_10000_snapshot_stm_v1.npz `
  --extension-data <returned-1200>/model_ready_1200_snapshot_stm_v1.npz `
  --base-context <returned-10000>/position_context_metadata.csv `
  --extension-context <returned-1200>/position_context_metadata.csv `
  --base-games <returned-10000>/games.csv `
  --extension-games <returned-1200>/games.csv `
  --output-dir outputs/oq_tcn_model_ready_11200_snapshot_stm_v1 `
  --output-name model_ready_11200_snapshot_stm_v1.npz `
  --expected-base-games 10000 --expected-extension-games 1200
```

Materialize the unified 31-feature profile branch and WLD labels using the existing
profile snapshot directory and the old base checkpoint only as the fixed numeric/
hint-value decoding contract. Both scripts preserve the new board perspective:

```powershell
python scripts/data/materialize_oq_profile_context.py `
  --input-npz outputs/oq_tcn_model_ready_11200_snapshot_stm_v1/model_ready_11200_snapshot_stm_v1.npz `
  --games outputs/oq_tcn_model_ready_11200_snapshot_stm_v1/games.csv `
  --snapshots-dir outputs/oq_player_profiles_2339_retrospective_20260804 `
  --output-dir outputs/oq_tcn_model_ready_11200_snapshot_stm_v1_profile `
  --output-name model_ready_11200_snapshot_stm_v1_profile.npz `
  --policy retrospective-current-profile-trusted-temporal-leakage-v1 `
  --allow-temporal-leakage

python scripts/data/materialize_wld_labels.py `
  --input outputs/oq_tcn_model_ready_11200_snapshot_stm_v1_profile/model_ready_11200_snapshot_stm_v1_profile.npz `
  --base-checkpoint data/oq_elo2000_5min_bilateral_10000_source_only_20260804/source_snapshot/tcn_board_cnn_time_model_best.pt `
  --output-dir outputs/oq_tcn_model_ready_11200_snapshot_stm_v1_profile_wld `
  --output-name model_ready_11200_snapshot_stm_v1_profile_wld.npz
```

Validate before any training:

```powershell
python train.py validate `
  --data outputs/oq_tcn_model_ready_11200_snapshot_stm_v1_profile_wld/model_ready_11200_snapshot_stm_v1_profile_wld.npz `
  --require-oq-profile `
  --board-perspective snapshot_side_to_move_v1
```

Only after that validation and explicit user authorization may Stage A be started
with the snapshot-perspective config and the SHA-bound transferred CNN checkpoint.
The exact single-seed Stage A command will be:

```powershell
python train.py train-profile `
  --data outputs/oq_tcn_model_ready_11200_snapshot_stm_v1_profile_wld/model_ready_11200_snapshot_stm_v1_profile_wld.npz `
  --context-metadata outputs/oq_tcn_model_ready_11200_snapshot_stm_v1/position_context_metadata.csv `
  --output-dir outputs/cnn_tcn_stage_a_snapshot_stm_v1_seed42 `
  --base-checkpoint checkpoints/transferred/tcn_board_cnn_residual64x6_pretrain_epoch6_seed42_v2.pt `
  --initial-profile-checkpoint outputs/oq_profile_full31_wld_ply39_ensemble12_11200_latest10_wldhead6_total16_baselineguard_20260809/members/member_01_seed_42/best.pt `
  --config config/cnn_tcn_stage_a_seed42.json `
  --run-name cnn-tcn-stage-a-snapshot-stm-v1-seed42 `
  --skip-test-evaluation `
  --confirm-new-data-ready
```

This command is documented only; it was not run during this repair.
