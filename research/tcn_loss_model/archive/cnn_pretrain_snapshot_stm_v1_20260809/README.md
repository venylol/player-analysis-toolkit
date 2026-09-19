# Pretrained CNN and snapshot-side-to-move research archive

Archived on 2026-08-09. This directory preserves the complete local research line
for the LV17 pretrained residual CNN and the `snapshot_side_to_move_v1` joint-model
experiments. Nothing in this archive is part of the active toolkit model contract.

## Scope

The archive contains:

- LV17 source records and board-CNN materialization shards;
- board-CNN pretraining code, configuration, tools, smoke outputs, and checkpoints;
- transferred residual64x6 checkpoints and perspective sidecars;
- the regenerated 11,200-game snapshot-side-to-move model-ready NPZ;
- Stage A, Stage B, both Stage C attempts, and direct-full-38 outputs;
- experiment-specific tests, data conversion scripts, configuration, and contracts;
- snapshots of shared runtime files as they existed during the experiment;
- a sparse GitHub reference checkout pinned to the pre-experiment commit
  `4a27115edfba8e82f07354a80740632b81aee28f`.

At final verification the directory contained 407 files (about 3.015 GiB).
All moves were same-volume and reversible. No source artifact was deleted.

## Key integrity hashes

| Artifact | SHA-256 |
| --- | --- |
| LV17 CNN checkpoint `outputs/board_cnn_pretrain_64x6_seed42_20260809T143514/checkpoints/epoch-0005-step-00140628.pt` | `157911ceb0558005cc7193c4faa3395493b6181e2f8e7b87fd87bb4d06dfc9b4` |
| Transferred checkpoint `checkpoints/transferred/tcn_board_cnn_residual64x6_pretrain_epoch6_seed42_v2.pt` | `004e7485255829ce14711650646e41539493cdd3390642aea7d41e72ef411746` |
| Snapshot model-ready NPZ `outputs/oq_tcn_model_ready_11200_snapshot_stm_v1_profile_wld_from_retained_20260809_v2/model_ready_11200_snapshot_stm_v1_profile_wld.npz` | `5ca959f7d96d87ed3a8dbed0a887fef1f085fff9b9b4418f1272abddb1ba8a89` |
| Stage A best | `ccd3e3e3a20ae3f67feea1aab45c4079a37543034225e6114ad7a0d4800b095c` |
| Stage B best | `e98e6f0a484127401a909884e977a75f8dfd7110d9084191b3cd024de2ea679f` |
| Stage C patience-6 best | `68478b74511f78dc87018410b07408d77772199497209c4afb8d5f6f8e720ce2` |
| Direct-full-38 best | `25bed8f3d435b629111c681a2fb9444f8bdc631ab81e8493fac4c0297557e9e8` |

## Active restoration

The active files were restored from the GitHub pre-experiment baseline rather than
manually approximated. `PRIMARY_MODEL.json` points to the published original
12-member ensemble at `models/primary_wld_ensemble12/ensemble_manifest.json`.
Its manifest SHA-256 is
`4a6148e3c3af4eecbc01b42ed489d4a85d88d233c868668237352f309ca98fba`.
