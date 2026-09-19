# Active board-state contract

The active toolkit uses the original fixed-color board representation:

- `X` is always black.
- `O` is always white.
- `-` is empty.
- board token `0` is padding, `1` is empty, `2` is black/X, and `3` is white/O.
- `side_to_move` may select previous-own and previous-opponent history rows, but it must never swap the meanings of X and O.

The active implementation is the pre-experiment version from GitHub repository
`venylol/othello-tournament-autopilot`, commit
`4a27115edfba8e82f07354a80740632b81aee28f` (2026-08-08).

The pretrained residual-CNN and snapshot-side-to-move experiment is retained only
under `archive/cnn_pretrain_snapshot_stm_v1_20260809/`. It is not an active model,
data contract, configuration, or materialization path.
