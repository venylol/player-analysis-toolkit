# Independent board-CNN pretraining and CNN+TCN joint-training plan

## Data and leakage contract

Pretraining reads only the current board, ordered `a1,b1,...,h8`, and decodes exactly three planes in this order: `current_empty`, `current_X`, `current_O`. `X` is always the side to move. Legal moves are generated from the board by Python Othello rules and are targets only. The score target is the current-player disc differential divided by 64. Neither target, hints, numerical features, history, player data, the TCN, nor any formal checkpoint is an input.

The fixed file split is train `0000000.txt`–`0000023.txt`, validation `0000024.txt`, test `0000025.txt`. The materializer emits one 25-byte-record shard per source file (16-byte 2-bit board, uint64 legal bitboard, int8 score), with a header and UTF-8 manifest. Complete shards are checksum-verified and reused. Every failed run retains its uniquely named attempt and failure manifest; a retry uses a new attempt path.

The source data and generated shards are local-only and must never be redistributed or committed. Full-dataset mirror/D4 normalization, canonicalization, symmetry deduplication, canonical hashes, and overlap checks against the 11,200 games are explicitly out of scope.

## Independent pretraining

The CNN starts from random initialization. Its final default capacity is `board_channels=64`, `residual_blocks=6`, `board_embedding_dim=96`, and `embedding_projection_kernel=1`. The shared network is a `3→64` 3×3 Conv/GroupNorm/GELU stem, six residual blocks (each two `64→64` 3×3 Conv/GroupNorm layers with GELU after the first convolution and after residual addition), then a `64→96` 1×1 projection/GELU, adaptive average pooling, and LayerNorm(96). Stem plus trunk contains thirteen 3×3 convolutions, giving an approximately 27×27 receptive field and full coverage of an 8×8 board. The legal head branches from the trunk's `64×8×8` feature and uses a 1×1 convolution to produce 64 logits with `BCEWithLogitsLoss`; a pass or terminal position has an all-zero target. The value head branches from the normalized 96-dimensional embedding and uses a `96→64→1` MLP with `SmoothL1Loss` against `score / 64.0`. The configured initial objective is:

`cnn_loss = 1.0 * legal_move_loss + 1.0 * value_loss`

Validation reports legal per-cell BCE, precision, recall, F1, exact legal-set accuracy, illegal-cell false-positive rate, score MAE/RMSE in discs, and score-sign accuracy. The exact default parameter counts are: 3-channel shared CNN 453,024; 23-channel shared CNN 464,544; legal head 65; value head 6,273. TCN, board projection, FiLM, numeric projection, and formal heads are excluded from these CNN counts. Checkpoints contain distinct shared-CNN, legal-head, value-head, and optimizer states plus epoch/optimizer-step, complete config and data contract, fixed split, seed, source/shard inventory with sizes and checksums, and best validation metrics.

Formal pretraining requires CUDA and an explicit confirmation flag. CPU is reserved for bounded smoke validation. The fixed test shard is not used for model selection.

## Transfer contract

Transfer constructs the new 23-channel residual encoder, copies the pretrained stem weight into channels 0–2, and writes exact zeros to channels 3–22. It strictly copies the stem bias and GroupNorm, all convolution and GroupNorm tensors in all six residual blocks, the 64→96 1×1 projection, and final LayerNorm. Every old `board_encoder.*` key is explicitly excluded and reported; every non-CNN key must match the new formal model exactly in name and shape before a final strict state load. No broad `strict=False` load is allowed. Legal/value heads remain training-only and are not inserted into the formal inference graph. A UTF-8 manifest records source, target, output hashes, excluded old CNN keys, missing/unexpected non-CNN keys, and every equality check.

If no transferred checkpoint is explicitly supplied, the existing model construction and loading paths remain unchanged.

## Fixed 11,200-game comparison

Baseline uses the current main-model training recipe with no independent CNN pretraining. Experiment uses the same TCN, 362 numerical inputs, fixed train/validation/test split, formal heads, thinking-time loss, four-class severity loss, and three-class WLD loss. The principal change is the independently pretrained CNN and the following staged schedule.

At every sequence time step, the existing 23-plane contract sends current board, previous-opponent board, previous-own board, prior actual moves, and hint/value context through one shared CNN. These contexts are channels, not three CNN instances, so CNN parameters are counted once. The CNN provides spatial and short-context representation; the causal TCN consumes the fused CNN+362-dimensional-numeric sequence for up to 60 valid decision nodes and models long behavior/time dependencies.

For every newly generated joint-training sequence, each of the three board snapshots is independently normalized with its selected source row's recorded `side_to_move` under `board_perspective=snapshot_side_to_move_v1`. Thus each context's X plane is that historical snapshot's mover, not necessarily the current prediction player. Passes are handled from recorded sides, never ply parity. Existing fixed-color model-ready files and checkpoints remain `legacy_fixed_color` artifacts and cannot be mixed with this plan.

Stage A (2–3 epochs) freezes the whole transferred CNN while training the inherited TCN, numerical branch, board projection, FiLM, and formal heads. Stage B unfreezes the 64→96 projection, final LayerNorm, and configured last residual blocks, with CNN learning rate around 0.1 times the other modules. Stage C unfreezes the full CNN, initially using `1e-5` for CNN, `1e-4` for TCN/fusion, and `1e-4` for formal heads.

Joint stages retain:

`total_loss = existing_model_loss + 0.10 * legal_move_loss + 0.10 * value_loss`

For cost control, `auxiliary_nodes_per_game` defaults to 2 (1 is also supported), and only that many randomly selected valid nodes per game receive the second auxiliary forward. This forward creates a 23-plane tensor with channels 0–2 unchanged and channels 3–22 strictly zero before the shared CNN. Consequently the auxiliary heads cannot observe previous boards, prior moves, `current_hint_tokens`, `current_hint_values`, other hint planes, or numeric features. The auxiliary heads are training-only and do not change formal inference outputs.

Compared with the old approximately 106k-parameter three-layer CNN, this 64×6 default is expected to require roughly four times the convolution work, materially below the roughly 8–9× cost of a 96×6 trunk. Batch size, mixed precision, and gradient accumulation are configurable. Smoke uses a small CPU batch. `board-cnn-benchmark` is deliberately bounded and reports forward/training-step milliseconds, samples/second, and CUDA peak allocated memory when run on a GPU; its synthetic 8×8 inputs avoid full-data preprocessing.

## Formal execution order

1. Complete one shared-CNN pretraining run on the already materialized LV17 shards.
2. Train one seed of the CNN+TCN joint model on the fixed 11,200-game split.
3. Compare that single model with the current baseline using unchanged validation gates.
4. Consider a 12-member ensemble only if the single transferred model demonstrates an improvement.

This implementation and validation work does not run any of those formal jobs.

## Evaluation and promotion gates

Compare the existing formal selection metric plus all thinking-time, severity four-class, and WLD three-class metrics. Break results down by joint-training stage and ply. Choose the final candidate from validation only, then open the unchanged formal test set once.

Promotion requires: rule tests for opening, ordinary positions, pass, and terminal boards; CPU forward/backward/save/resume smoke; strict 3→23 transfer with only channels 0–2 nonzero; unchanged default behavior without an explicit transferred checkpoint; and a recorded baseline-versus-experiment report on the identical split.

## Command drafts

Run these from `research/tcn_loss_model`. Materialization and formal training are intentionally separate. On Windows, the preferred full materializer is the compiled C++17 program; it uses 16 file-level worker threads and writes the same shard contract consumed by Python training.

```powershell
# Build with the locally installed Visual Studio 2022 Build Tools x64 environment.
cmd /d /s /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cl /nologo /std:c++17 /O2 /EHsc /utf-8 /W4 scripts\data\materialize_board_cnn_pretrain_data.cpp /Fe:tools\board_cnn_materializer\board_cnn_materializer.exe /Fo:tools\board_cnn_materializer\board_cnn_materializer.obj /link bcrypt.lib'

tools\board_cnn_materializer\board_cnn_materializer.exe `
  --source-dir data\egaroucid_7_5_1_lv17 `
  --output-dir data\board_cnn_pretrain_shards `
  --threads 16 --first-index 0 --last-index 25

# Bounded Python alternative for smoke/debug only.
python scripts/data/materialize_board_cnn_pretrain_data.py --config config/board_cnn_pretrain.json --split all --max-samples 64
python train.py board-cnn-pretrain-smoke --config config/board_cnn_pretrain.json --output-dir outputs/board_cnn_pretrain_smoke_<run-id>
python train.py board-cnn-benchmark --config config/board_cnn_pretrain.json --device cpu --batch-size 8 --steps 5
python train.py board-cnn-pretrain --config config/board_cnn_pretrain.json --output-dir outputs/board_cnn_pretrain_<run-id> --confirm-full-pretraining
python scripts/model/transfer_board_cnn_pretrain.py --pretrained-checkpoint <best-board-cnn.pt> --target-checkpoint <current-main-checkpoint.pt> --output-checkpoint <transferred-main-checkpoint.pt>
```

The joint-training command is deliberately not implemented yet. After validation of the transferred checkpoint, add an explicit joint-training entry that consumes `config/cnn_tcn_joint_training_draft.json`; do not repurpose the current default `train` command or silently activate transfer when no pretrained checkpoint is supplied.
