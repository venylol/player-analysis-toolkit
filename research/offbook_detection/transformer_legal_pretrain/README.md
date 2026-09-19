# Transformer legal-move pretraining data

The base dataset is the retained local LV17 board dataset in the CNN archive. It
contains 24,000,000 training positions, 1,000,000 validation positions, and
514,097 fixed-test positions. The archive is read-only for this research line.

`data/pass_supplement_11200_v1.npz` adds the 10,998 real pass positions from the
existing 11,200-game OQ dataset. The old model-ready NPZ deliberately excludes
pass rows, so the extractor reconstructs each pass from an original `move_index`
gap and the authoritative board before the following placement. Each board is
normalized so `X` is the passing side. Rule checks require that the passing side
has no legal move and that the following placement side has at least one.

Game-level train/validation/test assignments are inherited from the retained
11,200-game NPZ. The supplement contains 8,755 training, 1,121 validation, and
1,122 test records. `data_config.json` defines the combined logical dataset;
base archive files are never rewritten.

The supplement was independently rechecked with the retained C++ bitboard rules
implementation in `materialize_board_cnn_pretrain_data.cpp` and its compiled
`board_cnn_materializer.exe`. For each record, the fixed-color board was normalized
so the recorded passing side is `X`, which is the C++ tool's current-player
contract. All 10,998 C++ legal-move bitboards were zero and every packed board
round-tripped unchanged. The hashes and output are recorded under
`data/cpp_pass_verification_v1/verification_manifest.json`.

All source and derived data is local-only and must not be redistributed,
committed, or included in release artifacts.

The frozen first-run model uses a 96-wide, two-layer, four-head spatial
Transformer with a 384-wide feed-forward block. CUDA benchmarking on the local
RTX 4060 Laptop GPU selected batch size 512. Each formal training batch contains
10 sampled real-pass records (1.953%) and 502 base records; validation and test
retain their natural distributions.
