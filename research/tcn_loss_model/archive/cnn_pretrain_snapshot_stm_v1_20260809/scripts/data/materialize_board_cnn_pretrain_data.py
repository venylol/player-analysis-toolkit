#!/usr/bin/env python3
"""Materialize compact current-board CNN shards without loading TXT files into memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.board_cnn_pretrain import BoardCNNTrainingConfig, materialize_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "board_cnn_pretrain.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    parser.add_argument("--max-samples", type=int, help="maximum records per selected source file")
    args = parser.parse_args()
    cfg = BoardCNNTrainingConfig.load(args.config)
    selected = ("train", "validation", "test") if args.split == "all" else (args.split,)
    names = [name for split in selected for name in cfg.splits[split]]
    report = materialize_files(cfg.data_dir, args.output_dir or cfg.shard_dir, names, args.max_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

