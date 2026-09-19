#!/usr/bin/env python3
"""Rebuild only board contexts from retained authoritative snapshot rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.board_perspective import (  # noqa: E402
    BOARD_PERSPECTIVE,
    encode_fixed_color_board,
    make_board_context_token_sequences,
)


FEATURE_REVISION = "board_context_snapshot_side_to_move_v1__23_planes_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if np.issubdtype(left.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--context-metadata", required=True, type=Path)
    parser.add_argument("--games", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--expected-games", required=True, type=int)
    parser.add_argument("--confirm-legacy-fixed-color-source", action="store_true")
    args = parser.parse_args()
    if not args.confirm_legacy_fixed_color_source:
        raise SystemExit("refusing conversion without --confirm-legacy-fixed-color-source")

    source_path = args.input.resolve()
    context_path = args.context_metadata.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    output_path = output_dir / args.output_name
    partial_path = output_dir / f"{args.output_name}.partial.npz"
    started = time.time()

    print("reading authoritative snapshot rows", flush=True)
    frame = pd.read_csv(
        context_path,
        usecols=["game_id", "move_index", "side_to_move", "board"],
        dtype={"game_id": str, "move_index": np.int64, "side_to_move": str, "board": str},
        encoding="utf-8-sig",
    )
    if frame.duplicated(["game_id", "move_index"]).any():
        raise ValueError("context metadata contains duplicate (game_id, move_index) rows")

    print("loading model-ready source contract", flush=True)
    with np.load(source_path, allow_pickle=False) as source:
        if "board_perspective" in source.files:
            raise ValueError("source already has an explicit board perspective; expected legacy input")
        game_ids = source["game_id"].astype(str)
        if len(game_ids) != args.expected_games or len(set(game_ids)) != len(game_ids):
            raise ValueError("source game membership differs from the expected conversion contract")
        order = {game_id: index for index, game_id in enumerate(game_ids)}
        if set(frame["game_id"].astype(str)) != set(game_ids):
            raise ValueError("context metadata and model-ready NPZ game memberships differ")
        frame["_game_order"] = frame["game_id"].map(order)
        frame.sort_values(["_game_order", "move_index"], kind="stable", inplace=True)
        row_views = {
            str(game_id): view.reset_index(drop=True)
            for game_id, view in frame.groupby("game_id", sort=False)
        }
        source_move_index = source["move_index"]
        source_side = source["side_to_move"].astype(str)
        for game_index, game_id in enumerate(game_ids):
            view = row_views[game_id]
            length = len(view)
            if not np.array_equal(
                source_move_index[game_index, :length], view["move_index"].to_numpy()
            ):
                raise ValueError(f"NPZ/context move_index alignment differs for game {game_id!r}")
            if not np.array_equal(
                source_side[game_index, :length], view["side_to_move"].astype(str).to_numpy()
            ):
                raise ValueError(f"NPZ/context side_to_move alignment differs for game {game_id!r}")

        print("rebuilding snapshot-side board contexts", flush=True)
        board_tokens, sources = make_board_context_token_sequences(
            frame, game_ids, return_source_positions=True
        )
        old_tokens = source["board_tokens"]
        if board_tokens.shape != old_tokens.shape:
            raise ValueError(
                f"rebuilt board shape differs: {board_tokens.shape} != {old_tokens.shape}"
            )

        print("auditing every legacy token against its authoritative source row", flush=True)
        expected_legacy = np.zeros_like(board_tokens)
        for game_index, game_id in enumerate(game_ids):
            view = row_views[game_id]
            fixed_boards = [encode_fixed_color_board(board) for board in view["board"]]
            for position in range(len(view)):
                for context in range(3):
                    source_position = int(sources[game_index, position, context])
                    if source_position >= 0:
                        expected_legacy[game_index, position, context] = fixed_boards[source_position]
        if not np.array_equal(old_tokens, expected_legacy):
            mismatch = np.argwhere(old_tokens != expected_legacy)[0].tolist()
            raise ValueError(f"legacy board tokens do not match retained source rows at {mismatch}")

        changed = int(np.count_nonzero(board_tokens != old_tokens))
        if changed <= 0:
            raise ValueError("conversion changed no board tokens")
        arrays = {name: source[name].copy() for name in source.files if name != "board_tokens"}
        arrays["board_tokens"] = board_tokens
        arrays["board_perspective"] = np.asarray(BOARD_PERSPECTIVE)
        arrays["board_feature_revision"] = np.asarray(FEATURE_REVISION)

    print("writing versioned model-ready NPZ", flush=True)
    np.savez_compressed(partial_path, **arrays)
    os.replace(partial_path, output_path)
    copied_context = output_dir / "position_context_metadata.csv"
    shutil.copy2(context_path, copied_context)
    copied_games = None
    if args.games is not None:
        copied_games = output_dir / "games.csv"
        shutil.copy2(args.games.resolve(), copied_games)

    with np.load(output_path, allow_pickle=False) as result:
        if str(result["board_perspective"].item()) != BOARD_PERSPECTIVE:
            raise ValueError("written board perspective contract is invalid")
        if not np.array_equal(result["board_tokens"], board_tokens):
            raise ValueError("written board tokens differ from the verified conversion")
        for name in arrays:
            if name in {"board_tokens", "board_perspective", "board_feature_revision"}:
                continue
            if not arrays_equal(result[name], arrays[name]):
                raise ValueError(f"unrelated array changed during conversion: {name}")

    manifest = {
        "schema": "legacy-board-context-perspective-conversion-v1",
        "ok": True,
        "sourceBoardPerspective": "legacy_fixed_color",
        "boardPerspective": BOARD_PERSPECTIVE,
        "boardFeatureRevision": FEATURE_REVISION,
        "games": int(len(game_ids)),
        "contextRows": int(len(frame)),
        "changedBoardTokenCells": changed,
        "allLegacyBoardTokensMatchedAuthoritativeRows": True,
        "allUnrelatedArraysPreserved": True,
        "sourceData": str(source_path),
        "sourceDataSha256": sha256_file(source_path),
        "contextMetadata": str(context_path),
        "contextMetadataSha256": sha256_file(context_path),
        "outputData": str(output_path),
        "outputDataSha256": sha256_file(output_path),
        "copiedGames": str(copied_games) if copied_games else None,
        "trainingStarted": False,
        "elapsedSeconds": time.time() - started,
    }
    (output_dir / "perspective_conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
