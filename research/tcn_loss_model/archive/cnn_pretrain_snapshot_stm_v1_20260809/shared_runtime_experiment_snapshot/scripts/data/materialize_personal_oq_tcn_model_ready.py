#!/usr/bin/env python3
"""Create a personal TCN bundle from a local OQ account bundle.

The entrypoint is generic: it normalizes source clocks to a requested effective
clock, computes official hint1/hint6 features with Egaroucid, and reuses the
official 362-feature/23-channel materializer.  Whole games listed as reported
are assigned to test; every other game is a personal control game.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.checkpoint import load_checkpoint_payload, sha256_file
from src.board_perspective import BOARD_PERSPECTIVE
from src.data_contract import validate_model_ready_npz
from src.labels import decision_nodes, generate_disc_loss_labels

import materialize_oq_tcn_model_ready as official

POLICY = "linear-clock-normalized-to-300000-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-bundle", type=Path, required=True)
    parser.add_argument("--target-player", required=True)
    parser.add_argument("--reported-game", action="append", required=True)
    parser.add_argument("--offbook-ply", action="append", default=[], help="GAME_ID=GLOBAL_PLACEMENT_PLY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--source-research", type=Path, required=True)
    parser.add_argument("--human-opening-book", type=Path, required=True)
    parser.add_argument("--engine-analyzer-source", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--reuse-engine-hints", type=Path, help="reuse complete board/engine hints and rebuild clock-derived inputs")
    parser.add_argument("--safe-assembled-raw", type=Path, help="audited safe hint assembly; bypasses the legacy engine path")
    parser.add_argument("--safe-assembly-manifest", type=Path, help="manifest paired with --safe-assembled-raw")
    parser.add_argument("--effective-time-limit-ms", type=int, default=300000)
    parser.add_argument("--hint1-level", type=int, default=2)
    parser.add_argument("--level18-workers", type=int, default=4)
    parser.add_argument("--level18-threads", type=int, default=8)
    parser.add_argument("--hash-level", type=int, default=25)
    parser.add_argument("--timeout-hint1", type=float, default=60.0)
    parser.add_argument("--timeout-hint6", type=float, default=600.0)
    parser.add_argument("--feature-workers", type=int, default=12)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("oq_hint_engine", path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_bundle(source: dict[str, Any], effective_limit: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(source.get("details"), list) or not source["details"]:
        raise ValueError("account bundle must contain a non-empty details list")
    normalized = json.loads(json.dumps(source))
    audit_games = []
    by_id = {str(item["id"]): item for item in normalized["details"]}
    source_by_id = {str(item["id"]): item for item in source["details"]}
    for game_id, detail in by_id.items():
        original = source_by_id[game_id]
        source_limit = int(original.get("tcb", 0) or 0)
        if source_limit <= 0:
            raise ValueError(f"game {game_id} has invalid source tcb")
        factor = effective_limit / source_limit
        detail["source_time_limit_ms"] = source_limit
        detail["effective_time_limit_ms"] = effective_limit
        detail["time_scale_factor"] = factor
        detail["time_control_policy"] = POLICY
        detail["tcb"] = effective_limit
        source_moves = (original.get("position") or {}).get("moves") or []
        moves = (detail.get("position") or {}).get("moves") or []
        if len(source_moves) != len(moves):
            raise AssertionError("normalized move list length changed")
        for source_move, move in zip(source_moves, moves, strict=True):
            raw_time = int(source_move.get("t", 0) or 0)
            move["raw_thinking_time_ms"] = raw_time
            move["t"] = raw_time * factor
            if "delay" in source_move:
                raw_delay = int(source_move.get("delay", 0) or 0)
                move["raw_delay_ms"] = raw_delay
                move["delay"] = raw_delay * factor
        audit_games.append({
            "gameId": game_id, "sourceTimeLimitMs": source_limit,
            "effectiveTimeLimitMs": effective_limit, "timeScaleFactor": factor,
            "moves": len(moves),
        })
    for item in normalized.get("index", []):
        game_id = str(item.get("id", ""))
        if game_id in by_id:
            item["source_time_limit_ms"] = int(source_by_id[game_id]["tcb"])
            item["effective_time_limit_ms"] = effective_limit
            item["time_scale_factor"] = effective_limit / int(source_by_id[game_id]["tcb"])
            item["time_control_policy"] = POLICY
            item["tcb"] = effective_limit
    audit = {
        "schema": "personal-oq-clock-normalization-v1", "policy": POLICY,
        "effectiveTimeLimitMs": effective_limit, "games": audit_games,
    }
    return normalized, audit


def game_summary(detail: dict[str, Any], index_by_id: dict[str, Any]) -> dict[str, Any]:
    game_id = str(detail["id"])
    index = index_by_id.get(game_id, {})
    moves = [move for move in (detail.get("position") or {}).get("moves", []) if "m" in move]
    final_status = str(index.get("finalStatus") or "")
    if not final_status and moves:
        final_status = str(moves[-1].get("s") or "")
    players = detail.get("players") or []
    return {
        "game_id": game_id, "mode": "reversi_5min", "gtype": "reversi",
        "tcb": int(detail["tcb"]), "created": detail.get("created", ""),
        "finalStatus": final_status, "length": len(moves),
        "black_id": str(players[0].get("id", "")).lower() if len(players) > 0 else "",
        "white_id": str(players[1].get("id", "")).lower() if len(players) > 1 else "",
    }


def engine_fields() -> list[str]:
    fields = [
        "game_id", "mode", "gtype", "tcb", "created", "finalStatus",
        "move_index", "ply", "side_to_move", "player_id", "actual_move",
        "actual_thinking_time_ms", "board", "n_legal_moves", "legal_moves",
        "hint1_level", "hint1_move", "hint1_score", "hint1_nodes", "hint1_depth", "hint1_is_book",
    ]
    for index in range(1, 7):
        fields.extend(f"hint6_{index}_{suffix}" for suffix in ("move", "score", "nodes", "depth", "is_book"))
    return fields + ["analyzed_at"]


def analyze_games(normalized: dict[str, Any], output_dir: Path, engine_source: Path, args: argparse.Namespace) -> Path:
    engine = load_module(engine_source)
    rows_path = output_dir / "position_hints.csv"
    progress_path = output_dir / "engine_progress.json"
    fields = engine_fields()
    completed: set[str] = set()
    if progress_path.is_file():
        completed = set(json.loads(progress_path.read_text(encoding="utf-8")).get("completedGames", []))
    if not rows_path.exists():
        with rows_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    index_by_id = {str(item.get("id")): item for item in normalized.get("index", [])}
    details = normalized["details"]
    engine_args = SimpleNamespace(
        engine=str(args.engine.resolve()), hint1_level=args.hint1_level,
        level6_threads=1, level18_workers=args.level18_workers,
        level18_threads=args.level18_threads, hash_level=args.hash_level,
        _engine_restart_count=0,
    )
    pair = engine.open_engine_pair(engine_args, output_dir)
    try:
        for number, detail in enumerate(details, start=1):
            game = game_summary(detail, index_by_id)
            game_id = game["game_id"]
            if game_id in completed:
                continue
            print(f"personal engine [{number}/{len(details)}] {game_id}", flush=True)
            targets = set(range(int(game["length"])))
            rows, jobs = engine.build_rows_for_game(
                game, detail, targets, pair.hint1, args.hint1_level, args.timeout_hint1
            )
            normalized_moves = [move for move in (detail.get("position") or {}).get("moves", []) if "m" in move]
            for row in rows:
                row["actual_thinking_time_ms"] = float(normalized_moves[int(row["move_index"])].get("t", 0) or 0)
            chunks = [jobs[index::len(pair.l18_workers)] for index in range(len(pair.l18_workers))]

            def run_chunk(worker_index: int) -> list[tuple[int, list[dict[str, Any]]]]:
                worker = pair.l18_workers[worker_index]
                results = []
                for job in chunks[worker_index]:
                    worker.setboard(str(job["board"]))
                    results.append((int(job["row_index"]), worker.hint(6, timeout=args.timeout_hint6)))
                return results

            with ThreadPoolExecutor(max_workers=len(pair.l18_workers)) as pool:
                for worker_results in pool.map(run_chunk, range(len(pair.l18_workers))):
                    for row_index, hints in worker_results:
                        for hint_index in range(1, 7):
                            hint = hints[hint_index - 1] if len(hints) >= hint_index else {}
                            for suffix in ("move", "score", "nodes", "depth", "is_book"):
                                rows[row_index][f"hint6_{hint_index}_{suffix}"] = hint.get(suffix, "")
            with rows_path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore").writerows(rows)
            completed.add(game_id)
            write_json(progress_path, {
                "schema": "personal-oq-engine-progress-v1", "status": "running",
                "completedGames": sorted(completed), "totalGames": len(details),
            })
    finally:
        pair.close()
    write_json(progress_path, {
        "schema": "personal-oq-engine-progress-v1", "status": "completed",
        "completedGames": sorted(completed), "totalGames": len(details),
    })
    return rows_path


def reuse_engine_hints(source_path: Path, normalized: dict[str, Any], output_dir: Path) -> Path:
    """Reuse clock-invariant engine results while replacing exact effective times."""
    frame = pd.read_csv(source_path, low_memory=False, encoding="utf-8")
    detail_by_id = {str(item["id"]): item for item in normalized["details"]}
    effective_times: dict[tuple[str, int], float] = {}
    for game_id, detail in detail_by_id.items():
        moves = [move for move in (detail.get("position") or {}).get("moves", []) if "m" in move]
        for move_index, move in enumerate(moves):
            effective_times[(game_id, move_index)] = float(move.get("t", 0) or 0)
    keys = list(zip(frame["game_id"].astype(str), frame["move_index"].astype(int), strict=True))
    missing = [key for key in keys if key not in effective_times]
    if missing:
        raise ValueError(f"reused engine hints contain unknown game/move keys: {missing[:3]}")
    frame["actual_thinking_time_ms"] = [effective_times[key] for key in keys]
    frame["tcb"] = 300000
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "position_hints.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8")
    write_json(output_dir / "engine_progress.json", {
        "schema": "personal-oq-engine-progress-v1", "status": "completed-reused",
        "source": str(source_path.resolve()), "sourceSha256": sha256_file(source_path),
        "rows": int(len(frame)), "games": int(frame["game_id"].nunique()),
        "reuseScope": "clock-invariant board and hint1/hint6 fields only; exact effective times replaced",
    })
    return output_path


def parse_offbook(items: list[str]) -> dict[str, int]:
    result = {}
    for item in items:
        game_id, separator, ply = item.partition("=")
        if not separator:
            raise ValueError(f"invalid --offbook-ply {item!r}; expected GAME_ID=PLY")
        result[game_id] = int(ply)
    return result


def add_personal_audit_arrays(
    arrays: dict[str, np.ndarray], normalized: dict[str, Any], target_player: str, reported: set[str]
) -> None:
    detail_by_id = {str(item["id"]): item for item in normalized["details"]}
    game_ids = arrays["game_id"].astype(str)
    shape = arrays["mask"].shape
    raw_times = np.zeros(shape, dtype=np.float32)
    source_limit = np.zeros(len(game_ids), dtype=np.int32)
    effective_limit = np.zeros(len(game_ids), dtype=np.int32)
    scale = np.zeros(len(game_ids), dtype=np.float32)
    target = target_player.casefold()
    for game_index, game_id in enumerate(game_ids):
        detail = detail_by_id[game_id]
        source_limit[game_index] = int(detail["source_time_limit_ms"])
        effective_limit[game_index] = int(detail["effective_time_limit_ms"])
        scale[game_index] = float(detail["time_scale_factor"])
        moves = [move for move in (detail.get("position") or {}).get("moves", []) if "m" in move]
        raw_by_index = {index: int(move.get("raw_thinking_time_ms", move.get("t", 0)) or 0) for index, move in enumerate(moves)}
        for step in range(shape[1]):
            move_index = int(arrays["move_index"][game_index, step])
            if move_index >= 0:
                raw_times[game_index, step] = raw_by_index[move_index]
    target_nodes = np.char.lower(arrays["player_id"].astype(str)) == target
    arrays["mask"] = arrays["mask"].astype(bool) & target_nodes
    arrays["split"] = np.asarray(["test" if game_id in reported else "train" for game_id in game_ids], dtype="U10")
    arrays["source_time_limit_ms"] = source_limit
    arrays["effective_time_limit_ms"] = effective_limit
    arrays["time_scale_factor"] = scale
    arrays["raw_thinking_time_ms"] = raw_times
    arrays["effective_thinking_time_ms"] = arrays["actual_thinking_time_ms"].copy()
    arrays["time_control_policy"] = np.asarray(POLICY)
    arrays["target_player_id"] = np.asarray(target_player)


def load_safe_assembled_raw(raw_path: Path, manifest_path: Path, expected_game_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "oq-safe-hint-assembly-v1" or manifest.get("status") != "complete":
        raise ValueError("personal safe assembly manifest is not complete")
    if Path(manifest["outputCsv"]).resolve() != raw_path.resolve():
        raise ValueError("personal safe assembly manifest points to a different CSV")
    if manifest.get("outputSha256") != sha256_file(raw_path):
        raise ValueError("personal safe assembled CSV hash mismatch")
    raw = pd.read_csv(raw_path, low_memory=False, encoding="utf-8", dtype={"game_id": str})
    if raw.duplicated(["game_id", "move_index"]).any():
        raise ValueError("safe assembled personal rows contain duplicate game_id/move_index keys")
    actual_games = set(raw["game_id"].astype(str))
    if actual_games != expected_game_ids:
        raise ValueError("safe assembled personal game IDs differ from the normalized account bundle")
    placements = raw["actual_move"].astype(str).str.strip() != "-"
    passes = ~placements
    shape = (len(raw), int(placements.sum()), int(passes.sum()), len(actual_games))
    manifest_shape = tuple(int(manifest[name]) for name in ("rows", "placements", "passes", "games"))
    if shape != manifest_shape:
        raise ValueError(f"personal safe assembled shape differs from its manifest: {shape} != {manifest_shape}")
    required = {
        "board_setboard", "hint1_request_board_setboard", "hint1_board_setboard",
        "hint1_setboard_response_board_setboard", "hint6_request_board_setboard", "hint6_board_setboard",
        "hint6_setboard_response_board_setboard", "hint1_request_id", "hint6_request_id",
        "hint1_engine_sha256", "hint6_engine_sha256", "hint1_engine_threads", "hint6_engine_threads",
        "hint1_engine_hash_level", "hint6_engine_hash_level", "hint1_use_book", "hint6_use_book",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"personal safe assembled CSV lacks provenance fields: {missing}")
    source_board = raw.loc[placements, "board_setboard"].astype(str)
    for field in (
        "hint1_request_board_setboard", "hint1_board_setboard",
        "hint1_setboard_response_board_setboard", "hint6_request_board_setboard", "hint6_board_setboard",
        "hint6_setboard_response_board_setboard",
    ):
        if not raw.loc[placements, field].astype(str).reset_index(drop=True).equals(source_board.reset_index(drop=True)):
            raise ValueError(f"personal safe provenance board mismatch in {field}")
    for field in ("hint1_request_id", "hint6_request_id"):
        values = raw.loc[placements, field].astype(str).str.strip()
        if values.eq("").any() or values.duplicated().any():
            raise ValueError(f"personal safe provenance has missing or duplicate {field}")
    exact_numeric = {
        "hint1_engine_threads": 1, "hint6_engine_threads": 16,
        "hint1_engine_hash_level": 25, "hint6_engine_hash_level": 25,
    }
    for field, expected in exact_numeric.items():
        values = pd.to_numeric(raw.loc[placements, field], errors="raise")
        if not (values == expected).all():
            raise ValueError(f"personal safe engine contract mismatch in {field}")
    for field, expected in (("hint1_use_book", False), ("hint6_use_book", True)):
        values = raw.loc[placements, field].astype(str).str.casefold()
        if not values.eq(str(expected).casefold()).all():
            raise ValueError(f"personal safe book contract mismatch in {field}")
    if not raw.loc[placements, "hint1_engine_sha256"].astype(str).equals(
        raw.loc[placements, "hint6_engine_sha256"].astype(str)
    ):
        raise ValueError("personal hint1/hint6 used different engine binaries")
    engine_value = raw.loc[placements, "hint1_engine_sha256"].astype(str)
    if engine_value.eq("").any() or engine_value.nunique() != 1:
        raise ValueError("personal safe engine hash is missing or inconsistent")
    for field in (
        "hint1_request_board_setboard", "hint1_board_setboard",
        "hint1_setboard_response_board_setboard", "hint6_request_board_setboard", "hint6_board_setboard",
        "hint6_setboard_response_board_setboard",
        "hint1_request_id", "hint6_request_id",
    ):
        if raw.loc[passes, field].notna().any():
            raise ValueError(f"pass rows retain forbidden safe engine provenance in {field}")
    return raw, manifest


def main() -> int:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.account_bundle.read_text(encoding="utf-8"))
    normalized, normalization_audit = normalize_bundle(source, args.effective_time_limit_ms)
    normalized_path = output / "normalized_account_bundle.json"
    audit_path = output / "clock_normalization_manifest.json"
    if not normalized_path.exists():
        write_json(normalized_path, normalized)
        write_json(audit_path, normalization_audit)
    else:
        existing = json.loads(normalized_path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError("existing normalized bundle differs from requested normalization")
    reported = set(args.reported_game)
    all_ids = {str(item["id"]) for item in normalized["details"]}
    if not reported <= all_ids:
        raise ValueError(f"reported games absent from bundle: {sorted(reported - all_ids)}")
    safe_requested = args.safe_assembled_raw is not None or args.safe_assembly_manifest is not None
    if safe_requested and (args.safe_assembled_raw is None or args.safe_assembly_manifest is None):
        raise ValueError("--safe-assembled-raw and --safe-assembly-manifest are required together")
    if safe_requested and args.reuse_engine_hints is not None:
        raise ValueError("safe assembled input cannot be combined with legacy hint reuse")
    safe_manifest = None
    if safe_requested:
        raw, safe_manifest = load_safe_assembled_raw(args.safe_assembled_raw, args.safe_assembly_manifest, all_ids)
    else:
        if args.engine_analyzer_source is None or args.engine is None:
            raise ValueError("legacy engine mode requires --engine-analyzer-source and --engine")
        engine_dir = output / "engine_analysis"
        engine_dir.mkdir(exist_ok=True)
        rows_path = (
            reuse_engine_hints(args.reuse_engine_hints, normalized, engine_dir)
            if args.reuse_engine_hints else
            analyze_games(normalized, engine_dir, args.engine_analyzer_source, args)
        )
        raw = pd.read_csv(rows_path, low_memory=False, encoding="utf-8")
    if raw.duplicated(["game_id", "move_index"]).any():
        raise ValueError("engine output has duplicate game_id+move_index rows")
    labelled = generate_disc_loss_labels(raw)
    decisions = decision_nodes(labelled).sort_values(["game_id", "move_index"], kind="stable").reset_index(drop=True)
    raw_path = output / "raw_nodes_with_pass.csv"
    decisions_path = output / "decision_feature_source.csv"
    context_path = output / "position_context_metadata.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    raw_columns = list(raw.columns)
    decision_source = decisions[raw_columns].copy()
    decision_source["ply"] = decisions["global_placement_ply"].astype("int64")
    decision_source["source_ply_including_pass"] = decisions["source_ply_including_pass"].astype("int64")
    decision_source["global_placement_ply"] = decisions["global_placement_ply"].astype("int64")
    decision_source["is_pass_record"] = 0
    decision_source.to_csv(decisions_path, index=False, encoding="utf-8")

    split_path = output / "split_manifest.csv"
    pd.DataFrame({
        "game_id": sorted(all_ids),
        "split": ["test" if game_id in reported else "train" for game_id in sorted(all_ids)],
    }).to_csv(split_path, index=False, encoding="utf-8")
    context_builder, board_context, board_tcn, v2, source_scripts = official.load_official_modules(args.source_research)
    checkpoint = load_checkpoint_payload(args.base_checkpoint)
    preprocessing = json.loads(args.preprocessing.read_text(encoding="utf-8"))
    if preprocessing != checkpoint["preprocessing"]:
        raise ValueError("external preprocessing differs from base checkpoint")
    official.build_context_metadata(decisions_path, context_path, context_builder, args.feature_workers)
    official.run_metadata_augmenters(args.source_research, decisions_path, context_path, args.human_opening_book, output)
    frame = official.official_feature_frame(v2, decisions_path, context_path)
    features = official.apply_checkpoint_preprocessing(frame, preprocessing)
    arrays = official.make_model_ready_arrays(
        frame, features, decisions, split_path, preprocessing, checkpoint,
        board_context, board_tcn, args.feature_workers,
    )
    add_personal_audit_arrays(arrays, normalized, args.target_player, reported)
    model_ready = output / "personal_model_ready.npz"
    temporary = model_ready.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(model_ready)
    validation = validate_model_ready_npz(
        model_ready, expected_input_features=checkpoint["input_features"],
        expected_board_channels=checkpoint["board_encoding"]["cnn_channels"],
        expected_preprocessing_sha256=official.canonical_json_hash(checkpoint["preprocessing"]),
        expected_board_perspective=BOARD_PERSPECTIVE,
    )
    offbook = parse_offbook(args.offbook_ply)
    manifest = {
        "schema": "personal-oq-tcn-model-ready-manifest-v1", "ok": True,
        "targetPlayer": args.target_player, "controlGameIds": sorted(all_ids - reported),
        "reportedGameIds": sorted(reported), "reportedExclusionFromPersonalAdapterTraining": True,
        "manualOffbookGlobalPlacementPly": offbook,
        "timeControlPolicy": POLICY, "normalization": normalization_audit,
        "sourceBundle": str(args.account_bundle.resolve()), "sourceBundleSha256": sha256_file(args.account_bundle),
        "modelReady": str(model_ready), "modelReadySha256": sha256_file(model_ready),
        "inputFeatureCount": len(checkpoint["input_features"]),
        "boardCnnChannelCount": len(checkpoint["board_encoding"]["cnn_channels"]),
        "boardPerspective": BOARD_PERSPECTIVE,
        "validation": validation, "officialSourceScripts": source_scripts,
        "safeHintProvenance": {
            "required": bool(safe_requested),
            "verified": bool(safe_requested),
            "assemblyManifest": str(args.safe_assembly_manifest.resolve()) if safe_requested else "",
            "assemblyManifestSha256": sha256_file(args.safe_assembly_manifest) if safe_requested else "",
            "assembly": safe_manifest,
        },
        "elapsedSeconds": round(time.time() - started, 3),
    }
    write_json(output / "personal_model_ready_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
