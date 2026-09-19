#!/usr/bin/env python3
"""Build a compact replay-review bundle from review packets and detector output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MOVE_RE = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)
SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-packet",
        type=Path,
        action="append",
        required=True,
        help="player-offbook-agent-review-packet-v1 JSON; repeat for multiple players",
    )
    parser.add_argument("--algorithm-summary", type=Path, required=True)
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        help="optional manual-offbook cohort manifest used to add the frozen initial clock",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(account: Any, game_id: Any) -> tuple[str, str]:
    return str(account or "").strip().casefold(), str(game_id or "").strip()


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def optional_int(value: Any) -> int | None:
    result = optional_float(value)
    if result is None:
        return None
    integer = int(result)
    if abs(result - integer) > 1e-9:
        raise ValueError(f"expected integer-compatible value, got {value!r}")
    return integer


def csv_bool(value: Any) -> bool:
    normalized = str(value or "").strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid CSV boolean: {value!r}")


def parse_scales(value: Any) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(token) for token in str(value).split(";") if token.strip()]


def resolve_repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def load_initial_time_limits(
    cohort_manifest_path: Path | None,
) -> dict[tuple[str, str], float]:
    if cohort_manifest_path is None:
        return {}
    manifest = read_json(cohort_manifest_path)
    if manifest.get("schema") != "manual-offbook-time-baseline-cohort-v1":
        raise ValueError("unexpected cohort manifest schema")
    result: dict[tuple[str, str], float] = {}
    for cohort in manifest.get("cohorts", []):
        account = str(cohort.get("account") or "").strip()
        data_path = resolve_repository_path(cohort["data"])
        with np.load(data_path, allow_pickle=False) as data:
            game_ids = data["game_id"].astype(str)
            limits = data["source_time_limit_ms"].astype(float)
            if limits.shape != game_ids.shape:
                raise ValueError(f"initial-clock shape mismatch: {data_path}")
            for game_id, limit in zip(game_ids, limits, strict=True):
                row_key = key(account, game_id)
                if row_key in result:
                    raise ValueError(f"duplicate cohort player/game key: {row_key}")
                if not math.isfinite(float(limit)) or float(limit) <= 0:
                    raise ValueError(f"invalid initial time limit for {row_key}: {limit}")
                result[row_key] = float(limit)
    return result


def load_algorithm_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "account",
            "game_id",
            "judgment",
            "manual_offbook_ply",
            "has_converged_candidate",
            "candidate_strict_ply",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"algorithm summary lacks columns: {sorted(missing)}")
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for row in reader:
            row_key = key(row["account"], row["game_id"])
            if not row_key[0] or not row_key[1] or row_key in result:
                raise ValueError(f"empty or duplicate algorithm player/game key: {row_key}")
            has_candidate = csv_bool(row.get("has_converged_candidate"))
            candidate_ply = optional_int(row.get("candidate_strict_ply"))
            if has_candidate != (candidate_ply is not None):
                raise ValueError(f"candidate flag/ply mismatch for {row_key}")
            result[row_key] = {
                "manualReview": {
                    "judgment": str(row.get("judgment") or ""),
                    "offbookPly": optional_int(row.get("manual_offbook_ply")),
                    "targetDecisionNumber": optional_int(
                        row.get("manual_offbook_target_decision")
                    ),
                },
                "algorithm": {
                    "hasCandidate": has_candidate,
                    "candidatePly": candidate_ply,
                    "candidateTargetDecision": optional_int(
                        row.get("candidate_target_decision")
                    ),
                    "confidence": str(row.get("candidate_confidence") or "") or None,
                    "supportCount": optional_int(row.get("candidate_support_count")),
                    "supportingScales": parse_scales(row.get("candidate_supporting_scales")),
                    "supportingDecisions": str(
                        row.get("candidate_supporting_decisions") or ""
                    ) or None,
                    "convergenceScore": optional_float(
                        row.get("candidate_convergence_score")
                    ),
                    "foldThreshold": optional_float(row.get("fold_convergence_threshold")),
                    "scaleCandidateCount": optional_int(row.get("scale_candidate_count")),
                    "convergenceClusterCount": optional_int(
                        row.get("convergence_cluster_count")
                    ),
                },
            }
    if not result:
        raise ValueError("algorithm summary is empty")
    return result


def compact_ply(raw: dict[str, Any], expected_ply: int, game_key: tuple[str, str]) -> dict[str, Any]:
    ply = optional_int(raw.get("ply"))
    move = str(raw.get("move") or "").strip().lower()
    color = str(raw.get("playerColor") or "").strip().lower()
    if ply != expected_ply:
        raise ValueError(f"non-consecutive strict ply for {game_key}: {ply} != {expected_ply}")
    if not MOVE_RE.fullmatch(move):
        raise ValueError(f"invalid move for {game_key} ply {ply}: {move!r}")
    if color not in {"black", "white"}:
        raise ValueError(f"invalid player color for {game_key} ply {ply}: {color!r}")
    thinking_time = optional_float(raw.get("thinkingTimeMs"))
    if thinking_time is not None and thinking_time < 0:
        raise ValueError(f"negative thinking time for {game_key} ply {ply}")
    return {
        "ply": ply,
        "sourceMoveIndex": optional_int(raw.get("sourceMoveIndex")),
        "move": move,
        "playerColor": color,
        "playerAccount": str(raw.get("playerAccount") or ""),
        "isTargetMove": bool(raw.get("isTargetMove")),
        "targetDecisionNumber": optional_int(raw.get("targetDecisionNumber")),
        "thinkingTimeMs": thinking_time,
    }


def build_bundle(
    packet_paths: list[Path],
    algorithm_path: Path,
    cohort_manifest_path: Path | None = None,
) -> dict[str, Any]:
    algorithm_rows = load_algorithm_rows(algorithm_path)
    initial_time_limits = load_initial_time_limits(cohort_manifest_path)
    games: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    packet_sources: list[dict[str, Any]] = []
    for packet_path in packet_paths:
        packet = read_json(packet_path)
        if packet.get("schema") != "player-offbook-agent-review-packet-v1":
            raise ValueError(f"unexpected review packet schema: {packet_path}")
        account = str(packet.get("account") or "").strip()
        if not account:
            raise ValueError(f"review packet lacks account: {packet_path}")
        packet_sources.append({
            "path": str(packet_path),
            "sha256": sha256_file(packet_path),
            "account": account,
            "games": int(packet.get("gameCount") or 0),
        })
        for raw_game in packet.get("games", []):
            game_id = str(raw_game.get("gameId") or "").strip()
            game_key = key(account, game_id)
            if not game_id or game_key in seen:
                raise ValueError(f"empty or duplicate review player/game key: {game_key}")
            seen.add(game_key)
            algorithm = algorithm_rows.get(game_key)
            if algorithm is None:
                raise ValueError(f"algorithm summary lacks review game: {game_key}")
            raw_plies = raw_game.get("plies")
            if not isinstance(raw_plies, list) or not raw_plies:
                raise ValueError(f"review game has no plies: {game_key}")
            plies = [
                compact_ply(raw, expected_ply, game_key)
                for expected_ply, raw in enumerate(raw_plies, start=1)
            ]
            side_time_summary: dict[str, dict[str, Any]] = {}
            initial_time_limit = initial_time_limits.get(game_key)
            for color in ("black", "white"):
                side_plies = [ply for ply in plies if ply["playerColor"] == color]
                timed = [
                    float(ply["thinkingTimeMs"])
                    for ply in side_plies
                    if ply["thinkingTimeMs"] is not None
                ]
                side_time_summary[color] = {
                    "initialTimeLimitMs": initial_time_limit,
                    "summedThinkingTimeMs": float(sum(timed)),
                    "moveCount": len(side_plies),
                    "timedMoveCount": len(timed),
                }
            games.append({
                "account": account,
                "gameId": game_id,
                "created": raw_game.get("created"),
                "targetColor": str(raw_game.get("targetColor") or "").lower(),
                "opponentAccount": str(raw_game.get("opponentAccount") or ""),
                "actualMoveCount": len(plies),
                "targetMoveCount": int(raw_game.get("targetMoveCount") or 0),
                "sideTimeSummary": side_time_summary,
                "manualReview": algorithm["manualReview"],
                "algorithm": algorithm["algorithm"],
                "plies": plies,
            })
    games.sort(key=lambda game: (
        str(game["account"]).casefold(),
        str(game.get("created") or ""),
        str(game["gameId"]),
    ))
    return {
        "schema": "offbook-replay-review-bundle-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Offline real-time replay review of algorithmic candidate off-book anchors; "
            "manual anchors remain comparison metadata only"
        ),
        "timePolicy": "thinkingTimeMs is original OQ per-move milliseconds; default playback is 1x",
        "counts": {
            "players": len({str(game["account"]).casefold() for game in games}),
            "playerGameEvaluations": len(games),
            "algorithmCandidates": sum(
                bool(game["algorithm"]["hasCandidate"]) for game in games
            ),
        },
        "sources": {
            "reviewPackets": packet_sources,
            "algorithmSummary": {
                "path": str(algorithm_path),
                "sha256": sha256_file(algorithm_path),
            },
            "cohortManifest": (
                {
                    "path": str(cohort_manifest_path),
                    "sha256": sha256_file(cohort_manifest_path),
                }
                if cohort_manifest_path is not None
                else None
            ),
        },
        "games": games,
    }


def main() -> int:
    args = parse_args()
    packet_paths = [path.resolve() for path in args.review_packet]
    algorithm_path = args.algorithm_summary.resolve()
    cohort_manifest_path = args.cohort_manifest.resolve() if args.cohort_manifest else None
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    bundle = build_bundle(packet_paths, algorithm_path, cohort_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "completed",
        "output": str(output_path),
        "counts": bundle["counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
