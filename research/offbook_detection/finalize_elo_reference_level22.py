#!/usr/bin/env python3
"""Independently verify an Elo reference Level22 run and hash every artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MOVE_RE = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--engine-dir",
        type=Path,
        help="audited engine directory; defaults to <reference-dir>/engine_level22",
    )
    args = parser.parse_args()
    root = args.reference_dir.resolve()
    bundle_path = root / "selected_account_bundle.json"
    engine_dir = args.engine_dir.resolve() if args.engine_dir is not None else root / "engine_level22"
    bundle = read_json(bundle_path)
    details = bundle.get("details") if isinstance(bundle.get("details"), list) else []
    expected = {str(detail.get("id") or ""): detail for detail in details}
    if not expected or len(expected) != len(details):
        raise ValueError("source bundle game IDs are empty or duplicated")

    game_paths = sorted(engine_dir.glob("game_*.json"))
    games = [read_json(path) for path in game_paths]
    actual = {str(game.get("gameId") or ""): game for game in games}
    if len(actual) != len(games) or set(actual) != set(expected):
        raise ValueError("game_*.json IDs do not exactly match the source bundle")

    audit = read_json(engine_dir / "audit.json")
    summary = read_json(engine_dir / "summary.json")
    progress = read_json(engine_dir / "progress.json")
    required_contract = {
        "ok": True,
        "gameCount": len(expected),
        "workerCount": 12,
        "threadsPerConsole": 16,
        "engineLevel": 22,
        "hashLevel": 25,
        "book": "enabled-default",
    }
    for key, value in required_contract.items():
        if audit.get(key) != value:
            raise ValueError(f"audit contract mismatch: {key}")
    if summary.get("gameCount") != len(expected) or summary.get("workerCount") != 12:
        raise ValueError("summary game/worker count mismatch")
    if summary.get("threadsPerConsole") != 16:
        raise ValueError("summary threads mismatch")
    progress_complete = (
        progress.get("completedCount") == len(expected)
        and progress.get("totalCount") == len(expected)
        and set(progress.get("completedGameIds") or []) == set(expected)
    )

    source_sha256 = sha256_file(bundle_path)
    if str(audit.get("sourceBundleSha256") or "").casefold() != source_sha256.casefold():
        raise ValueError("source bundle SHA-256 mismatch")
    engine_path = Path(str(games[0]["engine"]["path"]))
    engine_sha256 = sha256_file(engine_path)
    if str(audit.get("engineSha256") or "").casefold() != engine_sha256.casefold():
        raise ValueError("engine SHA-256 mismatch")

    node_count = 0
    event_count = 0
    for game_id, detail in expected.items():
        game = actual[game_id]
        source_events = (detail.get("position") or {}).get("moves") or []
        coordinate_sources = [
            (index, event) for index, event in enumerate(source_events)
            if isinstance(event, dict) and MOVE_RE.fullmatch(str(event.get("m") or ""))
        ]
        nodes = game.get("nodes") if isinstance(game.get("nodes"), list) else []
        events = game.get("events") if isinstance(game.get("events"), list) else []
        if len(nodes) != len(coordinate_sources) or len(events) != len(source_events):
            raise ValueError(f"source/result length mismatch for {game_id}")
        for ply, (node, (source_index, source)) in enumerate(zip(nodes, coordinate_sources, strict=True), start=1):
            if node.get("ply") != ply:
                raise ValueError(f"non-continuous global actual ply for {game_id}:{ply}")
            if node.get("sourceMoveIndex") != source_index:
                raise ValueError(f"sourceMoveIndex mismatch for {game_id}:{ply}")
            if str(node.get("move") or "").casefold() != str(source.get("m") or "").casefold():
                raise ValueError(f"move mismatch for {game_id}:{ply}")
            if node.get("thinkingTimeMs") != source.get("t"):
                raise ValueError(f"thinkingTimeMs mismatch for {game_id}:{ply}")
            for field in ("bestEval", "actualEval", "lossClipped"):
                if not finite_number(node.get(field)):
                    raise ValueError(f"invalid {field} for {game_id}:{ply}")
            contract = game.get("engine") or {}
            if (
                contract.get("level") != 22 or contract.get("threads") != 16
                or contract.get("hash") != 25 or contract.get("book") != "enabled-default"
                or str(contract.get("sha256") or "").casefold() != engine_sha256.casefold()
            ):
                raise ValueError(f"per-game engine contract mismatch for {game_id}")
        for source_index, (source, event) in enumerate(zip(source_events, events, strict=True)):
            if event.get("sourceMoveIndex") != source_index or event.get("thinkingTimeMs") != source.get("t"):
                raise ValueError(f"event provenance mismatch for {game_id}:{source_index}")
        node_count += len(nodes)
        event_count += len(events)

    if audit.get("nodeCount") != node_count or audit.get("eventCount") != event_count:
        raise ValueError("independent node/event totals disagree with audit.json")

    wld = read_json(engine_dir / "engine_wld_loss_totals_from_ply39.json")
    if wld.get("wldFromPly") != 39:
        raise ValueError("WLD boundary is not inclusive ply 39")
    wld_rows = wld.get("gamePlayerTotals") if isinstance(wld.get("gamePlayerTotals"), list) else []
    wld_game_ids = {str(row.get("game_id") or "") for row in wld_rows}
    if wld_game_ids != set(expected):
        raise ValueError("WLD output does not cover every selected game")
    wld_keys = [(str(row.get("game_id") or ""), str(row.get("side") or "")) for row in wld_rows]
    if len(wld_keys) != len(set(wld_keys)):
        raise ValueError("WLD game/side keys are duplicated")

    completion = {
        "schema": "oq-reference-level22-completion-audit-v1",
        "ok": True,
        "verifiedAtUtc": utc_now(),
        "gameCount": len(expected),
        "gameFileCount": len(game_paths),
        "nodeCount": node_count,
        "eventCount": event_count,
        "wldGameCount": len(wld_game_ids),
        "engineDirectory": str(engine_dir),
        "progressSnapshot": {
            "complete": progress_complete,
            "completedCount": progress.get("completedCount"),
            "totalCount": progress.get("totalCount"),
            "updatedAt": progress.get("updatedAt"),
            "authority": (
                "complete" if progress_complete else
                "stale-non-authoritative-after-interrupted-parent; full audit and exact game files are authoritative"
            ),
        },
        "contract": {
            "level": 22, "workers": 12, "threadsPerConsole": 16, "hash": 25,
            "book": "enabled-default", "wldFromPlyInclusive": 39,
            "enginePath": str(engine_path), "engineSha256": engine_sha256,
            "sourceBundle": str(bundle_path), "sourceBundleSha256": source_sha256,
        },
        "checks": [
            "exactly one complete game_*.json per selected gameId",
            "continuous pass-free global actual ply and exact move/sourceMoveIndex/thinkingTimeMs provenance",
            "finite bestEval, actualEval, and lossClipped at every coordinate move",
            "summary, runner audit, exact game files, engine contract, and source hashes agree",
            "WLD ply39 output covers every selected game with unique game/side rows",
        ],
    }
    completion_path = root / "reference_completion_audit.json"
    write_json(completion_path, completion)

    manifest_path = root / "final_sha256_manifest.json"
    files = []
    for path in sorted((item for item in root.rglob("*") if item.is_file() and item != manifest_path), key=str):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    final_manifest = {
        "schema": "oq-reference-final-sha256-manifest-v1",
        "createdAtUtc": utc_now(),
        "referenceDirectory": str(root),
        "fileCount": len(files),
        "files": files,
        "selfHashPolicy": "final_sha256_manifest.json is excluded to avoid a recursive self-hash",
    }
    write_json(manifest_path, final_manifest)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    print(json.dumps({"finalManifest": str(manifest_path), "fileCount": len(files)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
