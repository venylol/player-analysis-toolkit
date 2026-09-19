#!/usr/bin/env python3
"""Validate and remap the Reference200 Level22 cache into Reference400."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MOVE_RE = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def load_bundle(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    details = bundle.get("details") if isinstance(bundle.get("details"), list) else []
    by_id = {str(row.get("id") or ""): row for row in details}
    tables = {str(row.get("id") or ""): index for index, row in enumerate(details, start=1)}
    if not details or "" in by_id or len(by_id) != len(details):
        raise ValueError(f"bundle does not contain unique game IDs: {path}")
    return by_id, tables


def validate_game(engine_game: dict[str, Any], detail: dict[str, Any], engine_sha256: str) -> None:
    game_id = str(detail.get("id") or "")
    if str(engine_game.get("gameId") or "") != game_id:
        raise ValueError(f"engine game ID mismatch: {game_id}")
    source_events = (detail.get("position") or {}).get("moves") or []
    source_moves = [
        str(event.get("m") or "").lower()
        for event in source_events
        if isinstance(event, dict) and MOVE_RE.fullmatch(str(event.get("m") or ""))
    ]
    nodes = engine_game.get("nodes") if isinstance(engine_game.get("nodes"), list) else []
    events = engine_game.get("events") if isinstance(engine_game.get("events"), list) else []
    if len(nodes) != len(source_moves) or int(engine_game.get("moveCount") or -1) != len(source_moves):
        raise ValueError(f"engine move count mismatch: {game_id}")
    if len(events) != len(source_events):
        raise ValueError(f"engine source event count mismatch: {game_id}")
    for ply, (node, move) in enumerate(zip(nodes, source_moves, strict=True), start=1):
        if int(node.get("ply") or 0) != ply or str(node.get("move") or "").lower() != move:
            raise ValueError(f"engine move provenance mismatch: {game_id} ply {ply}")
        if not isinstance(node.get("lossClipped"), (int, float)):
            raise ValueError(f"engine loss is incomplete: {game_id} ply {ply}")
    for index, (source, event) in enumerate(zip(source_events, events, strict=True)):
        if int(event.get("sourceMoveIndex", -1)) != index or event.get("thinkingTimeMs") != source.get("t"):
            raise ValueError(f"engine time provenance mismatch: {game_id} event {index}")
    engine = engine_game.get("engine") if isinstance(engine_game.get("engine"), dict) else {}
    expected = {
        "sha256": engine_sha256.casefold(), "level": 22, "threads": 16,
        "hash": 25, "book": "enabled-default",
    }
    actual = {key: str(engine.get(key)).casefold() if key == "sha256" else engine.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"engine contract mismatch: {game_id}: {actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-reference", type=Path, required=True)
    parser.add_argument("--new-reference", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()
    old_reference = args.old_reference.resolve()
    new_reference = args.new_reference.resolve()
    engine_path = args.engine.resolve()
    old_details, _ = load_bundle(old_reference / "selected_account_bundle.json")
    new_details, new_tables = load_bundle(new_reference / "selected_account_bundle.json")
    if not set(old_details) <= set(new_details):
        raise ValueError("new Reference does not retain every old Reference game")
    old_index_payload = json.loads((old_reference / "engine_game_index.json").read_text(encoding="utf-8"))
    old_index = {str(row.get("gameId") or ""): row for row in old_index_payload.get("games") or []}
    if set(old_index) != set(old_details):
        raise ValueError("old engine index does not exactly cover the old Reference")
    engine_sha256 = sha256_file(engine_path)
    output = new_reference / "engine_level22"
    output.mkdir(parents=True, exist_ok=True)
    remapped = []
    for game_id in sorted(old_details):
        if old_details[game_id] != new_details[game_id]:
            raise ValueError(f"retained source detail changed: {game_id}")
        source_row = old_index[game_id]
        source_path = old_reference / str(source_row["engineFile"])
        if sha256_file(source_path).casefold() != str(source_row["engineFileSha256"]).casefold():
            raise ValueError(f"old engine index SHA-256 mismatch: {game_id}")
        engine_game = json.loads(source_path.read_text(encoding="utf-8"))
        validate_game(engine_game, old_details[game_id], engine_sha256)
        old_table = int(engine_game.get("table") or 0)
        new_table = new_tables[game_id]
        engine_game["table"] = new_table
        target_path = output / f"game_0_{new_table}_{game_id}.json"
        atomic_json(target_path, engine_game)
        remapped.append({
            "gameId": game_id,
            "sourceDetailCanonicalSha256": canonical_sha256(old_details[game_id]),
            "oldTable": old_table, "newTable": new_table,
            "sourceEngineFile": str(source_path),
            "sourceEngineFileSha256": str(source_row["engineFileSha256"]),
            "targetEngineFile": str(target_path.relative_to(new_reference)).replace("\\", "/"),
            "targetEngineFileSha256": sha256_file(target_path),
        })
    manifest = {
        "schema": "oq-reference400-level22-cache-remap-v1",
        "ok": True, "createdAtUtc": utc_now(),
        "oldReference": str(old_reference), "newReference": str(new_reference),
        "enginePath": str(engine_path), "engineSha256": engine_sha256,
        "contract": {"level": 22, "threads": 16, "hash": 25, "book": "enabled-default"},
        "sourceGameCount": len(old_details), "remappedGameCount": len(remapped),
        "checks": {
            "allOldGamesRetained": True,
            "sourceDetailsIdentical": True,
            "oldEngineIndexHashesVerified": True,
            "moveAndTimeProvenanceVerified": True,
            "engineContractVerified": True,
        },
        "games": remapped,
    }
    atomic_json(output / "cache_remap_manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "games"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
