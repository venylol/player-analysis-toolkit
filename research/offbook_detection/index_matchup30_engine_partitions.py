#!/usr/bin/env python3
"""Join matchup partition metadata to completed Level22 game files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path)
    args = parser.parse_args()
    root = args.reference_dir.resolve()
    engine = args.engine_dir.resolve() if args.engine_dir else root / "engine_level22"

    games_payload = json.loads((root / "selected_games_with_partitions.json").read_text(encoding="utf-8"))
    game_rows = games_payload.get("games") if isinstance(games_payload.get("games"), list) else []
    if not game_rows:
        raise ValueError("partition selection contains no games")
    metadata_by_id = {str(row.get("gameId") or ""): row for row in game_rows}
    if len(metadata_by_id) != len(game_rows):
        raise ValueError("partition metadata game IDs are duplicated")

    engine_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(engine.glob("game_*.json")):
        game = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(game.get("gameId") or "")
        if not game_id or game_id in engine_by_id:
            raise ValueError(f"empty or duplicate engine game ID: {path}")
        engine_by_id[game_id] = (path, game)
    if set(engine_by_id) != set(metadata_by_id):
        raise ValueError("engine files do not exactly match partition metadata")

    index_rows = []
    for game_id, metadata in sorted(metadata_by_id.items(), key=lambda item: int(item[1]["bundleTable"])):
        path, game = engine_by_id[game_id]
        if int(game.get("table") or 0) != int(metadata["bundleTable"]):
            raise ValueError(f"engine table mismatch for {game_id}")
        relative = path.relative_to(root).as_posix()
        if relative != metadata["expectedEngineFile"]:
            raise ValueError(f"expected engine filename mismatch for {game_id}")
        index_rows.append({
            "gameId": game_id,
            "bundleTable": metadata["bundleTable"],
            "sourceKind": metadata["sourceKind"],
            "partitionScope": metadata["partitionScope"],
            "unorderedPartitionKey": metadata["unorderedPartitionKey"],
            "blackTargetDirectedPartition": metadata["blackTargetDirectedPartition"],
            "whiteTargetDirectedPartition": metadata["whiteTargetDirectedPartition"],
            "engineFile": relative,
            "engineFileSha256": sha256_file(path),
            "nodeCount": len(game.get("nodes") or []),
            "eventCount": len(game.get("events") or []),
        })
    write_csv(root / "engine_game_index.csv", index_rows)
    write_json(root / "engine_game_index.json", {
        "schema": "oq-reference-engine-game-index-v1", "games": index_rows,
    })

    index_by_id = {row["gameId"]: row for row in index_rows}
    partition_outputs = []
    for source_name, output_name, schema in (
        ("partitions_unordered.json", "partitions_unordered_engine_index.json", "oq-reference-unordered-engine-index-v1"),
        ("partitions_directed.json", "partitions_directed_engine_index.json", "oq-reference-directed-engine-index-v1"),
    ):
        payload = json.loads((root / source_name).read_text(encoding="utf-8"))
        partitions = payload.get("partitions") if isinstance(payload.get("partitions"), list) else []
        indexed = []
        for partition in partitions:
            row = dict(partition)
            game_ids = [str(game_id) for game_id in partition.get("mergedGameIds") or []]
            row["engineGames"] = [
                {
                    "gameId": game_id,
                    "sourceKind": index_by_id[game_id]["sourceKind"],
                    "engineFile": index_by_id[game_id]["engineFile"],
                }
                for game_id in game_ids
            ]
            indexed.append(row)
        output_path = root / output_name
        write_json(output_path, {"schema": schema, "partitions": indexed})
        partition_outputs.append(str(output_path))

    audit = {
        "schema": "oq-reference-partition-engine-index-audit-v1",
        "ok": True,
        "verifiedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gameCount": len(index_rows),
        "existingReferenceGameCount": sum(row["sourceKind"] == "existingReference" for row in index_rows),
        "cacheSelectedGameCount": sum(row["sourceKind"] == "validatedCacheExpansion" for row in index_rows),
        "snapshotSelectedGameCount": sum(row["sourceKind"] == "uniqueSnapshotExpansion" for row in index_rows),
        "addedGameCount": sum(row["sourceKind"] != "existingReference" for row in index_rows),
        "mainMatrixGameCount": sum(row["partitionScope"] == "main_bilateral" for row in index_rows),
        "lowEloExtensionGameCount": sum(
            row["partitionScope"] == "baseline_low_elo_extension" for row in index_rows
        ),
        "outsideUnpartitionedGameCount": sum(
            row["partitionScope"] == "outside_unpartitioned" for row in index_rows
        ),
        "uniqueEngineFileCount": len({row["engineFile"] for row in index_rows}),
        "engineDirectory": str(engine),
        "partitionEngineIndexes": partition_outputs,
    }
    write_json(root / "partition_engine_index_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
