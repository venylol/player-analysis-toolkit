#!/usr/bin/env python3
"""Remap complete Level22 game caches to a combined bundle's ordinal filenames."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle_path = args.bundle.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cache directory: {output}")
    output.mkdir(parents=True)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    details = bundle.get("details") if isinstance(bundle.get("details"), list) else []
    details = sorted(details, key=lambda item: str(item.get("created") or ""))
    detail_by_id = {str(item.get("id") or ""): item for item in details}
    if not detail_by_id or len(detail_by_id) != len(details):
        raise ValueError("bundle IDs are empty or duplicated")

    cached: dict[str, tuple[Path, dict[str, Any]]] = {}
    for directory in [path.resolve() for path in args.source_dir]:
        for path in sorted(directory.glob("game_*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            game_id = str(value.get("gameId") or "")
            if game_id in detail_by_id and game_id not in cached:
                cached[game_id] = (path, value)

    files = []
    for ordinal, detail in enumerate(details, start=1):
        game_id = str(detail.get("id") or "")
        if game_id not in cached:
            continue
        source_path, game = cached[game_id]
        source_moves = [
            str(event.get("m") or "").casefold()
            for event in (detail.get("position") or {}).get("moves") or []
            if isinstance(event, dict) and MOVE_RE.fullmatch(str(event.get("m") or ""))
        ]
        result_moves = [str(node.get("move") or "").casefold() for node in game.get("nodes") or []]
        if source_moves != result_moves:
            raise ValueError(f"cached moves differ from combined source for {game_id}")
        engine = game.get("engine") if isinstance(game.get("engine"), dict) else {}
        if engine.get("level") != 22 or engine.get("threads") != 16 or engine.get("hash") != 25:
            raise ValueError(f"cached engine contract mismatch for {game_id}")
        remapped = dict(game)
        remapped["round"] = 0
        remapped["table"] = ordinal
        target_path = output / f"game_0_{ordinal}_{game_id}.json"
        write_json(target_path, remapped)
        files.append({
            "gameId": game_id,
            "source": str(source_path),
            "sourceSha256": sha256_file(source_path),
            "target": str(target_path),
            "targetSha256": sha256_file(target_path),
            "newTable": ordinal,
        })
    manifest = {
        "schema": "ega-combined-cache-remap-v1",
        "createdAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bundle": str(bundle_path),
        "bundleSha256": sha256_file(bundle_path),
        "sourceDirectories": [str(path.resolve()) for path in args.source_dir],
        "remappedGameCount": len(files),
        "files": files,
    }
    write_json(output / "cache_remap_manifest.json", manifest)
    print(json.dumps({"ok": True, "output": str(output), "remappedGameCount": len(files)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
