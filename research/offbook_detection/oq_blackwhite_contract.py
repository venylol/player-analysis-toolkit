#!/usr/bin/env python3
"""Shared contracts for configuration-driven OQ black/white references.

The expansion and Level22 runners intentionally remain separate stages.  This
module only owns the immutable configuration, hashing, path, and partition
helpers that both stages must interpret identically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            result.append(value)
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{os.getpid()}.{target.name}.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _require_path(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"black-white expansion config requires {key}")
    return resolve_repo_path(value)


def validate_expansion_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the shared OQ expansion contract without touching the network."""

    result = dict(config)
    if result.get("schema") != "oq-reference-blackwhite-expansion-config-v2":
        raise ValueError("unexpected black-white expansion config schema")
    required_strings = (
        "batchId",
        "runDateAsiaShanghai",
        "baselineSourceReference",
        "baselineSentinelReference",
        "sourceOutputDirectory",
        "snapshotOutputDirectory",
        "sentinelOutputDirectory",
        "playerEloOutputDirectory",
        "anscombeOutputDirectory",
        "calibrationOutputDirectory",
        "enginePath",
    )
    for key in required_strings:
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"black-white expansion config requires non-empty {key}")
    integer_fields = (
        "baselineGameCount",
        "baselineMainMatrixGameCount",
        "baselineLowEloExtensionCount",
        "targetPerBlackWhiteCell",
        "minimumElo",
        "binWidth",
        "formalBinCount",
        "historicalFormalMaximum",
        "selectionSeed",
        "calibrationSplitSeed",
    )
    for key in integer_fields:
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"black-white expansion config requires integer {key}")
    if result["baselineGameCount"] < 0 or result["baselineMainMatrixGameCount"] < 0 or result["baselineLowEloExtensionCount"] < 0:
        raise ValueError("baseline counts must be non-negative")
    if result["baselineMainMatrixGameCount"] + result["baselineLowEloExtensionCount"] != result["baselineGameCount"]:
        raise ValueError("baseline matrix and low-Elo counts must add to baselineGameCount")
    if result["targetPerBlackWhiteCell"] <= 0:
        raise ValueError("targetPerBlackWhiteCell must be positive")
    if result["minimumElo"] < 0 or result["binWidth"] <= 0 or result["formalBinCount"] != 9:
        raise ValueError("formal black-white dimensions must be minimum>=0, positive width, and 9 bins")
    if result["historicalFormalMaximum"] < result["minimumElo"] + result["binWidth"] * (result["formalBinCount"] - 1):
        raise ValueError("historicalFormalMaximum would shrink the final formal bucket")
    if result["selectionSeed"] == result["calibrationSplitSeed"]:
        raise ValueError("selectionSeed and calibrationSplitSeed must be distinct")
    if result["selectionSeed"] == 20260821502 or result["calibrationSplitSeed"] == 20260821502:
        raise ValueError("new expansion contracts cannot reuse the old selection seed")
    for key in (
        "baselineSourceReference",
        "baselineSentinelReference",
        "sourceOutputDirectory",
        "snapshotOutputDirectory",
        "sentinelOutputDirectory",
        "playerEloOutputDirectory",
        "anscombeOutputDirectory",
        "calibrationOutputDirectory",
        "enginePath",
    ):
        _require_path(result, key)
    if "baselinePlayerEloReference" in result:
        _require_path(result, "baselinePlayerEloReference")
    if "baselineV4Reference" in result:
        _require_path(result, "baselineV4Reference")
    local_sources = result.get("localSources")
    if not isinstance(local_sources, dict):
        raise ValueError("black-white expansion config requires localSources")
    for key in ("rawCache", "materializedDataset", "baselineSnapshot"):
        if key not in local_sources:
            raise ValueError(f"localSources requires {key}")
        _require_path(local_sources, key)
    network = result.get("network")
    if not isinstance(network, dict):
        raise ValueError("black-white expansion config requires network contract")
    if network.get("direct") is not True or network.get("proxyDisabled") is not True:
        raise ValueError("network contract must require direct=true and proxyDisabled=true")
    for key in ("workers", "leaderboardMaximumAttempts", "playerListMaximumAttempts", "detailMaximumAttempts"):
        if isinstance(network.get(key), bool) or not isinstance(network.get(key), int) or int(network[key]) < 1:
            raise ValueError(f"network.{key} must be a positive integer")
    if float(network.get("requestTimeoutSeconds", 0)) <= 0:
        raise ValueError("network.requestTimeoutSeconds must be positive")
    level22 = result.get("level22")
    if not isinstance(level22, dict):
        raise ValueError("black-white expansion config requires level22 contract")
    expected_level22 = {
        "workers": 12,
        "threadsPerWorker": 16,
        "level": 22,
        "hash": 25,
        "wldFromPlyInclusive": 39,
        "book": "enabled-default",
    }
    for key, expected in expected_level22.items():
        if level22.get(key) != expected:
            raise ValueError(f"level22.{key} must be {expected!r}")
    paths = {key: _require_path(result, key) for key in (
        "baselineSourceReference",
        "baselineSentinelReference",
        "sourceOutputDirectory",
        "snapshotOutputDirectory",
        "sentinelOutputDirectory",
        "playerEloOutputDirectory",
        "anscombeOutputDirectory",
        "calibrationOutputDirectory",
        "enginePath",
    )}
    paths["baselineSnapshot"] = _require_path(local_sources, "baselineSnapshot")
    if len({paths[key] for key in (
        "sourceOutputDirectory",
        "sentinelOutputDirectory",
        "playerEloOutputDirectory",
        "anscombeOutputDirectory",
        "calibrationOutputDirectory",
    )}) != 5:
        raise ValueError("new output directories must be distinct")
    result["paths"] = {key: str(value) for key, value in paths.items()}
    result["topBinLower"] = result["minimumElo"] + result["binWidth"] * (result["formalBinCount"] - 1)
    result["formalCells"] = result["formalBinCount"] ** 2
    result["legacyUnorderedCells"] = result["formalBinCount"] * (result["formalBinCount"] + 1) // 2
    result["selectionSorting"] = [
        "canonicalBlackWhiteCellKey",
        "gameId",
        "sourcePriority",
        "stableSha256",
    ]
    result["canonicalization"] = "UTF-8 JSON with sorted keys and compact separators for summary hash"
    return result


def load_expansion_config(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    config_path = resolve_repo_path(path)
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"expansion config is not a JSON object: {config_path}")
    validated = validate_expansion_config(config)
    return validated, config_path, canonical_sha256(config)


def formal_cells(config: dict[str, Any]) -> list[tuple[int, int]]:
    minimum = int(config["minimumElo"])
    width = int(config["binWidth"])
    count = int(config["formalBinCount"])
    lowers = [minimum + index * width for index in range(count)]
    return [(black, white) for black in lowers for white in lowers]


def unordered_cells(config: dict[str, Any]) -> list[tuple[int, int]]:
    minimum = int(config["minimumElo"])
    width = int(config["binWidth"])
    count = int(config["formalBinCount"])
    lowers = [minimum + index * width for index in range(count)]
    return [(a, b) for index, a in enumerate(lowers) for b in lowers[index:]]


def dynamic_formal_maximum(config: dict[str, Any], baseline_maximum: int, leaderboard_maximum: int) -> int:
    minimum = int(config["minimumElo"])
    top_lower = int(config["topBinLower"])
    maximum = max(
        int(config["historicalFormalMaximum"]),
        int(baseline_maximum),
        int(leaderboard_maximum),
    )
    if maximum < top_lower:
        raise ValueError("dynamic formal maximum is below the final formal bucket lower bound")
    if minimum + int(config["binWidth"]) * int(config["formalBinCount"]) <= maximum:
        # The 9-bin contract deliberately absorbs every value at or above the
        # last lower bound; this check documents that no tenth bucket is made.
        return maximum
    return maximum


def formal_bin(value: Any, maximum: int, config: dict[str, Any]) -> int | None:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rating):
        return None
    minimum = int(config["minimumElo"])
    width = int(config["binWidth"])
    top_lower = int(config["topBinLower"])
    if rating < minimum or rating > int(maximum):
        return None
    return top_lower if rating >= top_lower else minimum + int((rating - minimum) // width) * width


def bin_label(lower: int, maximum: int, config: dict[str, Any]) -> str:
    top_lower = int(config["topBinLower"])
    width = int(config["binWidth"])
    return f"[{lower},{maximum}]" if int(lower) == top_lower else f"[{lower},{int(lower) + width})"


def source_priority(source_kind: str) -> int:
    return {
        "existingReference": 0,
        "validatedCacheExpansion": 1,
        "priorFrozenSnapshotExpansion": 1,
        "uniqueSnapshotExpansion": 2,
    }.get(str(source_kind), 99)


def stable_summary_sha(
    cell: tuple[int, int],
    game_id: str,
    source_kind: str,
    summary: dict[str, Any],
    selection_seed: int,
) -> str:
    value = (
        f"{int(selection_seed)}|{int(cell[0])}__{int(cell[1])}|{game_id}|"
        f"{source_kind}|{canonical_json(summary)}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def capacity_target(existing_count: int, frozen_valid_capacity: int, target: int) -> int:
    return max(int(existing_count), min(int(target), int(frozen_valid_capacity)))


def matrix_counts(rows: Iterable[dict[str, Any]], config: dict[str, Any], maximum: int) -> dict[tuple[int, int], int]:
    counts = {cell: 0 for cell in formal_cells(config)}
    for row in rows:
        if row.get("inMainMatrix") is not True:
            continue
        cell = (int(row.get("blackBinLower")), int(row.get("whiteBinLower")))
        if cell not in counts:
            raise ValueError(f"row maps outside the configured directed matrix: {cell}")
        counts[cell] += 1
    return counts
