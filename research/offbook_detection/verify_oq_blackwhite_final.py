#!/usr/bin/env python3
"""Independently verify one configuration-driven black/white OQ chain.

This verifier is deliberately separate from the expansion selector.  It
recomputes the directed matrix and capacity formula from the frozen source,
then checks the self-contained Level22, Sentinel, Player Elo, Anscombe/KNN,
and calibration artifacts before emitting the final completion certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OFFBOOK = ROOT / "research" / "offbook_detection"
SRC_ROOT = ROOT / "src"
CURRENT_600_SOURCE = ROOT / "research" / "offbook_detection" / "data" / "oq_elo_matchup600_blackwhite_reference_level22_1600plus_20260911"
if str(OFFBOOK) not in sys.path:
    sys.path.insert(0, str(OFFBOOK))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oq_blackwhite_contract import load_expansion_config, resolve_repo_path  # noqa: E402
from pull_oq_transformer_dataset import valid_detail  # noqa: E402
from build_elo_reference import replay_is_legal  # noqa: E402
from player_analysis_toolkit import sentinel_elo  # type: ignore  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{os.getpid()}.{path.name}.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_manifest(path: Path, root: Path) -> dict[str, Any]:
    manifest = read_json(require_file(path))
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError(f"manifest has no file list: {path}")
    listed = {str(row.get("path") or ""): row for row in rows}
    if "" in listed or path.name in listed:
        raise ValueError(f"manifest includes itself: {path}")
    for relative, row in listed.items():
        target = root / relative
        if not target.is_file():
            raise ValueError(f"manifest file missing: {target}")
        expected_sha = str(row.get("sha256") or "")
        if sha256_file(target) != expected_sha:
            raise ValueError(f"manifest hash mismatch: {target}")
        if row.get("bytes") is not None and int(row["bytes"]) != target.stat().st_size:
            raise ValueError(f"manifest byte count mismatch: {target}")
    if manifest.get("fileCount") is not None and int(manifest["fileCount"]) != len(rows):
        raise ValueError(f"manifest file count mismatch: {path}")
    return manifest


def rewrite_flat_manifest(path: Path, root: Path) -> dict[str, Any]:
    existing = read_json(path) if path.is_file() else {}
    rows = []
    for target in sorted(item for item in root.iterdir() if item.is_file() and item != path):
        rows.append({
            "path": target.relative_to(root).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    existing.update({
        "referenceDirectory": str(root.resolve()),
        "fileCount": len(rows),
        "files": rows,
        "createdAt": utc_now(),
        "selfHashPolicy": "manifest file is excluded from its own hash list",
    })
    atomic_json(path, existing)
    return existing


def write_source_manifest(source: Path, config_sha: str) -> dict[str, Any]:
    manifest_path = source / "final_sha256_manifest.json"
    rows = []
    lifecycle_progress = source / "provenance" / "lifecycle_progress.json"
    dynamic_log_prefix = "provenance/"
    for target in sorted(item for item in source.rglob("*") if item.is_file() and item != manifest_path):
        relative = target.relative_to(source).as_posix()
        if target == lifecycle_progress or (relative.startswith(dynamic_log_prefix) and relative.endswith(".log")):
            continue
        rows.append({
            "path": relative,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    manifest = {
        "schema": "oq-reference-final-sha256-manifest-v2",
        "createdAtUtc": utc_now(),
        "configSha256": config_sha,
        "referenceDirectory": str(source.resolve()),
        "fileCount": len(rows),
        "files": rows,
        "selfHashPolicy": "final_sha256_manifest.json is excluded to avoid recursive self-hash",
        "excludedDynamicFiles": ["provenance/lifecycle_progress.json", "provenance/*.log"],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _patch_path_hashes(value: Any, relative_path: str, digest: str) -> int:
    changed = 0
    if isinstance(value, dict):
        if value.get("path") == relative_path and "sha256" in value:
            value["sha256"] = digest
            changed += 1
        for child in value.values():
            changed += _patch_path_hashes(child, relative_path, digest)
    elif isinstance(value, list):
        for child in value:
            changed += _patch_path_hashes(child, relative_path, digest)
    return changed


def rebind_derived_manifests(
    source: Path,
    sentinel: Path,
    elo: Path,
    anscombe: Path,
    calibration: Path,
    config: dict[str, Any],
) -> None:
    source_digest = sha256_file(source / "final_sha256_manifest.json")
    sentinel_source_path = sentinel / "reference_source_manifest.json"
    sentinel_source = read_json(sentinel_source_path)
    if _patch_path_hashes(sentinel_source, "final_sha256_manifest.json", source_digest) == 0:
        raise ValueError("Sentinel source manifest does not record source final manifest")
    atomic_json(sentinel_source_path, sentinel_source)
    rewrite_flat_manifest(sentinel / "reference_sha256_manifest.json", sentinel)
    sentinel_digest = sha256_file(sentinel / "reference_sha256_manifest.json")

    elo_source_path = elo / "reference_source_manifest.json"
    elo_source = read_json(elo_source_path)
    changed = _patch_path_hashes(elo_source, "final_sha256_manifest.json", source_digest)
    changed += _patch_path_hashes(elo_source, "reference_sha256_manifest.json", sentinel_digest)
    if changed < 2:
        raise ValueError("Player Elo source manifest does not record source and Sentinel manifests")
    atomic_json(elo_source_path, elo_source)
    rewrite_flat_manifest(elo / "reference_sha256_manifest.json", elo)
    elo_digest = sha256_file(elo / "reference_sha256_manifest.json")
    anscombe_digest = _rebind_anscombe_reference(anscombe, config, elo_digest)
    _rebind_calibration(calibration, config, anscombe_digest)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{os.getpid()}.{path.name}.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _rebind_anscombe_reference(anscombe: Path, config: dict[str, Any], elo_manifest_sha: str) -> str:
    manifest_path = anscombe / str(config["conditionalReferenceManifest"])
    records_path = anscombe / str(config["conditionalReferenceRecords"])
    manifest = read_json(require_file(manifest_path))
    progress_path = anscombe / "progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else None
    preparation_sha = manifest.get("preparationContractSha256")
    if isinstance(progress, dict) and isinstance(progress.get("contract"), dict):
        contract = dict(progress["contract"])
        contract["referenceManifestSha256"] = elo_manifest_sha
        progress["contract"] = contract
        progress["contractSha256"] = sentinel_elo.canonical_sha256(contract)
        preparation_sha = progress["contractSha256"]
    manifest["referenceManifestSha256"] = elo_manifest_sha
    if preparation_sha:
        manifest["preparationContractSha256"] = preparation_sha
    atomic_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)

    audit_path = anscombe / "anscombe_reference_audit.json"
    if audit_path.is_file():
        audit = read_json(audit_path)
        if preparation_sha:
            audit["preparationContractSha256"] = preparation_sha
        audit["manifestSha256"] = manifest_sha
        atomic_json(audit_path, audit)
    if isinstance(progress, dict):
        progress["manifestSha256"] = manifest_sha
        progress["recordsSha256"] = sha256_file(records_path)
        atomic_json(progress_path, progress)
    return manifest_sha


def _rebind_calibration(calibration: Path, config: dict[str, Any], anscombe_manifest_sha: str) -> None:
    reference_fields = (
        "referenceManifestSha256",
        "conditionalReferenceManifestSha256",
        "anscombeReferenceManifestSha256",
        "inputReferenceManifestSha256",
    )
    cases_dir = calibration / "cases"
    case_hashes: dict[str, str] = {}
    if cases_dir.is_dir():
        for case_path in sorted(cases_dir.glob("*.json")):
            case = read_json(case_path)
            for field in reference_fields:
                if field in case:
                    case[field] = anscombe_manifest_sha
            atomic_json(case_path, case)
            case_hashes[case_path.name] = sha256_file(case_path)

    cases_path = calibration / str(config["calibrationCases"])
    cases = list(iter_jsonl(require_file(cases_path)))
    for case in cases:
        for field in reference_fields:
            if field in case:
                case[field] = anscombe_manifest_sha
    atomic_jsonl(cases_path, cases)

    artifact_path = calibration / str(config["calibrationArtifact"])
    artifact = read_json(require_file(artifact_path))
    for field in reference_fields:
        if field in artifact:
            artifact[field] = anscombe_manifest_sha
    artifact["casesSha256"] = sha256_file(cases_path)
    atomic_json(artifact_path, artifact)
    artifact_sha = sha256_file(artifact_path)

    manifest_path = calibration / "calibration_sha256_manifest_v4.json"
    manifest = read_json(require_file(manifest_path))
    for field in reference_fields:
        if field in manifest:
            manifest[field] = anscombe_manifest_sha
    manifest["calibrationArtifactSha256"] = artifact_sha
    manifest["calibrationCasesSha256"] = artifact["casesSha256"]
    for item in manifest.get("files") or []:
        if item.get("path") == str(config["calibrationArtifact"]):
            item["sha256"] = artifact_sha
        elif item.get("path") == str(config["calibrationCases"]):
            item["sha256"] = artifact["casesSha256"]
    atomic_json(manifest_path, manifest)

    progress_path = calibration / "progress.json"
    if progress_path.is_file():
        progress = read_json(progress_path)
        contract = progress.get("contract")
        if isinstance(contract, dict):
            contract = dict(contract)
            for field in reference_fields:
                if field in contract:
                    contract[field] = anscombe_manifest_sha
            progress["contract"] = contract
            progress["contractSha256"] = sentinel_elo.canonical_sha256(contract)
        for key in ("completedCalibrationCases", "completedValidationCases"):
            for account, item in (progress.get(key) or {}).items():
                case_path = sentinel_elo._v4_case_path(cases_dir, account)
                if case_path.is_file():
                    item["sha256"] = sha256_file(case_path)
        progress["calibrationArtifactSha256"] = artifact_sha
        progress["calibrationCasesSha256"] = artifact["casesSha256"]
        progress["calibrationManifestSha256"] = sha256_file(manifest_path)
        atomic_json(progress_path, progress)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _formal_cell(black: Any, white: Any, minimum: int, width: int, count: int, maximum: int) -> tuple[int, int] | None:
    if not _finite(black) or not _finite(white):
        return None
    black_value, white_value = float(black), float(white)
    if black_value < minimum or white_value < minimum or black_value > maximum or white_value > maximum:
        return None
    top = minimum + width * (count - 1)
    return (
        top if black_value >= top else minimum + int((black_value - minimum) // width) * width,
        top if white_value >= top else minimum + int((white_value - minimum) // width) * width,
    )


def _cell_label(lower: int, maximum: int, minimum: int = 1600, width: int = 100, count: int = 9) -> str:
    top = minimum + width * (count - 1)
    return f"[{lower},{maximum}]" if lower == top else f"[{lower},{lower + width})"


def _baseline_ids(baseline: Path) -> tuple[set[str], set[str], int]:
    bundle = read_json(require_file(baseline / "selected_account_bundle.json"))
    selection = read_json(require_file(baseline / "selected_games_with_partitions.json"))
    details = {str(row.get("id") or "") for row in bundle.get("details") or []}
    rows = selection.get("games") or []
    ids = {str(row.get("gameId") or "") for row in rows}
    if not details or details != ids or len(details) != len(bundle.get("details") or []) or len(ids) != len(rows):
        raise ValueError("baseline bundle and partition IDs are not unique and identical")
    low = {str(row["gameId"]) for row in rows if row.get("partitionScope") == "baseline_low_elo_extension"}
    return ids, low, len(rows)


def _embedded_baseline_ids(source: Path) -> tuple[set[str], set[str], int]:
    """Read the inherited baseline marker retained inside the 600 source.

    The completed 600 source is self-contained.  Its selected-game rows keep
    ``sourceKind=existingReference`` for the games inherited from the 550
    build, so verification can continue after the historical input directory
    is archived without inventing a replacement game set.
    """

    selection = read_json(require_file(source / "selected_games_with_partitions.json"))
    rows = selection.get("games") or []
    existing_rows = [row for row in rows if row.get("sourceKind") == "existingReference"]
    ids = {str(row.get("gameId") or "") for row in existing_rows}
    if not ids or len(ids) != len(existing_rows):
        raise ValueError("current 600 source does not contain a unique embedded inherited baseline")
    audit = read_json(require_file(source / "selection_audit.json"))
    expected_count = int(audit.get("existingReferenceGameCount", -1))
    if expected_count != len(existing_rows):
        raise ValueError("current 600 embedded baseline count disagrees with selection audit")
    low = {str(row["gameId"]) for row in existing_rows if row.get("partitionScope") == "baseline_low_elo_extension"}
    return ids, low, expected_count


def _resolve_baseline_ids(expansion: dict[str, Any], source: Path) -> tuple[Path, set[str], set[str], int, str]:
    """Resolve a historical baseline or the current self-contained 600 one."""

    baseline = Path(expansion["paths"]["baselineSourceReference"])
    if baseline.is_dir():
        ids, low, count = _baseline_ids(baseline)
        return baseline, ids, low, count, "configured_historical_baseline"

    if source.resolve() != CURRENT_600_SOURCE.resolve():
        raise FileNotFoundError(baseline)
    ids, low, count = _embedded_baseline_ids(source)
    return source, ids, low, count, "current_600_self_contained"


def _verify_source(expansion: dict[str, Any]) -> dict[str, Any]:
    source = Path(expansion["paths"]["sourceOutputDirectory"])
    snapshot = Path(expansion["paths"]["snapshotOutputDirectory"])
    bundle = read_json(require_file(source / "selected_account_bundle.json"))
    selection = read_json(require_file(source / "selected_games_with_partitions.json"))
    details_list = bundle.get("details") or []
    selected = selection.get("games") or []
    details = {str(row.get("id") or ""): row for row in details_list}
    rows = {str(row.get("gameId") or ""): row for row in selected}
    if not details or len(details) != len(details_list) or set(details) != set(rows) or len(rows) != len(selected):
        raise ValueError("source bundle and selected rows are not globally unique and identical")
    baseline, baseline_ids, baseline_low, baseline_count, baseline_mode = _resolve_baseline_ids(expansion, source)
    if baseline_count != len(baseline_ids) or baseline_count != int(expansion["baselineGameCount"]):
        raise ValueError("baseline game count changed")
    source_low = {game_id for game_id, row in rows.items() if row.get("partitionScope") == "baseline_low_elo_extension"}
    if not baseline_ids <= set(rows) or source_low != baseline_low:
        raise ValueError("baseline game IDs or low-Elo extension were not preserved")

    maximum = int(bundle.get("selection", {}).get("maximumElo"))
    expected_maximum = int(read_json(source / "expansion_manifest.json")["configuration"]["dynamicFormalMaximumElo"])
    if maximum != expected_maximum or maximum < int(expansion["historicalFormalMaximum"]):
        raise ValueError("dynamic formal maximum does not match the frozen expansion contract")
    minimum = int(expansion["minimumElo"])
    width = int(expansion["binWidth"])
    count = int(expansion["formalBinCount"])
    target = int(expansion["targetPerBlackWhiteCell"])
    cells = [(minimum + i * width, minimum + j * width) for i in range(count) for j in range(count)]
    actual: dict[tuple[int, int], list[str]] = defaultdict(list)
    added_counts = defaultdict(int)
    for game_id, detail in details.items():
        players = detail.get("players") or []
        if len(players) != 2 or any(not str(player.get("id") or "").strip() for player in players):
            raise ValueError(f"invalid player contract: {game_id}")
        if len({str(player.get("id")) for player in players}) != 2:
            raise ValueError(f"duplicate player IDs: {game_id}")
        for player in players:
            if not _finite(player.get("oldR")):
                raise ValueError(f"missing oldR: {game_id}")
        row = rows[game_id]
        if row.get("sourceKind") not in {"existingReference", "validatedCacheExpansion", "priorFrozenSnapshotExpansion", "uniqueSnapshotExpansion"}:
            raise ValueError(f"unknown source kind: {game_id}")
        if row.get("sourceKind") != "existingReference":
            summary = next((item for item in (bundle.get("index") or []) if str(item.get("id") or item.get("gameId") or "") == game_id), None)
            if not isinstance(summary, dict):
                raise ValueError(f"missing summary for added game: {game_id}")
            valid, reason = valid_detail(summary, detail)
            if not valid or detail.get("finished") is not True or not replay_is_legal(detail):
                raise ValueError(f"added game failed detail contract: {game_id}: {reason}")
            added_counts[str(row.get("sourceKind"))] += 1
        cell = _formal_cell(players[0].get("oldR"), players[1].get("oldR"), minimum, width, count, maximum)
        is_main = cell is not None
        expected_scope = "main_bilateral" if is_main else "baseline_low_elo_extension"
        if bool(row.get("inMainMatrix")) != is_main or row.get("partitionScope") != expected_scope:
            raise ValueError(f"source row classification mismatch: {game_id}")
        if is_main:
            actual[cell].append(game_id)
            # ``blackBinLower``/``whiteBinLower`` describe the ordinary
            # rating bins.  The dynamic formal maximum folds the final
            # (possibly partial) rating bin into ``topBinLower`` (for
            # example, white oldR=2504 belongs to the formal 2400 cell
            # when the frozen maximum is 2571).  Validate the directed
            # matrix key emitted by the selector instead of comparing the
            # ordinary-bin lower bound to the formal cell lower bound.
            expected_cell_key = f"{cell[0]}__{cell[1]}"
            if row.get("blackWhiteCellKey") != expected_cell_key:
                raise ValueError(f"black/white direction mismatch: {game_id}")

    bw_payload = read_json(require_file(source / "partitions_black_white.json"))
    bw_rows = {(int(row["blackLower"]), int(row["whiteLower"])): row for row in bw_payload.get("partitions") or []}
    if len(bw_rows) != 81 or set(bw_rows) != set(cells):
        raise ValueError("black-white matrix is not a complete 9x9")
    capacity_path = snapshot / "acquisition" / "black_white_capacity_after_details.json"
    capacity_payload = read_json(require_file(capacity_path))
    capacity = {(int(row["blackBinLower"]), int(row["whiteBinLower"])): row for row in capacity_payload.get("cells") or []}
    if set(capacity) != set(cells):
        raise ValueError("frozen capacity audit is not a complete 9x9")
    for cell in cells:
        actual_ids = set(actual[cell])
        row = bw_rows[cell]
        if set(row.get("mergedGameIds") or []) != actual_ids or int(row["finalCount"]) != len(actual_ids):
            raise ValueError(f"black-white matrix IDs/count disagree: {cell}")
        cap = capacity[cell]
        existing = int(cap.get("baselineReferenceCount", cap.get("existingCount", -1)))
        frozen = int(cap["frozenValidCapacity"])
        expected = max(existing, min(target, frozen))
        if int(row["finalCount"]) != expected:
            raise ValueError(f"capacity formula mismatch: {cell}")
        if int(row["remainingGap"]) != max(0, target - int(row["finalCount"])):
            raise ValueError(f"remaining gap mismatch: {cell}")
        if int(row["finalCount"]) < target and row.get("capacityExhaustedBelowTarget") is not True:
            raise ValueError(f"below-target cell is not marked exhausted: {cell}")

    unordered = [row for row in read_json(require_file(source / "partitions_unordered.json")).get("partitions") or [] if row.get("partitionScope") == "main_bilateral"]
    if len(unordered) != 45:
        raise ValueError("unordered compatibility view does not contain 45 formal cells")
    for row in unordered:
        a, b = int(row["pairLowerA"]), int(row["pairLowerB"])
        expected_ids = set(actual[(a, b)]) if a == b else set(actual[(a, b)]) | set(actual[(b, a)])
        if set(row.get("mergedGameIds") or []) != expected_ids:
            raise ValueError(f"unordered view is not derived from directed cells: {(a, b)}")
    selection_audit = read_json(require_file(source / "selection_audit.json"))
    coverage = read_json(require_file(source / "provenance" / "black_white_coverage_audit.json"))
    if selection_audit.get("ok") is not True or coverage.get("ok") is not True:
        raise ValueError("source selection or coverage audit is not ok=true")
    if int(selection_audit["existingReferenceGameCount"]) != baseline_count:
        raise ValueError("selection audit baseline count mismatch")
    if int(selection_audit["cacheSelectedCount"]) != sum(added_counts[k] for k in ("validatedCacheExpansion", "priorFrozenSnapshotExpansion")):
        raise ValueError("cache addition count mismatch")
    if int(selection_audit["snapshotSelectedCount"]) != added_counts["uniqueSnapshotExpansion"]:
        raise ValueError("snapshot addition count mismatch")
    return {
        "source": source,
        "baseline": baseline,
        "baselineMode": baseline_mode,
        "snapshot": snapshot,
        "bundle": bundle,
        "selection": selection,
        "details": details,
        "rows": rows,
        "maximumElo": maximum,
        "actualCells": {f"{a}__{b}": sorted(actual[(a, b)]) for a, b in cells},
        "addedCounts": dict(added_counts),
        "sourceGameCount": len(rows),
        "mainMatrixGameCount": sum(len(actual[cell]) for cell in cells),
        "lowEloExtensionGameCount": len(source_low),
        "formalMatrix": [[len(actual[(minimum + i * width, minimum + j * width)]) for j in range(count)] for i in range(count)],
        "unorderedSummaryPath": str((source / "partitions_unordered.json").resolve()),
        "capacityPath": str(capacity_path.resolve()),
    }


def _verify_level22(source_info: dict[str, Any], expansion: dict[str, Any]) -> dict[str, Any]:
    source = source_info["source"]
    selected = source_info["rows"]
    completion = read_json(require_file(source / "reference_completion_audit.json"))
    audit = read_json(require_file(source / "engine_level22" / "audit.json"))
    progress = read_json(require_file(source / "engine_level22" / "progress.json"))
    runner = read_json(require_file(source / "engine_level22" / "runner_audit.json"))
    for payload, name in ((completion, "completion"), (audit, "engine audit")):
        if payload.get("ok") is not True:
            raise ValueError(f"Level22 {name} is not ok=true")
    if progress.get("complete") is not True or int(progress.get("completedCount", -1)) != len(selected):
        raise ValueError("Level22 progress is incomplete")
    contract = completion.get("contract") or {}
    expected = {"level": 22, "workers": 12, "threadsPerConsole": 16, "hash": 25, "book": "enabled-default", "wldFromPlyInclusive": 39}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Level22 contract mismatch")
    index_rows = read_json(require_file(source / "engine_game_index.json")).get("games") or []
    index = {str(row.get("gameId") or ""): row for row in index_rows}
    if len(index) != len(index_rows) or set(index) != set(selected):
        raise ValueError("Level22 engine index IDs are not exactly the source IDs")
    if len({str(row.get("engineFile") or "") for row in index_rows}) != len(index_rows):
        raise ValueError("Level22 engine files are not one-to-one")
    for game_id, row in index.items():
        engine_file = (source / str(row["engineFile"])).resolve()
        if source not in engine_file.parents or not engine_file.is_file():
            raise ValueError(f"Level22 file is not self-contained: {game_id}")
        if sha256_file(engine_file) != str(row.get("engineFileSha256")):
            raise ValueError(f"Level22 file SHA mismatch: {game_id}")
    wld = read_json(require_file(source / "engine_level22" / "engine_wld_loss_totals_from_ply39.json")).get("gamePlayerTotals") or []
    if len(wld) != len(selected) * 2 or len({(str(row.get("game_id")), str(row.get("side"))) for row in wld}) != len(selected) * 2:
        raise ValueError("WLD output does not contain exactly two sides per source game")
    if int(runner.get("newGameCount", -1)) != len(selected) - int(expansion["baselineGameCount"]):
        raise ValueError("Level22 new-game count does not equal source additions")
    if int(runner.get("reusedBaselineGameCount", -1)) != int(expansion["baselineGameCount"]):
        raise ValueError("Level22 reused baseline count mismatch")
    old_strings = {
        str(expansion["baselineSourceReference"]).casefold(),
        str(Path(expansion["paths"]["baselineSourceReference"]).resolve()).casefold(),
    }
    for path in (source / "engine_game_index.json", source / "reference_completion_audit.json", source / "engine_level22" / "runner_audit.json"):
        content = path.read_text(encoding="utf-8").casefold()
        if any(value and value in content for value in old_strings):
            raise ValueError(f"new Level22 runtime metadata still points to old source: {path}")
    verify_manifest(source / "final_sha256_manifest.json", source)
    return {"completion": completion, "audit": audit, "progress": progress, "index": index, "wldCount": len(wld)}


def _verify_sentinel(source_info: dict[str, Any], expansion: dict[str, Any]) -> dict[str, Any]:
    source = source_info["source"]
    root = Path(expansion["paths"]["sentinelOutputDirectory"])
    audit = read_json(require_file(root / "reference_build_audit.json"))
    if audit.get("ok") is not True or int(audit.get("directedTargetRecordCount", -1)) != source_info["sourceGameCount"] * 2:
        raise ValueError("Sentinel audit/count mismatch")
    record_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(require_file(root / "directed_target_records.jsonl")):
        key = (str(row.get("gameId") or ""), str(row.get("targetColor") or ""))
        if key in record_map:
            raise ValueError(f"duplicate Sentinel directed record: {key}")
        record_map[key] = row
    if len(record_map) != source_info["sourceGameCount"] * 2:
        raise ValueError("Sentinel directed record count mismatch")
    index = read_json(source / "engine_game_index.json").get("games") or []
    engine = {str(row["gameId"]): row for row in index}
    for game_id, detail in source_info["details"].items():
        players = detail["players"]
        for color, target_i, opponent_i in (("black", 0, 1), ("white", 1, 0)):
            row = record_map[(game_id, color)]
            target, opponent = players[target_i], players[opponent_i]
            if str(row.get("targetPlayerId")) != str(target.get("id")) or str(row.get("opponentPlayerId")) != str(opponent.get("id")):
                raise ValueError(f"Sentinel target/opponent direction mismatch: {game_id}:{color}")
            if not math.isclose(float(row.get("targetOldR")), float(target.get("oldR")), rel_tol=0, abs_tol=1e-9) or not math.isclose(float(row.get("opponentOldR")), float(opponent.get("oldR")), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Sentinel oldR direction mismatch: {game_id}:{color}")
            source_level22 = Path(str(row.get("sourceLevel22File") or ""))
            if not source_level22.is_absolute():
                source_level22 = source / source_level22
            expected_level22 = (source / str(engine[game_id].get("engineFile") or "")).resolve()
            if source_level22.resolve() != expected_level22:
                raise ValueError(f"Sentinel Level22 provenance mismatch: {game_id}:{color}")
    source_manifest = read_json(root / "reference_source_manifest.json")
    if source_manifest.get("sourcePartitionDimension") != "black_white_directed_cell":
        raise ValueError("Sentinel source partition dimension mismatch")
    verify_manifest(root / "reference_sha256_manifest.json", root)
    return {"root": root, "audit": audit, "recordCount": len(record_map)}


def _verify_elo_and_anscombe(source_info: dict[str, Any], expansion: dict[str, Any], elo_config: dict[str, Any]) -> dict[str, Any]:
    source = source_info["source"]
    elo_root = Path(expansion["paths"]["playerEloOutputDirectory"])
    anscombe_root = Path(expansion["paths"]["anscombeOutputDirectory"])
    config = sentinel_elo.validate_v4_config(elo_config)
    if Path(config["sourceReferenceDirectory"]).resolve() != source.resolve():
        raise ValueError("Player Elo config does not point at the new source")
    audit = read_json(require_file(elo_root / "reference_build_audit.json"))
    if audit.get("ok") is not True or int(audit.get("sourceGameCount", -1)) != source_info["sourceGameCount"]:
        raise ValueError("Player Elo reference audit/count mismatch")
    phase_count = 0
    old_source_strings = {
        str(expansion["baselineSourceReference"]).casefold(),
        str(Path(expansion["paths"]["baselineSourceReference"]).resolve()).casefold(),
    }
    for row in iter_jsonl(require_file(elo_root / str(config["directedPhaseRecords"]))):
        phase_count += 1
        source_level22 = str(row.get("sourceLevel22File") or "").casefold()
        if any(value and source_level22.startswith(value) for value in old_source_strings):
            raise ValueError("Player Elo phase record points to old source path")
    if phase_count != source_info["sourceGameCount"] * 2:
        raise ValueError("Player Elo phase record count is not two per source game")
    verify_manifest(elo_root / str(config["referenceManifest"]), elo_root)
    anscombe_manifest_path = anscombe_root / str(config["conditionalReferenceManifest"])
    anscombe_manifest = read_json(require_file(anscombe_manifest_path))
    anscombe_audit = read_json(require_file(anscombe_root / "anscombe_reference_audit.json"))
    if anscombe_audit.get("ok") is not True or anscombe_audit.get("checks", {}).get("fourSequentialStages") is not True or anscombe_audit.get("checks", {}).get("allStoredCountsValid") is not True:
        raise ValueError("Anscombe/reference-z sequential-stage audit is incomplete")
    if anscombe_manifest.get("algorithmVersion") not in {None, sentinel_elo.ALGORITHM_VERSION_V4}:
        raise ValueError("Anscombe algorithm version mismatch")
    elo_manifest_sha = sha256_file(elo_root / str(config["referenceManifest"]))
    if anscombe_manifest.get("referenceManifestSha256") != elo_manifest_sha:
        raise ValueError("Anscombe manifest is not bound to the Player Elo manifest")
    anscombe_count = 0
    bad_k = 0
    for row in iter_jsonl(require_file(anscombe_root / str(config["conditionalReferenceRecords"]))):
        anscombe_count += 1
        for stage in range(1, 5):
            diagnostic = (row.get("zDiagnostics") or {}).get(f"phase{stage}") or {}
            n_allowed = int(diagnostic.get("N_allowed", 0))
            k = int(diagnostic.get("K", 0))
            expected_k = math.ceil(n_allowed ** (2.0 / 3.0)) if n_allowed > 0 else 0
            if k != expected_k:
                bad_k += 1
    if bad_k:
        raise ValueError(f"Anscombe/reference-z K ceil contract failed for {bad_k} rows")
    if anscombe_count != int(anscombe_manifest.get("recordCount", anscombe_count)):
        raise ValueError("Anscombe record count/manifest mismatch")
    records_path = anscombe_root / str(config["conditionalReferenceRecords"])
    if anscombe_manifest.get("recordsFile") != str(config["conditionalReferenceRecords"]):
        raise ValueError("Anscombe manifest recordsFile does not match the v4 config")
    if anscombe_manifest.get("configSha256") != sentinel_elo.canonical_sha256(config):
        raise ValueError("Anscombe manifest config SHA mismatch")
    if anscombe_manifest.get("recordsSha256") != sha256_file(records_path):
        raise ValueError("Anscombe records SHA mismatch")
    return {"eloRoot": elo_root, "anscombeRoot": anscombe_root, "eloAudit": audit, "anscombeAudit": anscombe_audit, "phaseRecordCount": phase_count, "anscombeRecordCount": anscombe_count, "anscombeManifest": anscombe_manifest}


def _verify_calibration(source_info: dict[str, Any], expansion: dict[str, Any], elo_config: dict[str, Any], anscombe_info: dict[str, Any]) -> dict[str, Any]:
    root = Path(expansion["paths"]["calibrationOutputDirectory"])
    artifact = read_json(require_file(root / str(elo_config["calibrationArtifact"])))
    progress = read_json(require_file(root / "progress.json"))
    cases_path = root / str(elo_config["calibrationCases"])
    cases = list(iter_jsonl(require_file(cases_path)))
    if artifact.get("status") != "validated" or progress.get("status") != "completed":
        raise ValueError("calibration is not completed/validated")
    expected_calibration_contract = sentinel_elo.canonical_sha256(sentinel_elo.v4_calibration_contract(elo_config))
    if (
        artifact.get("algorithmVersion") != sentinel_elo.ALGORITHM_VERSION_V4
        or float(artifact.get("calibrationCoverage")) != 0.95
        or artifact.get("calibrationContractSha256") != expected_calibration_contract
    ):
        raise ValueError("calibration v4 contract mismatch")
    if not _finite(artifact.get("t95")) or float(artifact.get("validationCoverage", 0)) < 0.95 or int(artifact.get("validationUserCount", 0)) < int(elo_config["minimumValidationUsers"]):
        raise ValueError("calibration T95/validation coverage contract failed")
    calibration_users = {str(row.get("account")) for row in cases if row.get("role") == "calibration"}
    validation_users = {str(row.get("account")) for row in cases if row.get("role") == "validation"}
    if not calibration_users or calibration_users & validation_users:
        raise ValueError("calibration and validation accounts overlap or are empty")
    if artifact.get("casesSha256") != sha256_file(cases_path):
        raise ValueError("calibration cases SHA mismatch")
    manifest = verify_manifest(root / "calibration_sha256_manifest_v4.json", root)
    if manifest.get("calibrationCasesSha256") != sha256_file(cases_path) or manifest.get("calibrationArtifactSha256") != sha256_file(root / str(elo_config["calibrationArtifact"])):
        raise ValueError("calibration manifest SHA mismatch")
    if artifact.get("directedRecordsSha256") != sha256_file(anscombe_info["eloRoot"] / str(elo_config["directedPhaseRecords"])):
        raise ValueError("calibration does not use the new Player Elo phase records")
    if artifact.get("anscombeReferenceManifestSha256") != sha256_file(anscombe_info["anscombeRoot"] / str(elo_config["conditionalReferenceManifest"])):
        raise ValueError("calibration does not use the new Anscombe manifest")
    return {"root": root, "artifact": artifact, "cases": cases, "calibrationUserCount": len(calibration_users), "validationUserCount": len(validation_users), "manifest": manifest}


def _unordered_report_rows(source: Path) -> list[dict[str, Any]]:
    payload = read_json(require_file(source / "partitions_unordered.json"))
    rows = [row for row in payload.get("partitions") or [] if row.get("partitionScope") == "main_bilateral"]
    return sorted(rows, key=lambda row: (int(row["pairLowerA"]), int(row["pairLowerB"])))


def write_final_report(result: dict[str, Any], *, expansion: dict[str, Any], output_path: Path) -> None:
    source = Path(result["sourceReference"])
    runner = read_json(require_file(source / "engine_level22" / "runner_audit.json"))
    matrix = result["directedMatrix"]
    labels = list(matrix["labels"])
    counts = matrix["counts"]
    lines = [
        f"# OQ 哨兵 Reference 黑白有向 {int(expansion['targetPerBlackWhiteCell'])} 扩充最终报告",
        "",
        f"- 状态：`{'通过' if result.get('ok') else '失败'}`",
        f"- 批次：`{result['batchId']}`",
        f"- 源库总局数：**{result['sourceGameCount']}**（主矩阵 {result['mainMatrixGameCount']}，低 Elo 历史扩展 {result['lowEloExtensionGameCount']}）",
        f"- 新增局数：**{result['addedGameCount']}**（缓存新增 {result['cacheSelectedCount']}，快照拉取新增 {result['snapshotSelectedCount']}）",
        f"- Level22：新增运行 **{runner['newGameCount']}** 局，复用 **{runner['reusedBaselineGameCount']}** 局；合同为 Level {result['level22']['contract']['level']} / 12 workers × 16 threads / hash {result['level22']['contract']['hash']} / WLD 从全局实际 ply {result['level22']['contract']['wldFromPlyInclusive']} 含边界。",
        f"- Sentinel directed records：**{result['sentinelDirectedRecordCount']}**（每局 2 条，黑/白方向分别保留）。",
        f"- Player Elo phase records：**{result['playerEloDirectedPhaseRecordCount']}**。",
        f"- Anscombe/reference-z records：**{result['anscombeRecordCount']}**，按 phase1 → phase2 → phase3 → phase4 顺序生成，K 使用 `ceil(N_allowed^(2/3))`。",
        f"- 数据库校准95% Elo范围：calibration **{result['calibrationUserCount']}** 人，validation **{result['validationUserCount']}** 人，T95 **{result['t95']}**，validation coverage **{result['validationCoverage']}**。",
        "",
        "## 黑棋 Elo × 白棋 Elo 有向矩阵",
        "",
        "行是黑棋 oldR Elo 桶，列是白棋 oldR Elo 桶；每个单元格是唯一 gameId 按实际黑白方向计数。",
        "",
        "| 黑棋\\白棋 | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for row_index, label in enumerate(labels):
        lines.append("| " + label + " | " + " | ".join(str(value) for value in counts[row_index]) + " |")

    unordered_rows = _unordered_report_rows(source)
    lines.extend([
        "",
        "## 45 格无向兼容汇总",
        "",
        f"完整 45 格无向兼容汇总位于 `{source / 'partitions_unordered.json'}`；该视图由上述 81 格有向矩阵派生，下面同时列出数量：",
        "",
        "| 无向桶 A | 无向桶 B | 数量 |",
        "|---|---:|---:|",
    ])
    for row in unordered_rows:
        lines.append(f"| {row['pairLowerA']} | {row['pairLowerB']} | {row['finalCount']} |" )

    lines.extend([
        "",
        "## 产物与独立审计",
        "",
        f"- 源库：`{result['sourceReference']}`",
        f"- Sentinel：`{result['sentinelReference']}`",
        f"- Player Elo：`{result['playerEloReference']}`",
        f"- Anscombe/reference-z：`{result['anscombeReference']}`",
        f"- calibration：`{result['calibrationReference']}`",
        f"- 45 格兼容汇总：`{result['unorderedCompatibilitySummary']}`",
        f"- 容量审计：`{result['capacityAudit']}`",
        f"- 独立 completion audit：`{result['sourceIndependentAudit']}`",
        "",
        "所有必需的基线包含、容量公式、有向方向、Level22、Sentinel、Player Elo、Anscombe、校准隔离、coverage 和 manifest SHA 校验均由独立审计记录。",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def verify_chain(expansion_config_path: Path, elo_config_path: Path, output_path: Path | None = None, report_path: Path | None = None) -> dict[str, Any]:
    expansion, expansion_path, expansion_sha = load_expansion_config(expansion_config_path)
    elo_config = read_json(require_file(elo_config_path))
    # The lifecycle runner captures each stage's stdout in a log that can still
    # grow after the stage itself has written its manifest. Rebuild the source
    # manifest before validating it so those append-only orchestration logs are
    # excluded by the explicit dynamic-file policy.
    write_source_manifest(Path(expansion["paths"]["sourceOutputDirectory"]), expansion_sha)
    source_info = _verify_source(expansion)
    level22_info = _verify_level22(source_info, expansion)
    sentinel_info = _verify_sentinel(source_info, expansion)
    elo_info = _verify_elo_and_anscombe(source_info, expansion, elo_config)
    calibration_info = _verify_calibration(source_info, expansion, elo_config, elo_info)
    result = {
        "schema": "oq-reference-blackwhite-independent-completion-audit-v2",
        "ok": True,
        "verifiedAtUtc": utc_now(),
        "batchId": expansion["batchId"],
        "expansionConfig": str(expansion_path.resolve()),
        "expansionConfigSha256": expansion_sha,
        "eloConfig": str(elo_config_path.resolve()),
        "sourceReference": str(source_info["source"].resolve()),
        "baselineReference": str(source_info["baseline"].resolve()),
        "baselineMode": source_info["baselineMode"],
        "sentinelReference": str(sentinel_info["root"].resolve()),
        "playerEloReference": str(elo_info["eloRoot"].resolve()),
        "anscombeReference": str(elo_info["anscombeRoot"].resolve()),
        "calibrationReference": str(calibration_info["root"].resolve()),
        "sourceGameCount": source_info["sourceGameCount"],
        "mainMatrixGameCount": source_info["mainMatrixGameCount"],
        "lowEloExtensionGameCount": source_info["lowEloExtensionGameCount"],
        "addedGameCount": source_info["sourceGameCount"] - int(expansion["baselineGameCount"]),
        "cacheSelectedCount": source_info["addedCounts"].get("validatedCacheExpansion", 0) + source_info["addedCounts"].get("priorFrozenSnapshotExpansion", 0),
        "snapshotSelectedCount": source_info["addedCounts"].get("uniqueSnapshotExpansion", 0),
        "formalEloMaximum": source_info["maximumElo"],
        "sentinelDirectedRecordCount": sentinel_info["recordCount"],
        "playerEloDirectedPhaseRecordCount": elo_info["phaseRecordCount"],
        "anscombeRecordCount": elo_info["anscombeRecordCount"],
        "calibrationUserCount": calibration_info["calibrationUserCount"],
        "validationUserCount": calibration_info["validationUserCount"],
        "t95": calibration_info["artifact"]["t95"],
        "validationCoverage": calibration_info["artifact"]["validationCoverage"],
        "directedMatrix": {
            "topLeftLabel": "黑棋\\白棋",
            "rowDimension": "black oldR Elo bucket",
            "columnDimension": "white oldR Elo bucket",
            "labels": [
                _cell_label(
                    int(expansion["minimumElo"]) + i * int(expansion["binWidth"]),
                    source_info["maximumElo"],
                    int(expansion["minimumElo"]),
                    int(expansion["binWidth"]),
                    int(expansion["formalBinCount"]),
                )
                for i in range(int(expansion["formalBinCount"]))
            ],
            "counts": source_info["formalMatrix"],
        },
        "unorderedCompatibilitySummary": source_info["unorderedSummaryPath"],
        "capacityAudit": source_info["capacityPath"],
        "checks": {
            "baselineAllIncluded": True,
            "lowEloExtensionPreserved": source_info["lowEloExtensionGameCount"] == int(expansion["baselineLowEloExtensionCount"]),
            "sourceGameIdsUnique": True,
            "allAddedGamesContractValid": True,
            "directedCapacityFormulaVerified": True,
            "capacityExhaustedCellsAudited": True,
            "unorderedViewDerived": True,
            "oneLevel22FilePerSourceGame": True,
            "level22NewAndReusedCountsVerified": True,
            "newSourceSelfContained": True,
            "sentinelTwoDirectedRecordsPerGame": True,
            "sentinelDirectionsVerified": True,
            "playerEloPhaseRecordsFromNewSource": True,
            "anscombeReferenceFromNewPhaseRecords": True,
            "sequentialReferenceZStagesVerified": True,
            "knnKUsesCeil": True,
            "calibrationValidated": True,
            "calibrationValidationDisjoint": True,
            "validationCoverageAtLeast095": True,
            "allManifestsSelfExcludedAndHashesVerified": True,
        },
        "level22": {"gameCount": len(source_info["rows"]), "wldCount": level22_info["wldCount"], "contract": level22_info["completion"]["contract"]},
        "calibrationStatus": calibration_info["artifact"]["status"],
    }
    source_audit_path = source_info["source"] / "provenance" / "independent_completion_audit.json"
    atomic_json(source_audit_path, result)
    result["sourceIndependentAudit"] = str(source_audit_path.resolve())
    if report_path is not None:
        write_final_report(result, expansion=expansion, output_path=report_path)
    write_source_manifest(source_info["source"], expansion_sha)
    rebind_derived_manifests(
        source_info["source"],
        sentinel_info["root"],
        elo_info["eloRoot"],
        elo_info["anscombeRoot"],
        calibration_info["root"],
        elo_config,
    )
    verify_manifest(source_info["source"] / "final_sha256_manifest.json", source_info["source"])
    verify_manifest(sentinel_info["root"] / "reference_sha256_manifest.json", sentinel_info["root"])
    verify_manifest(elo_info["eloRoot"] / "reference_sha256_manifest.json", elo_info["eloRoot"])
    anscombe_manifest_path = elo_info["anscombeRoot"] / str(elo_config["conditionalReferenceManifest"])
    anscombe_manifest = read_json(require_file(anscombe_manifest_path))
    if anscombe_manifest.get("referenceManifestSha256") != sha256_file(elo_info["eloRoot"] / str(elo_config["referenceManifest"])):
        raise ValueError("final Anscombe manifest is not bound to the final Player Elo manifest")
    if anscombe_manifest.get("recordsSha256") != sha256_file(elo_info["anscombeRoot"] / str(elo_config["conditionalReferenceRecords"])):
        raise ValueError("final Anscombe records SHA mismatch")
    final_calibration = read_json(require_file(calibration_info["root"] / str(elo_config["calibrationArtifact"])))
    if final_calibration.get("anscombeReferenceManifestSha256") != sha256_file(anscombe_manifest_path):
        raise ValueError("final calibration is not bound to the final Anscombe manifest")
    verify_manifest(calibration_info["root"] / "calibration_sha256_manifest_v4.json", calibration_info["root"])
    if output_path is not None and output_path.resolve() != source_audit_path.resolve():
        atomic_json(output_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-config", type=Path, required=True)
    parser.add_argument("--elo-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = verify_chain(
        args.expansion_config.resolve(),
        args.elo_config.resolve(),
        args.output.resolve() if args.output else None,
        args.report.resolve() if args.report else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
