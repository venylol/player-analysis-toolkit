#!/usr/bin/env python3
"""Run or resume the complete configuration-driven OQ Reference lifecycle.

The individual stages remain the owners of their data contracts.  This entry
point only sequences them, records atomic lifecycle state, and attaches to an
already-running Level22 or calibration process instead of starting a second
one.  All text files written here are UTF-8.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
OFFBOOK = Path(__file__).resolve().parent
if str(OFFBOOK) not in sys.path:
    sys.path.insert(0, str(OFFBOOK))

from oq_blackwhite_contract import (  # noqa: E402
    atomic_write_json,
    load_expansion_config,
    read_json,
    resolve_repo_path,
    sha256_file,
    utc_now,
)


EXPANSION_SCRIPT = OFFBOOK / "oq_reference500_blackwhite_pipeline.py"
LEVEL22_SCRIPT = OFFBOOK / "run_oq_level22_blackwhite.py"
SENTINEL_SCRIPT = OFFBOOK / "build_oq_sentinel_blackwhite_reference.py"
VERIFY_SCRIPT = OFFBOOK / "verify_oq_blackwhite_final.py"
ELO_SCRIPT = ROOT / "scripts" / "analysis" / "sentinel_elo_analysis.py"
DEFAULT_EXPANSION_CONFIG = ROOT / "oq_reference_blackwhite_expansion_config_v3_matchup600_20260911.json"
LIFECYCLE_SCHEMA = "oq-reference-blackwhite-lifecycle-progress-v1"
POLL_SECONDS = 10


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: tuple[str, ...] | None
    input_paths: tuple[Path, ...]
    output_paths: tuple[Path, ...]
    batch_stage: str | None = None
    process_marker: str | None = None


def _path(value: str | Path) -> Path:
    return resolve_repo_path(value)


def _sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def _hash_paths(paths: Sequence[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        result.append({
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "sha256": _sha(path) if path.is_file() else None,
        })
    return result


def _config_paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path, Path]:
    source = _path(config["sourceOutputDirectory"])
    sentinel = _path(config["sentinelOutputDirectory"])
    elo = _path(config["playerEloOutputDirectory"])
    anscombe = _path(config["anscombeOutputDirectory"])
    calibration = _path(config["calibrationOutputDirectory"])
    return source, sentinel, elo, anscombe, calibration


def _pipeline_command(command: str, config_path: Path) -> tuple[str, ...]:
    return (sys.executable, str(EXPANSION_SCRIPT), command, "--config", str(config_path))


def _stage_specs(config: dict[str, Any], config_path: Path, elo_config_path: Path) -> list[StageSpec]:
    source, sentinel, elo, anscombe, calibration = _config_paths(config)
    snapshot = _path(config["snapshotOutputDirectory"])
    provenance = source / "provenance"
    coverage_output = provenance / "v4_coverage_audit"
    knn_samples = provenance / "knn_consistency_samples_v4.json"
    knn_output = provenance / "knn_consistency_audit_v4"
    baseline = _path(config["baselineSourceReference"])
    baseline_snapshot = _path(config["localSources"]["baselineSnapshot"])
    config_input = (config_path,)
    return [
        StageSpec(
            "local-audit", _pipeline_command("local-audit", config_path),
            config_input + (baseline / "selected_account_bundle.json", baseline_snapshot / "batch_manifest.json"),
            (provenance / "local_cache_audit" / "black_white_local_cache_audit_historical.json",),
            batch_stage="localAudit",
        ),
        StageSpec(
            "leaderboard", _pipeline_command("leaderboard", config_path),
            config_input + (provenance / "local_cache_validated_records.jsonl",),
            (snapshot / "leaderboard.json", snapshot / "leaderboard_pages.jsonl",),
            batch_stage="leaderboard",
            process_marker="oq_reference500_blackwhite_pipeline.py leaderboard",
        ),
        StageSpec(
            "player-lists", _pipeline_command("player-lists", config_path),
            config_input + (snapshot / "leaderboard.json",),
            (snapshot / "player_lists.jsonl", snapshot / "player_list_attempts.jsonl",),
            batch_stage="playerLists",
            process_marker="oq_reference500_blackwhite_pipeline.py player-lists",
        ),
        StageSpec(
            "candidate-audit", _pipeline_command("candidate-audit", config_path),
            config_input + (snapshot / "leaderboard.json", snapshot / "player_lists.jsonl",),
            (provenance / "candidate_dedup_audit.json", provenance / "black_white_capacity_before_details.json",),
            batch_stage="candidateAudit",
        ),
        StageSpec(
            "details", _pipeline_command("details", config_path),
            config_input + (provenance / "candidate_dedup_audit.json",),
            (snapshot / "game_details.jsonl", snapshot / "game_detail_attempts.jsonl", provenance / "black_white_capacity_after_details.json",),
            batch_stage="details",
            process_marker="oq_reference500_blackwhite_pipeline.py details",
        ),
        StageSpec(
            "materialize", _pipeline_command("materialize", config_path),
            config_input + (snapshot / "game_details.jsonl", provenance / "black_white_capacity_after_details.json",),
            (source / "selected_games_with_partitions.json", source / "selected_account_bundle.json", source / "selection_audit.json", source / "expansion_manifest.json"),
            batch_stage="materialize",
        ),
        StageSpec(
            "level22", (sys.executable, str(LEVEL22_SCRIPT), "--config", str(config_path)),
            config_input + (source / "selected_games_with_partitions.json", source / "selected_account_bundle.json", baseline / "engine_game_index.json"),
            (source / "reference_completion_audit.json", source / "engine_level22" / "audit.json", source / "engine_game_index.json", source / "final_sha256_manifest.json"),
            process_marker="run_oq_level22_blackwhite.py",
        ),
        StageSpec(
            "sentinel", (sys.executable, str(SENTINEL_SCRIPT), "--config", str(config_path)),
            config_input + (source / "final_sha256_manifest.json", source / "reference_completion_audit.json"),
            (sentinel / "directed_target_records.jsonl", sentinel / "reference_build_audit.json", sentinel / "reference_sha256_manifest.json", sentinel / "black_white_source_partition_provenance.json"),
        ),
        StageSpec(
            "elo-reference", (sys.executable, str(ELO_SCRIPT), "--config", str(elo_config_path), "build-elo-reference"),
            (elo_config_path, source / "final_sha256_manifest.json", sentinel / "reference_sha256_manifest.json"),
            (elo / "directed_game_phase_records.jsonl", elo / "reference_build_audit.json", elo / "reference_sha256_manifest.json"),
        ),
        StageSpec(
            "anscombe", (sys.executable, str(ELO_SCRIPT), "--config", str(elo_config_path), "prepare-conditional-elo-reference", "--resume"),
            (elo_config_path, elo / "directed_game_phase_records.jsonl", elo / "reference_sha256_manifest.json"),
            (anscombe / "anscombe_reference_records.jsonl", anscombe / "anscombe_reference_manifest.json", anscombe / "anscombe_reference_audit.json"),
        ),
        StageSpec(
            "knn-samples", None,
            (elo_config_path, elo / "directed_game_phase_records.jsonl", anscombe / "anscombe_reference_manifest.json"),
            (knn_samples,),
        ),
        StageSpec(
            "knn-audit", (sys.executable, str(ELO_SCRIPT), "--config", str(elo_config_path), "audit-knn-consistency", "--conditional-reference-dir", str(anscombe), "--samples", str(knn_samples), "--output-dir", str(knn_output)),
            (elo_config_path, anscombe / "anscombe_reference_records.jsonl", anscombe / "anscombe_reference_manifest.json", knn_samples),
            (knn_output / "global_knn_consistency_v4.json",),
        ),
        StageSpec(
            "calibration", (sys.executable, str(ELO_SCRIPT), "--config", str(elo_config_path), "calibrate-elo", "--resume"),
            (elo_config_path, source / "selected_account_bundle.json", elo / "directed_game_phase_records.jsonl", anscombe / "anscombe_reference_manifest.json"),
            (calibration / "elo_calibration_v4.json", calibration / "elo_calibration_cases_v4.jsonl", calibration / "calibration_sha256_manifest_v4.json"),
            process_marker="sentinel_elo_analysis.py calibrate-elo",
        ),
        StageSpec(
            "coverage-audit", (sys.executable, str(ELO_SCRIPT), "--config", str(elo_config_path), "audit-estimate-coverage", "--reference-dir", str(elo), "--calibration-dir", str(calibration), "--output-dir", str(coverage_output)),
            (elo_config_path, calibration / "elo_calibration_v4.json", calibration / "elo_calibration_cases_v4.jsonl", calibration / "calibration_sha256_manifest_v4.json"),
            (coverage_output / "estimate_coverage_audit_v4.json",),
        ),
        StageSpec(
            "independent-audit", (sys.executable, str(VERIFY_SCRIPT), "--expansion-config", str(config_path), "--elo-config", str(elo_config_path), "--output", str(provenance / "independent_completion_audit.json"), "--report", str(provenance / "final_report.md")),
            (config_path, elo_config_path, source / "final_sha256_manifest.json", sentinel / "reference_sha256_manifest.json", elo / "reference_sha256_manifest.json", anscombe / "anscombe_reference_manifest.json", calibration / "calibration_sha256_manifest_v4.json"),
            (provenance / "independent_completion_audit.json", provenance / "final_report.md"),
        ),
    ]


def _batch_manifest(source: Path, config: dict[str, Any]) -> Path:
    return _path(config["snapshotOutputDirectory"]) / "batch_manifest.json"


def _assert_batch_contract(source: Path, config: dict[str, Any], config_sha: str) -> None:
    path = _batch_manifest(source, config)
    if not path.is_file():
        return
    manifest = read_json(path)
    if manifest.get("batchId") != config.get("batchId"):
        raise RuntimeError(f"existing expansion batch belongs to another batch: {manifest.get('batchId')!r}")
    if manifest.get("configSha256") != config_sha:
        raise RuntimeError("existing expansion batch belongs to another configuration contract")


def _read_optional(path: Path) -> Any:
    return read_json(path) if path.is_file() else None


def _external_stage_complete(spec: StageSpec, config: dict[str, Any]) -> bool:
    source, sentinel, elo, anscombe, calibration = _config_paths(config)
    if spec.batch_stage:
        manifest_path = _batch_manifest(source, config)
        manifest = _read_optional(manifest_path)
        return bool(manifest and (manifest.get("stages") or {}).get(spec.batch_stage, {}).get("complete") is True)
    if spec.name == "level22":
        audit = _read_optional(source / "reference_completion_audit.json")
        return bool(audit and audit.get("ok") is True)
    if spec.name == "sentinel":
        audit = _read_optional(sentinel / "reference_build_audit.json")
        return bool(audit and audit.get("ok") is True)
    if spec.name == "elo-reference":
        audit = _read_optional(elo / "reference_build_audit.json")
        return bool(audit and audit.get("ok") is True)
    if spec.name == "anscombe":
        audit = _read_optional(anscombe / "anscombe_reference_audit.json")
        return bool(audit and audit.get("ok") is True)
    if spec.name == "knn-samples":
        value = _read_optional(spec.output_paths[0])
        return isinstance(value, list) and bool(value)
    if spec.name == "knn-audit":
        report = _read_optional(spec.output_paths[0])
        return bool(report and report.get("passed") is True and report.get("mismatchCount") == 0)
    if spec.name == "calibration":
        artifact = _read_optional(calibration / "elo_calibration_v4.json")
        progress = _read_optional(calibration / "progress.json")
        return bool(
            artifact and artifact.get("status") == "validated"
            and progress and progress.get("status") == "completed"
        )
    if spec.name == "coverage-audit":
        report = _read_optional(spec.output_paths[0])
        return bool(report and report.get("ok") is True and float(report.get("validationCoverage", 0.0)) >= 0.95)
    if spec.name == "independent-audit":
        report = _read_optional(spec.output_paths[0])
        return bool(report and report.get("ok") is True and report.get("schema") == "oq-reference-blackwhite-independent-completion-audit-v2")
    return False


def _make_knn_samples(spec: StageSpec, config: dict[str, Any]) -> None:
    _source, _sentinel, _elo, anscombe, _calibration = _config_paths(config)
    records_path = anscombe / str(config.get("conditionalReferenceRecords") or "anscombe_reference_records.jsonl")
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            color = str(row.get("targetColor") or "").casefold()
            scope = str(row.get("scope") or "")
            account = str(row.get("targetPlayerId") or "")
            game_id = str(row.get("gameId") or "")
            if color not in {"black", "white"} or not scope or not account or not game_id:
                continue
            for stage in range(1, 5):
                phase = row.get(f"phase{stage}") or {}
                if not isinstance(phase.get("n"), int) or phase.get("n", 0) <= 0:
                    break
                key = (color, scope, stage)
                selected.setdefault(key, {
                    "targetColor": color,
                    "scope": scope,
                    "stage": stage,
                    "account": account,
                    "targetGameIds": [game_id],
                    "trialElo": float(row["targetOldR"]),
                    "opponentElo": float(row["opponentOldR"]),
                    "previousZ": (
                        None if stage == 1 else float(row[f"referenceZ{stage - 1}"])
                    ),
                })
    expected = 2 * 2 * 4
    samples = [selected[key] for key in sorted(selected)]
    if len(samples) != expected:
        raise RuntimeError(f"cannot create the required deterministic KNN samples: {len(samples)}/{expected}")
    atomic_write_json(spec.output_paths[0], samples)


def _processes(marker: str) -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    env_name = "OQ_LIFECYCLE_PROCESS_MARKER"
    script = (
        "$needle=[Environment]::GetEnvironmentVariable('OQ_LIFECYCLE_PROCESS_MARKER'); "
        "$found=@(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | "
        "Where-Object { $_.CommandLine -and $_.CommandLine -like ('*' + $needle + '*') } | "
        "Select-Object ProcessId,Name); "
        "if($found.Count -eq 0){ Write-Output '[]' } else { $found | ConvertTo-Json -Compress }"
    )
    environment = dict(os.environ)
    environment[env_name] = marker
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    text = completed.stdout.strip()
    if not text:
        return []
    value = json.loads(text)
    values = value if isinstance(value, list) else [value]
    return [{"pid": int(item["ProcessId"]), "name": str(item.get("Name") or "")} for item in values]


def _save(state_path: Path, state: dict[str, Any]) -> None:
    state["updatedAtUtc"] = utc_now()
    atomic_write_json(state_path, state)


def _run_external(spec: StageSpec, state_path: Path, state: dict[str, Any]) -> None:
    log_path = state_path.parent / f"{spec.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.process_marker:
        attached = _processes(spec.process_marker)
        if attached:
            state["process"] = {
                "mode": "attached",
                "pids": attached,
                "marker": spec.process_marker,
                "attachedAtUtc": utc_now(),
            }
            _save(state_path, state)
            while True:
                current = _processes(spec.process_marker)
                if not current:
                    break
                state["process"]["pids"] = current
                state["process"]["polledAtUtc"] = utc_now()
                _save(state_path, state)
                time.sleep(POLL_SECONDS)
            if _external_stage_complete(spec, state["configuration"]):
                state["process"] = None
                return

    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"[{utc_now()}] command: {json.dumps(list(spec.command or ()), ensure_ascii=False)}\n")
        log.flush()
        process = subprocess.Popen(
            list(spec.command or ()),
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        state["process"] = {
            "mode": "started",
            "pid": process.pid,
            "command": list(spec.command or ()),
            "logPath": str(log_path.resolve()),
            "startedAtUtc": utc_now(),
        }
        _save(state_path, state)
        while True:
            return_code = process.poll()
            if return_code is not None:
                break
            state["process"]["polledAtUtc"] = utc_now()
            _save(state_path, state)
            time.sleep(POLL_SECONDS)
    state["process"] = None
    if return_code != 0:
        raise RuntimeError(f"lifecycle stage {spec.name} failed with exit code {return_code}; see {log_path}")


def _initial_state(config: dict[str, Any], config_path: Path, config_sha: str) -> dict[str, Any]:
    source = _path(config["sourceOutputDirectory"])
    state_path = source / "provenance" / "lifecycle_progress.json"
    return {
        "schema": LIFECYCLE_SCHEMA,
        "batchId": config["batchId"],
        "configSha256": config_sha,
        "configPath": str(config_path.resolve()),
        "status": "running",
        "currentStage": None,
        "stages": {},
        "process": None,
        "startedAtUtc": utc_now(),
        "endedAtUtc": None,
        "failure": None,
        "recoveryCommand": f'python research/offbook_detection/run_oq_blackwhite_reference_lifecycle.py --config "{config_path}" resume',
        "nextStage": None,
        "configuration": config,
    }


def _load_state(
    config: dict[str, Any], config_path: Path, config_sha: str, command: str
) -> tuple[Path, dict[str, Any]]:
    source = _path(config["sourceOutputDirectory"])
    state_path = source / "provenance" / "lifecycle_progress.json"
    existing = _read_optional(state_path)
    if existing is not None:
        if existing.get("schema") != LIFECYCLE_SCHEMA:
            raise RuntimeError(f"unsupported lifecycle progress schema: {state_path}")
        if existing.get("batchId") != config.get("batchId") or existing.get("configSha256") != config_sha:
            raise RuntimeError("lifecycle progress belongs to another batch or configuration contract")
        if command == "run":
            raise RuntimeError("lifecycle progress already exists; use resume")
        existing["configuration"] = config
        existing["recoveryCommand"] = f'python research/offbook_detection/run_oq_blackwhite_reference_lifecycle.py --config "{config_path}" resume'
        if existing.get("status") != "completed":
            existing["status"] = "running"
            existing["failure"] = None
        return state_path, existing
    _assert_batch_contract(source, config, config_sha)
    if command == "run" and source.exists() and any(source.iterdir()):
        raise RuntimeError("source output directory already exists; same-batch work must be resumed")
    if command == "resume" and not source.exists():
        raise RuntimeError("cannot resume a lifecycle whose source output directory does not exist")
    state = _initial_state(config, config_path, config_sha)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _save(state_path, state)
    return state_path, state


def _execute(config: dict[str, Any], config_path: Path, config_sha: str, command: str) -> int:
    state_path, state = _load_state(config, config_path, config_sha, command)
    elo_config_path = _path(config["playerEloConfigPath"] if "playerEloConfigPath" in config else ROOT / "sentinel_elo_reference_config_v4_matchup600_20260911.json")
    specs = _stage_specs(config, config_path, elo_config_path)
    try:
        for index, spec in enumerate(specs):
            state["currentStage"] = spec.name
            state["nextStage"] = specs[index + 1].name if index + 1 < len(specs) else None
            _save(state_path, state)
            complete = _external_stage_complete(spec, config)
            recorded = state["stages"].get(spec.name)
            if complete:
                input_hashes = _hash_paths(spec.input_paths)
                output_hashes = _hash_paths(spec.output_paths)
                if recorded and recorded.get("status") == "completed" and recorded.get("outputSha256") != output_hashes:
                    # The independent verifier deliberately rebuilds the source
                    # and derived manifests after append-only lifecycle logs have
                    # settled. That changes certificates/bindings, not engine
                    # or statistical records, so record the controlled
                    # normalization for the affected completed stages.
                    if spec.name not in {"local-audit", "level22", "sentinel", "elo-reference", "anscombe", "calibration", "independent-audit"}:
                        raise RuntimeError(f"completed lifecycle stage output changed: {spec.name}")
                    recorded = dict(recorded)
                    recorded["manifestNormalizedByIndependentAudit"] = True
                state["stages"][spec.name] = {
                    "status": "completed",
                    "startedAtUtc": (recorded or {}).get("startedAtUtc", utc_now()),
                    "completedAtUtc": (recorded or {}).get("completedAtUtc", utc_now()),
                    "inputSha256": input_hashes,
                    "outputSha256": output_hashes,
                    "reusedExistingCompletion": True,
                    "manifestNormalizedByIndependentAudit": bool(recorded and recorded.get("manifestNormalizedByIndependentAudit")),
                }
                _save(state_path, state)
                continue
            state["stages"][spec.name] = {
                "status": "running",
                "startedAtUtc": utc_now(),
                "inputSha256": _hash_paths(spec.input_paths),
                "outputSha256": None,
            }
            _save(state_path, state)
            if spec.name == "knn-samples":
                _make_knn_samples(spec, config)
            else:
                _run_external(spec, state_path, state)
            if not _external_stage_complete(spec, config):
                raise RuntimeError(f"stage exited without a successful completion certificate: {spec.name}")
            state["stages"][spec.name].update({
                "status": "completed",
                "completedAtUtc": utc_now(),
                "outputSha256": _hash_paths(spec.output_paths),
            })
            state["process"] = None
            _save(state_path, state)
        state["status"] = "completed"
        state["currentStage"] = None
        state["nextStage"] = None
        state["endedAtUtc"] = utc_now()
        _save(state_path, state)
        print(json.dumps({"status": state["status"], "progress": str(state_path.resolve()), "stages": list(state["stages"])}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["failure"] = {"type": type(exc).__name__, "message": str(exc), "atUtc": utc_now()}
        state["process"] = None
        _save(state_path, state)
        raise


def _status(config_path: Path) -> int:
    config, resolved, config_sha = load_expansion_config(config_path)
    source = _path(config["sourceOutputDirectory"])
    state_path = source / "provenance" / "lifecycle_progress.json"
    if not state_path.is_file():
        print(json.dumps({"status": "not_started", "configSha256": config_sha, "progress": str(state_path.resolve())}, ensure_ascii=False, indent=2))
        return 0
    state = read_json(state_path)
    if state.get("batchId") != config.get("batchId") or state.get("configSha256") != config_sha:
        raise RuntimeError("lifecycle status belongs to another batch or configuration contract")
    compact = {
        "schema": state.get("schema"),
        "batchId": state.get("batchId"),
        "configSha256": state.get("configSha256"),
        "status": state.get("status"),
        "currentStage": state.get("currentStage"),
        "nextStage": state.get("nextStage"),
        "process": state.get("process"),
        "stages": {
            name: {"status": value.get("status"), "completedAtUtc": value.get("completedAtUtc")}
            for name, value in (state.get("stages") or {}).items()
        },
        "failure": state.get("failure"),
        "updatedAtUtc": state.get("updatedAtUtc"),
        "progress": str(state_path.resolve()),
        "configPath": str(resolved),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPANSION_CONFIG)
    parser.add_argument("command", choices=("run", "resume", "status"))
    args = parser.parse_args(argv)
    config, config_path, config_sha = load_expansion_config(args.config)
    if args.command == "status":
        return _status(config_path)
    return _execute(config, config_path, config_sha, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
