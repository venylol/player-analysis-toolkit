#!/usr/bin/env python3
"""Refresh replay-review candidate anchors with the active off-book detector."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLKIT_ROOT = Path(__file__).resolve().parents[3]
DETECT_PATH = TOOLKIT_ROOT / "scripts" / "analysis" / "detect_offbook.py"
SPEC = importlib.util.spec_from_file_location("active_detect_offbook", DETECT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load detector: {DETECT_PATH}")
DETECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DETECT)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-bundle", type=Path, required=True)
    parser.add_argument("--engine-directory", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replay_path = args.replay_bundle.resolve()
    engine_directory = args.engine_directory.resolve()
    source_path = args.source_bundle.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    audit_path = engine_directory / "audit.json"
    if not audit_path.is_file() or read_json(audit_path).get("ok") is not True:
        raise ValueError(f"Level22 audit is missing or unsuccessful: {audit_path}")

    replay = read_json(replay_path)
    if replay.get("schema") != "offbook-replay-review-bundle-v1":
        raise ValueError("unsupported replay bundle schema")
    source = read_json(source_path)
    details = {
        str(item.get("id") or ""): item
        for item in source.get("details", [])
        if isinstance(item, dict)
    }
    engine_games = {
        str(game.get("gameId") or ""): game
        for game in DETECT.load_engine_games(engine_directory)
    }

    candidate_count = 0
    for game in replay.get("games", []):
        if str(game.get("account") or "").casefold() != args.account.casefold():
            raise ValueError(f"unexpected replay account: {game.get('account')!r}")
        game_id = str(game.get("gameId") or "")
        if game_id not in details or game_id not in engine_games:
            raise ValueError(f"missing source or Level22 game: {game_id}")
        record = DETECT.detect_game(engine_games[game_id], args.account, details[game_id].get("tcb"))
        has_candidate = record["algorithmLabel"] == "offbook"
        candidate_count += int(has_candidate)
        game["algorithm"] = {
            "hasCandidate": has_candidate,
            "candidatePly": record["offBookPly"],
            "candidateTargetDecision": record["targetDecisionNumber"],
            "candidateSource": record["anchorSource"],
            "candidateMove": record["move"],
            "candidateThinkingTimeMs": record["thinkingTimeMs"],
            "candidateBestEval": record["bestEval"],
            "timeLimitMs": record["timeLimitMs"],
            "timeThresholdMs": record["timeThresholdMs"],
            "fastThresholdMs": record["fastThresholdMs"],
            "labelSource": record["labelSource"],
            "postFastCheck": record["postFastCheck"],
            "candidateChecks": record["algorithmEvidence"]["candidateChecks"],
            "rejectedCandidates": record["algorithmEvidence"]["rejectedCandidates"],
            "evaluationCutoff": record["algorithmEvidence"]["evaluationCutoff"],
        }
        for side in ("black", "white"):
            summary = (game.get("sideTimeSummary") or {}).get(side)
            if isinstance(summary, dict):
                summary["initialTimeLimitMs"] = record["timeLimitMs"]

    replay["generatedAt"] = datetime.now(timezone.utc).isoformat()
    replay["purpose"] = "Offline review of candidate anchors from the active deterministic off-book detector"
    replay["counts"]["algorithmCandidates"] = candidate_count
    replay["sources"] = {
        "sourceReplayBundle": {
            "path": str(replay_path),
            "sha256": sha256_file(replay_path),
        },
        "sourceBundle": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "engineDirectory": str(engine_directory),
        "detector": {
            "path": str(DETECT_PATH),
            "sha256": sha256_file(DETECT_PATH),
            "label": DETECT.ALGORITHM_LABEL,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(replay, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "output": str(output),
        "gameCount": len(replay.get("games", [])),
        "algorithmCandidates": candidate_count,
        "algorithm": DETECT.ALGORITHM_LABEL,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
