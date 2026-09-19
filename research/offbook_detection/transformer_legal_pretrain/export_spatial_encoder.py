"""Export a trained spatial Transformer without its training-only legal head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forced-delivery-reason", required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    test_evaluation = args.test_evaluation.resolve(strict=True)
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing encoder export or manifest")

    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if source.get("format") != "spatial-transformer-legal-checkpoint-v1":
        raise ValueError("unsupported source checkpoint")
    legal_head_keys = sorted(
        key for key in source["model_state_dict"] if key.startswith("legal_head.")
    )
    if legal_head_keys != ["legal_head.bias", "legal_head.weight"]:
        raise ValueError(f"unexpected legal-head keys: {legal_head_keys}")
    encoder_state = {
        key: value for key, value in source["model_state_dict"].items()
        if not key.startswith("legal_head.")
    }
    payload = {
        "format": "spatial-transformer-board-encoder-v1",
        "encoder_state_dict": encoder_state,
        "model_config": source["config"]["model"],
        "board_embedding_dim": int(source["config"]["model"]["hidden_size"]),
        "board_contract": {
            "squares": 64,
            "boardOrder": "a1,b1,...,h8",
            "occupancyCodes": {"empty": 0, "targetPlayer": 1, "opponent": 2},
            "actorIsTargetRequired": True,
            "output": "CLS token after final encoder normalization",
        },
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_epoch": int(source["epoch"]),
        "source_global_step": int(source["global_step"]),
        "forced_delivery": True,
        "forced_delivery_reason": args.forced_delivery_reason,
        "legal_head_removed": True,
        "optimizer_state_removed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    evaluation = json.loads(test_evaluation.read_text(encoding="utf-8"))
    gates = source["config"]["validation_gates"]
    validation = source["best_validation"]["targetRole"]
    equivariance = source["best_validation"]["equivariance"]
    gate_results = {
        "exactLegalSetAccuracy": validation["exactLegalSetAccuracy"] >= gates["exact_legal_set_accuracy"],
        "f1": validation["f1"] >= gates["f1"],
        "illegalCellFalsePositiveRate": validation["illegalCellFalsePositiveRate"] <= gates["illegal_cell_false_positive_rate"],
        "passExactLegalSetAccuracy": validation["passExactLegalSetAccuracy"] >= gates["pass_exact_legal_set_accuracy"],
        "d4MaskConsistency": equivariance["d4MaskConsistency"] >= gates["d4_mask_consistency"],
        "roleMaskConsistency": equivariance["roleMaskConsistency"] >= gates["role_mask_consistency"],
    }
    manifest = {
        "schema": "spatial-transformer-board-encoder-delivery-v1",
        "status": "forced-delivery-validation-gates-not-passed",
        "forcedDeliveryReason": args.forced_delivery_reason,
        "sourceCheckpoint": str(checkpoint),
        "sourceCheckpointSha256": sha256_file(checkpoint),
        "sourceEpochHumanNumber": int(source["epoch"]) + 1,
        "sourceGlobalStep": int(source["global_step"]),
        "encoder": str(output),
        "encoderSha256": sha256_file(output),
        "encoderBytes": output.stat().st_size,
        "boardEmbeddingDim": payload["board_embedding_dim"],
        "removedStateKeys": legal_head_keys,
        "optimizerStateRemoved": True,
        "validation": validation,
        "validationEquivariance": equivariance,
        "validationGateResults": gate_results,
        "allValidationGatesPassed": all(gate_results.values()),
        "fixedTestEvaluation": str(test_evaluation),
        "fixedTestEvaluationSha256": sha256_file(test_evaluation),
        "fixedTest": evaluation,
        "encoding": "UTF-8",
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
