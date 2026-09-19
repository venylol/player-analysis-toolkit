"""Train one frozen-protocol Student-t or Gaussian single-change-point model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.stage3_single_change_point.core import (  # noqa: E402
    DiagonalSegmentModel, candidate_decisions, marginal_nll,
)
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(cache_dir: Path, manifest: dict[str, object], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    plies: list[np.ndarray] = []
    for item in manifest["shards"]:  # type: ignore[index]
        with np.load(cache_dir / item["file"], allow_pickle=False) as shard:  # type: ignore[index]
            shard_splits = shard["split"]
            finite_counts = shard["finite_candidate_count"]
            offsets = shard["sequence_offset"]
            counts = shard["sequence_count"]
            shard_z = shard["z"]
            shard_plies = shard["strict_ply"]
            for sequence_index, value in enumerate(shard_splits.tolist()):
                if str(value) != split or int(finite_counts[sequence_index]) == 0:
                    continue
                offset = int(offsets[sequence_index])
                count = int(counts[sequence_index])
                sequences.append(shard_z[offset:offset + count].astype(np.float32))
                plies.append(shard_plies[offset:offset + count].astype(np.int16))
    if not sequences:
        raise ValueError(f"no eligible sequences in {split}")
    max_length = max(map(len, sequences))
    dimension = sequences[0].shape[1]
    values = np.zeros((len(sequences), max_length, dimension), dtype=np.float32)
    candidate_mask = np.zeros((len(sequences), max_length), dtype=np.bool_)
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    for row, (sequence, strict_ply) in enumerate(zip(sequences, plies)):
        values[row, :len(sequence)] = sequence
        candidates = candidate_decisions(strict_ply)
        candidate_mask[row, candidates - 1] = True
    return values, lengths, candidate_mask


@torch.inference_mode()
def evaluate(
    model: DiagonalSegmentModel, values: np.ndarray, lengths: np.ndarray, candidates: np.ndarray,
    batch_size: int, device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    total_nll = 0.0
    total_nodes = 0
    for start in range(0, len(values), batch_size):
        stop = start + batch_size
        batch_values = torch.from_numpy(values[start:stop]).to(device)
        batch_lengths = torch.from_numpy(lengths[start:stop]).to(device)
        batch_candidates = torch.from_numpy(candidates[start:stop]).to(device)
        nll, nodes = marginal_nll(model, batch_values, batch_lengths, batch_candidates)
        total_nll += float(nll)
        total_nodes += int(nodes)
    return {"sequences": len(values), "decisions": total_nodes, "total_nll": total_nll, "nll_per_decision": total_nll / total_nodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--family", choices=("student_t", "gaussian"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cache_dir = args.projected_cache.resolve(strict=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "stage3-projected-sequence-cache-v1" or manifest.get("status") != "complete":
        raise ValueError("complete projected cache required")
    if manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("projected cache and training configuration differ")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "best.pt").exists():
        raise FileExistsError("completed model output already exists")
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_values, train_lengths, train_candidates = load_split(cache_dir, manifest, "train")
    validation_values, validation_lengths, validation_candidates = load_split(cache_dir, manifest, "validation")
    model = DiagonalSegmentModel(
        int(config["projection_dim"]), args.family, float(config["student_t_degrees_of_freedom"]),
        float(config["scale_floor"]),
    ).to(device)
    flattened = train_values[np.arange(train_values.shape[1])[None, :] < train_lengths[:, None]]
    mean = torch.from_numpy(flattened.mean(0)).to(device)
    std = torch.from_numpy(flattened.std(0).clip(min=1e-3)).to(device)
    with torch.no_grad():
        model.location[:] = mean
        model.raw_scale[:] = torch.log(torch.expm1((std - model.scale_floor).clamp_min(1e-4)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    batch_size = int(config["batch_size"])
    run_config = {
        "schema": "stage3-single-change-point-training-v1", "family": args.family, "seed": seed,
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "projected_cache_manifest": str(manifest_path), "projected_cache_manifest_sha256": sha256_file(manifest_path),
        "device": str(device), "train_sequences": len(train_values), "validation_sequences": len(validation_values),
    }
    write_json(output_dir / "run_config.json", run_config)
    generator = np.random.default_rng(seed)
    best_metric = math.inf
    stale = 0
    metrics: list[dict[str, object]] = []
    for epoch in range(int(config["max_epochs"])):
        model.train()
        order = generator.permutation(len(train_values))
        train_nll = 0.0
        train_nodes = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            values = torch.from_numpy(train_values[indices]).to(device)
            lengths = torch.from_numpy(train_lengths[indices]).to(device)
            candidates = torch.from_numpy(train_candidates[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            nll, nodes = marginal_nll(model, values, lengths, candidates)
            loss = nll / nodes
            loss.backward()
            optimizer.step()
            train_nll += float(nll.detach())
            train_nodes += int(nodes)
        validation = evaluate(model, validation_values, validation_lengths, validation_candidates, batch_size, device)
        metric = float(validation["nll_per_decision"])
        item = {
            "epoch": epoch, "train_nll_per_decision": train_nll / train_nodes, "validation": validation,
            "location": model.location.detach().cpu().tolist(), "scale": model.scale().detach().cpu().tolist(),
        }
        metrics.append(item)
        write_json(output_dir / "metrics.json", metrics)
        checkpoint = {
            "format": "stage3-single-change-point-checkpoint-v1", "epoch": epoch, "family": args.family,
            "model_state_dict": model.state_dict(), "dimension": int(config["projection_dim"]),
            "degrees_of_freedom": float(config["student_t_degrees_of_freedom"]),
            "scale_floor": float(config["scale_floor"]), "validation": validation, "run_config": run_config,
        }
        if metric < best_metric:
            best_metric = metric
            stale = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale += 1
        item["epochs_without_improvement"] = stale
        write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(item), flush=True)
        if stale >= int(config["early_stopping_patience"]):
            break
    write_json(output_dir / "training_complete.json", {
        "status": "complete", "family": args.family, "best_validation_nll_per_decision": best_metric,
        "epochs_run": len(metrics), "best_checkpoint_sha256": sha256_file(output_dir / "best.pt"),
    })


if __name__ == "__main__":
    main()
