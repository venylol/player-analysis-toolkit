"""Train one family/initialization of the frozen two-state HMM experiment."""

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

from research.offbook_detection.stage3_two_state_hmm.core import (  # noqa: E402
    DiagonalHMM,
    initialize_model,
    marginal_nll,
)
from research.offbook_detection.stage3_two_state_hmm.data import load_manifest, load_split  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(
    model: DiagonalHMM,
    values: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    total_nll = 0.0
    total_nodes = 0
    for start in range(0, len(values), batch_size):
        stop = start + batch_size
        batch_values = torch.from_numpy(values[start:stop]).to(device)
        batch_lengths = torch.from_numpy(lengths[start:stop]).to(device)
        nll, nodes = marginal_nll(model, batch_values, batch_lengths)
        total_nll += float(nll)
        total_nodes += int(nodes)
    return {
        "sequences": len(values),
        "decisions": total_nodes,
        "total_nll": total_nll,
        "nll_per_decision": total_nll / total_nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initializations", type=Path, required=True)
    parser.add_argument("--initialization-index", type=int, required=True)
    parser.add_argument("--family", choices=("student_t", "gaussian"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cache_dir = args.projected_cache.resolve(strict=True)
    cache_manifest_path, cache_manifest = load_manifest(cache_dir)
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    initialization_dir = args.initializations.resolve(strict=True)
    initialization_manifest_path = initialization_dir / "manifest.json"
    initialization_manifest = json.loads(initialization_manifest_path.read_text(encoding="utf-8"))
    initialization_path = initialization_dir / "initializations.npz"
    if initialization_manifest.get("status") != "complete" or initialization_manifest["data_sha256"] != sha256_file(initialization_path):
        raise ValueError("invalid initialization artifact")
    if initialization_manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("initializations/config mismatch")
    if initialization_manifest["projected_cache_manifest_sha256"] != sha256_file(cache_manifest_path):
        raise ValueError("initializations/cache mismatch")

    with np.load(initialization_path, allow_pickle=False) as prepared:
        run_count = len(prepared["seeds"])
        if not 0 <= args.initialization_index < run_count:
            raise ValueError(f"initialization index outside [0, {run_count})")
        seed = int(prepared["seeds"][args.initialization_index])
        locations = prepared["locations"][args.initialization_index].astype(np.float32)
        scales = prepared["scales"][args.initialization_index].astype(np.float32)
        initial_probabilities = prepared["initial_probabilities"].astype(np.float32)
        transition_matrix = prepared["transition_matrix"].astype(np.float32)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("best.pt", "training_complete.json")):
        raise FileExistsError("refusing to overwrite completed HMM training output")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(seed)
    train = load_split(cache_dir, cache_manifest, config, "train")
    validation = load_split(cache_dir, cache_manifest, config, "validation")
    model = DiagonalHMM(
        int(config["projection_dim"]),
        args.family,
        float(config["student_t_degrees_of_freedom"]),
        float(config["scale_floor"]),
    ).to(device)
    initialize_model(
        model,
        torch.from_numpy(locations).to(device),
        torch.from_numpy(scales).to(device),
        torch.from_numpy(initial_probabilities).to(device),
        torch.from_numpy(transition_matrix).to(device),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    batch_size = int(config["batch_size"])
    run_config = {
        "schema": "stage3-two-state-hmm-training-v1",
        "family": args.family,
        "initialization_index": args.initialization_index,
        "seed": seed,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "projected_cache_manifest": str(cache_manifest_path),
        "projected_cache_manifest_sha256": sha256_file(cache_manifest_path),
        "initialization_manifest": str(initialization_manifest_path),
        "initialization_manifest_sha256": sha256_file(initialization_manifest_path),
        "device": str(device),
        "train_sequences": train.sequence_count,
        "train_decisions": train.decision_count,
        "validation_sequences": validation.sequence_count,
        "validation_decisions": validation.decision_count,
    }
    write_json(output_dir / "run_config.json", run_config)
    generator = np.random.default_rng(seed)
    best_metric = math.inf
    best_epoch = -1
    stale = 0
    metrics: list[dict[str, object]] = []
    for epoch in range(int(config["max_epochs"])):
        model.train()
        order = generator.permutation(train.sequence_count)
        train_nll = 0.0
        train_nodes = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            values = torch.from_numpy(train.values[indices]).to(device)
            lengths = torch.from_numpy(train.lengths[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            nll, nodes = marginal_nll(model, values, lengths)
            loss = nll / nodes
            loss.backward()
            optimizer.step()
            train_nll += float(nll.detach())
            train_nodes += int(nodes)
        validation_metrics = evaluate(model, validation.values, validation.lengths, batch_size, device)
        metric = float(validation_metrics["nll_per_decision"])
        improved = metric < best_metric - float(config["early_stopping_min_delta"])
        item = {
            "epoch": epoch,
            "train_nll_per_decision": train_nll / train_nodes,
            "validation": validation_metrics,
            "initial_probabilities": model.log_initial().exp().detach().cpu().tolist(),
            "transition_matrix": model.log_transition().exp().detach().cpu().tolist(),
            "location": model.location.detach().cpu().tolist(),
            "scale": model.scale().detach().cpu().tolist(),
        }
        if improved:
            best_metric = metric
            best_epoch = epoch
            stale = 0
            checkpoint = {
                "format": "stage3-two-state-hmm-checkpoint-v1",
                "epoch": epoch,
                "family": args.family,
                "initialization_index": args.initialization_index,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "dimension": int(config["projection_dim"]),
                "degrees_of_freedom": float(config["student_t_degrees_of_freedom"]),
                "scale_floor": float(config["scale_floor"]),
                "validation": validation_metrics,
                "run_config": run_config,
            }
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale += 1
        item["epochs_without_improvement"] = stale
        item["best_validation_nll_per_decision"] = best_metric
        metrics.append(item)
        write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(item, separators=(",", ":")), flush=True)
        if stale >= int(config["early_stopping_patience"]):
            break
    write_json(output_dir / "training_complete.json", {
        "schema": "stage3-two-state-hmm-training-complete-v1",
        "status": "complete",
        "family": args.family,
        "initialization_index": args.initialization_index,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_nll_per_decision": best_metric,
        "epochs_run": len(metrics),
        "best_checkpoint_sha256": sha256_file(output_dir / "best.pt"),
    })


if __name__ == "__main__":
    main()
