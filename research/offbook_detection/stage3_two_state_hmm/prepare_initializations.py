"""Prepare five shared train-only emission initializations for both HMM families."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.stage3_two_state_hmm.data import (  # noqa: E402
    flattened_valid_values,
    load_manifest,
    load_split,
)
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402


def save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cache_dir = args.projected_cache.resolve(strict=True)
    manifest_path, manifest = load_manifest(cache_dir)
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "stage3-two-state-hmm-config-v1" or config.get("status") != "frozen":
        raise ValueError("frozen HMM config required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "initializations.npz"
    metadata_path = output_dir / "manifest.json"
    if data_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite HMM initializations")
    train = load_split(cache_dir, manifest, config, "train")
    values = flattened_valid_values(train)
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    inertias: list[float] = []
    seeds = [int(value) for value in config["initialization_seeds"]]
    for index, seed in enumerate(seeds):
        estimator = MiniBatchKMeans(
            n_clusters=2,
            init="k-means++",
            n_init=1,
            batch_size=int(config["kmeans_batch_size"]),
            max_iter=int(config["kmeans_max_iter"]),
            random_state=seed,
            reassignment_ratio=0.0,
        )
        labels = estimator.fit_predict(values)
        if any(not np.any(labels == state) for state in range(2)):
            raise RuntimeError(f"empty KMeans cluster for seed {seed}")
        run_scales = np.stack([
            values[labels == state].std(axis=0).clip(min=1e-3)
            for state in range(2)
        ]).astype(np.float32)
        locations.append(estimator.cluster_centers_.astype(np.float32))
        scales.append(run_scales)
        inertias.append(float(estimator.inertia_))
        print(json.dumps({"initialization": index, "seed": seed, "inertia": inertias[-1]}), flush=True)
    save_npz_atomic(data_path, {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "locations": np.stack(locations),
        "scales": np.stack(scales),
        "initial_probabilities": np.asarray(config["initial_state_probabilities"], dtype=np.float32),
        "transition_matrix": np.asarray(config["initial_transition_matrix"], dtype=np.float32),
    })
    payload = {
        "schema": "stage3-two-state-hmm-initializations-v1",
        "status": "complete",
        "fit_split": "train",
        "train_sequences": train.sequence_count,
        "train_decisions": train.decision_count,
        "seeds": seeds,
        "inertias": inertias,
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "projected_cache_manifest": str(manifest_path),
        "projected_cache_manifest_sha256": sha256_file(manifest_path),
        "pca_sha256": manifest["pca_sha256"],
    }
    write_json(metadata_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
