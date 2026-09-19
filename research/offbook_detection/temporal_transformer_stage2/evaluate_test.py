"""Run the one-time fixed test evaluation for the frozen stage-2 model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.cache import (  # noqa: E402
    BoardCache,
    TemporalSequenceDataset,
    collate_sequences,
)
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.delivery import (  # noqa: E402
    load_frozen_temporal_model,
    load_time_stats,
)
from research.offbook_detection.temporal_transformer_stage2.train import to_device  # noqa: E402


TASKS = {
    1: ("thinking_time_mse", 1),
    2: ("both_remaining_times_mse", 2),
    3: ("board_embedding_mse", 96),
}


def length_bin(length: int) -> str:
    if length <= 59:
        return "le_59"
    if length == 60:
        return "60"
    if length == 61:
        return "61"
    return "ge_62"


class MetricAccumulator:
    def __init__(self) -> None:
        self.squared_error: dict[str, float] = defaultdict(float)
        self.elements: dict[str, int] = defaultdict(int)
        self.sequences = 0

    def add(self, task_name: str, error_sum: float, elements: int) -> None:
        self.squared_error[task_name] += error_sum
        self.elements[task_name] += elements

    def result(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {"sequences": self.sequences}
        for name, _ in TASKS.values():
            result[name] = self.squared_error[name] / self.elements[name]
        result["selection_metric_sum_mse"] = sum(float(result[name]) for name, _ in TASKS.values())
        return result


@torch.inference_mode()
def evaluate_fixed_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    amp: bool,
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    overall = MetricAccumulator()
    slices: dict[str, MetricAccumulator] = {}
    for batch_index, host_batch in enumerate(loader):
        batch = to_device(host_batch, device)
        batch_size = int(batch["lengths"].shape[0])
        rows = torch.arange(batch_size, device=device)
        masked_node = (rows * 104729 + seed + batch_index * 1009) % batch["lengths"]
        slice_names: list[list[str]] = []
        for row in range(batch_size):
            node = int(masked_node[row])
            target_view = int(batch["target_view"][row])
            actor = "target" if bool(batch["actor_is_target"][row, node]) else "opponent"
            names = [
                f"length/{length_bin(int(batch['lengths'][row]))}",
                f"target_color/{'black' if target_view == 0 else 'white'}",
                f"masked_actor/{actor}",
            ]
            slice_names.append(names)
            for name in names:
                slices.setdefault(name, MetricAccumulator()).sequences += 1
        overall.sequences += batch_size
        for task_id, (task_name, dimensions) in TASKS.items():
            task_ids = torch.full_like(masked_node, task_id)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _, thinking, remaining, board = model.recover(batch, masked_node, task_ids)
            if task_id == 1:
                target = batch["thinking_time"][rows, masked_node]
                prediction = thinking
            elif task_id == 2:
                target = torch.stack(
                    (
                        batch["target_remaining_time"][rows, masked_node],
                        batch["opponent_remaining_time"][rows, masked_node],
                    ),
                    dim=-1,
                )
                prediction = remaining
            else:
                target = batch["board_embedding"][rows, masked_node]
                prediction = board
            per_sequence = F.mse_loss(prediction.float(), target.float(), reduction="none").reshape(batch_size, -1).sum(1)
            overall.add(task_name, float(per_sequence.sum()), batch_size * dimensions)
            for row, names in enumerate(slice_names):
                error = float(per_sequence[row])
                for name in names:
                    slices[name].add(task_name, error, dimensions)
    return overall.result(), {name: accumulator.result() for name, accumulator in sorted(slices.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-cache", type=Path, required=True)
    parser.add_argument("--time-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.seed != 42 or args.batch_size <= 0:
        raise ValueError("fixed test evaluation requires seed 42 and a positive batch size")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to repeat or overwrite the fixed test evaluation: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cache = BoardCache(args.board_cache, verify_hashes=True)
    stats, stats_payload = load_time_stats(args.time_stats)
    model, checkpoint = load_frozen_temporal_model(
        args.checkpoint, cache.directory / "manifest.json", args.time_stats, device
    )
    dataset = TemporalSequenceDataset(cache, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_sequences, stats=stats),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    amp = device.type == "cuda" and not args.no_amp
    overall, slices = evaluate_fixed_test(model, loader, device, args.seed, amp)
    report = {
        "schema": "stage2-fixed-test-evaluation-v1",
        "status": "complete",
        "split": "test",
        "test_used_for_model_selection": False,
        "checkpoint": str(args.checkpoint.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve(strict=True)),
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "board_cache_manifest": str((cache.directory / "manifest.json").resolve()),
        "board_cache_manifest_sha256": sha256_file(cache.directory / "manifest.json"),
        "time_stats": str(args.time_stats.resolve(strict=True)),
        "time_stats_sha256": sha256_file(args.time_stats.resolve(strict=True)),
        "time_stats_fit_split": stats_payload["fit_split"],
        "fixed_mask": {
            "seed": args.seed,
            "formula": "(batch_row * 104729 + seed + batch_index * 1009) % sequence_length",
            "all_three_tasks_at_same_node": True,
            "batch_size": args.batch_size,
        },
        "inference": {"device": str(device), "amp_fp16": amp},
        "slice_contract": {
            "game_length_nodes": ["le_59", "60", "61", "ge_62"],
            "target_color": ["black", "white"],
            "masked_node_actor": ["target", "opponent"],
        },
        "overall": overall,
        "slices": slices,
    }
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
