"""Train the frozen-contract stage-2 temporal Transformer."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.cache import (  # noqa: E402
    BoardCache,
    ShardBatchSampler,
    TemporalSequenceDataset,
    collate_sequences,
)
from research.offbook_detection.temporal_transformer_stage2.data import sha256_file, write_json  # noqa: E402
from research.offbook_detection.temporal_transformer_stage2.model import (  # noqa: E402
    TemporalTransformer,
    TemporalTransformerConfig,
    parameter_count,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items() if isinstance(value, torch.Tensor)}


def task_loss(
    batch: dict[str, torch.Tensor],
    masked_node: torch.Tensor,
    task_id: torch.Tensor,
    thinking_prediction: torch.Tensor,
    remaining_prediction: torch.Tensor,
    board_prediction: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    rows = torch.arange(masked_node.shape[0], device=masked_node.device)
    node_thinking = batch["thinking_time"][rows, masked_node]
    node_remaining = torch.stack(
        (
            batch["target_remaining_time"][rows, masked_node],
            batch["opponent_remaining_time"][rows, masked_node],
        ),
        dim=-1,
    )
    node_board = batch["board_embedding"][rows, masked_node]
    total = torch.zeros((), device=masked_node.device)
    metrics: dict[str, float] = {}
    for current_task, name, prediction, target in (
        (1, "thinking_time", thinking_prediction, node_thinking),
        (2, "both_remaining_times", remaining_prediction, node_remaining),
        (3, "board_embedding", board_prediction, node_board),
    ):
        selected = task_id == current_task
        if selected.any():
            loss = F.mse_loss(prediction[selected], target[selected], reduction="mean")
            total = total + loss
            metrics[name] = float(loss.detach())
    return total, metrics


@torch.inference_mode()
def evaluate(
    model: TemporalTransformer,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    amp: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    squared_error = {1: 0.0, 2: 0.0, 3: 0.0}
    elements = {1: 0, 2: 0, 3: 0}
    sequences = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device(host_batch, device)
        rows = torch.arange(batch["lengths"].shape[0], device=device)
        masked_node = (rows * 104729 + seed + batch_index * 1009) % batch["lengths"]
        for task in (1, 2, 3):
            task_id = torch.full_like(masked_node, task)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _, thinking, remaining, board = model.recover(batch, masked_node, task_id)
            if task == 1:
                target = batch["thinking_time"][rows, masked_node]
                prediction = thinking
            elif task == 2:
                target = torch.stack(
                    (batch["target_remaining_time"][rows, masked_node], batch["opponent_remaining_time"][rows, masked_node]),
                    dim=-1,
                )
                prediction = remaining
            else:
                target = batch["board_embedding"][rows, masked_node]
                prediction = board
            squared_error[task] += float(F.mse_loss(prediction.float(), target.float(), reduction="sum"))
            elements[task] += target.numel()
        sequences += int(batch["lengths"].shape[0])
    names = {1: "thinking_time_mse", 2: "both_remaining_times_mse", 3: "board_embedding_mse"}
    result = {names[task]: squared_error[task] / elements[task] for task in (1, 2, 3)}
    result["selection_metric_sum_mse"] = sum(result.values())
    result["sequences"] = float(sequences)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-cache", type=Path, required=True)
    parser.add_argument("--time-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("the frozen stage-2 seed is 42")
    if (
        args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0
        or args.gradient_accumulation <= 0 or args.early_stopping_patience <= 0
    ):
        raise ValueError("epochs, batch size, learning rate, and gradient accumulation must be positive")
    seed_everything(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists() and args.resume_checkpoint is None:
        raise FileExistsError(f"refusing to overwrite an existing training run: {output_dir}")
    if args.resume_checkpoint is not None and not config_path.exists():
        raise FileNotFoundError("resume requires the original run_config.json in output-dir")
    cache = BoardCache(args.board_cache, verify_hashes=True)
    stats_path = args.time_stats.resolve(strict=True)
    stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats_payload.get("schema") != "stage2-train-only-time-stats-v1":
        raise ValueError("unsupported time statistics")
    stats = {
        name: stats_payload[name]
        for name in ("thinking_time", "target_remaining_time", "opponent_remaining_time")
    }
    collate = partial(collate_sequences, stats=stats)
    train_dataset = TemporalSequenceDataset(cache, "train")
    validation_dataset = TemporalSequenceDataset(cache, "validation")
    if not train_dataset or not validation_dataset:
        raise ValueError("board cache must contain both train and validation sequences")
    train_batch_sampler = ShardBatchSampler(train_dataset, args.batch_size, args.seed)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    device = torch.device(args.device)
    model_config = TemporalTransformerConfig()
    model = TemporalTransformer(model_config).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run_config = {
        "schema": "stage2-temporal-training-run-v1",
        "board_cache_manifest": str((cache.directory / "manifest.json").resolve()),
        "board_cache_manifest_sha256": sha256_file(cache.directory / "manifest.json"),
        "time_stats": str(stats_path),
        "time_stats_sha256": sha256_file(stats_path),
        "model": model_config.as_dict(),
        "parameter_count": parameter_count(model),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_accumulation": args.gradient_accumulation,
        "early_stopping_patience": args.early_stopping_patience,
        "amp": amp,
        "task_probabilities": {"thinking_time": 1 / 3, "both_remaining_times": 1 / 3, "board_embedding": 1 / 3},
        "task_loss_weights": {"thinking_time": 1.0, "both_remaining_times": 1.0, "board_embedding": 1.0},
        "selection_metric": "sum of the three validation per-element MSE values",
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "validation_mask_contract": "fixed node per sequence across every selected epoch; all three tasks",
    }
    start_epoch = 0
    resume_task_generator_state: torch.Tensor | None = None
    metrics: list[dict[str, object]] = []
    best_metric = math.inf
    epochs_without_improvement = 0
    global_step = 0
    if args.resume_checkpoint is None:
        write_json(config_path, run_config)
    else:
        resume_path = args.resume_checkpoint.resolve(strict=True)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "stage2-temporal-transformer-checkpoint-v1":
            raise ValueError("unsupported resume checkpoint")
        if checkpoint["model_config"] != model_config.as_dict():
            raise ValueError("resume checkpoint model config differs from the frozen model")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
            parameter_group["weight_decay"] = args.weight_decay
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        resume_task_generator_state = checkpoint.get("task_generator_state")
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics = [item for item in metrics if int(item["epoch"]) < start_epoch]
        original_config = json.loads(config_path.read_text(encoding="utf-8"))
        original_config.setdefault("continuations", []).append(
            {
                "resume_checkpoint": str(resume_path),
                "resume_checkpoint_sha256": sha256_file(resume_path),
                "start_epoch": start_epoch,
                "new_max_epochs": args.epochs,
                "fixed_validation_mask_seed": args.seed,
                "early_stopping_patience": args.early_stopping_patience,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
        original_config["epochs"] = args.epochs
        original_config["validation_mask_contract"] = run_config["validation_mask_contract"]
        write_json(config_path, original_config)
        baseline = evaluate(
            model, validation_loader, device, args.seed, amp, args.max_validation_batches
        )
        best_metric = baseline["selection_metric_sum_mse"]
        write_json(
            output_dir / "fixed_validation_resume_baseline.json",
            {
                "checkpoint": str(resume_path),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "fixed_mask_seed": args.seed,
                "validation": baseline,
            },
        )

    task_generator = torch.Generator(device=device).manual_seed(args.seed + start_epoch)
    if resume_task_generator_state is not None:
        task_generator.set_state(resume_task_generator_state)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        loss_sum = 0.0
        train_batches = 0
        for batch_index, host_batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            batch = to_device(host_batch, device)
            task_id = torch.randint(1, 4, (batch["lengths"].shape[0],), generator=task_generator, device=device)
            masked_node = (torch.rand(batch["lengths"].shape[0], generator=task_generator, device=device) * batch["lengths"]).long()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _, thinking, remaining, board = model.recover(batch, masked_node, task_id)
                loss, _ = task_loss(batch, masked_node, task_id, thinking, remaining, board)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated += 1
            loss_sum += float(loss.detach())
            train_batches += 1
            if accumulated == args.gradient_accumulation:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                global_step += 1
        if accumulated:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        validation = evaluate(
            model, validation_loader, device, args.seed, amp, args.max_validation_batches
        )
        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_mean_batch_loss": loss_sum / train_batches,
            "train_batches": train_batches,
            "validation": validation,
        }
        if device.type == "cuda":
            epoch_metrics["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            epoch_metrics["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        metrics.append(epoch_metrics)
        write_json(output_dir / "metrics.json", metrics)
        checkpoint = {
            "format": "stage2-temporal-transformer-checkpoint-v1",
            "epoch": epoch,
            "global_step": global_step,
            "model_config": model_config.as_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "task_generator_state": task_generator.get_state(),
            "run_config": run_config,
            "validation": validation,
        }
        torch.save(checkpoint, output_dir / f"epoch-{epoch:03d}.pt")
        if validation["selection_metric_sum_mse"] < best_metric:
            best_metric = validation["selection_metric_sum_mse"]
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        epoch_metrics["epochs_without_improvement"] = epochs_without_improvement
        write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False), flush=True)
        if epochs_without_improvement >= args.early_stopping_patience:
            write_json(
                output_dir / "early_stopping.json",
                {
                    "stopped_after_epoch": epoch,
                    "patience": args.early_stopping_patience,
                    "best_selection_metric_sum_mse": best_metric,
                },
            )
            break


if __name__ == "__main__":
    main()
