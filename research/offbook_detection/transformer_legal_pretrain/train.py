"""Train and evaluate the small spatial Transformer on legal-move prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from legal_data import PassAwareBatchSampler, collate_legal_records, load_combined_dataset
from model import SpatialLegalTransformer, SpatialTransformerConfig, parameter_count
from transforms import apply_d4, to_target_perspective, transform_flat


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_DATA_CONFIG = ROOT / "data_config.json"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    minimum_learning_rate_ratio: float
    pass_fraction: float
    num_workers: int
    seed: int
    mixed_precision: bool
    d4_audit_samples: int

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        return cls(**values)


def load_config(path: Path) -> tuple[dict[str, Any], SpatialTransformerConfig, TrainingConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "spatial-transformer-legal-pretrain-config-v1":
        raise ValueError("unsupported training configuration")
    return raw, SpatialTransformerConfig(**raw["model"]), TrainingConfig.from_dict(raw["training"])


def loader_options(num_workers: int) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": collate_legal_records,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return options


def prepare_train_batch(
    codes: torch.Tensor,
    legal: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = codes.to(device, non_blocking=True)
    legal = legal.to(device, non_blocking=True)
    actor_is_target = torch.rand(codes.shape[0], device=device) < 0.5
    target_codes = to_target_perspective(codes, actor_is_target)
    transform_ids = torch.randint(8, (codes.shape[0],), device=device)
    target_codes, legal = apply_d4(target_codes, legal, transform_ids)
    return target_codes, legal, actor_is_target


def empty_metric_totals() -> dict[str, float]:
    return {
        key: 0.0
        for key in (
            "lossSum", "tp", "fp", "fn", "tn", "exact", "samples",
            "passExact", "passSamples", "passPositiveCells", "ordinaryExact", "ordinarySamples",
        )
    }


def add_metrics(
    totals: dict[str, float],
    logits: torch.Tensor,
    targets: torch.Tensor,
    is_pass: torch.Tensor,
) -> None:
    prediction = logits >= 0
    truth = targets.bool()
    exact = (prediction == truth).all(dim=1)
    totals["lossSum"] += F.binary_cross_entropy_with_logits(logits, targets, reduction="sum").item()
    totals["tp"] += (prediction & truth).sum().item()
    totals["fp"] += (prediction & ~truth).sum().item()
    totals["fn"] += (~prediction & truth).sum().item()
    totals["tn"] += (~prediction & ~truth).sum().item()
    totals["exact"] += exact.sum().item()
    totals["samples"] += targets.shape[0]
    pass_mask = is_pass.bool()
    ordinary_mask = ~pass_mask
    totals["passExact"] += exact[pass_mask].sum().item()
    totals["passSamples"] += pass_mask.sum().item()
    totals["passPositiveCells"] += prediction[pass_mask].sum().item()
    totals["ordinaryExact"] += exact[ordinary_mask].sum().item()
    totals["ordinarySamples"] += ordinary_mask.sum().item()


def finalize_metrics(totals: dict[str, float]) -> dict[str, float]:
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    samples = totals["samples"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "legalBcePerCell": totals["lossSum"] / max(samples * 64, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "exactLegalSetAccuracy": totals["exact"] / max(samples, 1),
        "illegalCellFalsePositiveRate": fp / max(fp + tn, 1),
        "ordinaryExactLegalSetAccuracy": totals["ordinaryExact"] / max(totals["ordinarySamples"], 1),
        "passExactLegalSetAccuracy": totals["passExact"] / max(totals["passSamples"], 1),
        "passFalsePositiveCells": totals["passPositiveCells"],
        "samples": samples,
        "passSamples": totals["passSamples"],
    }


@torch.inference_mode()
def evaluate_role(
    model: SpatialLegalTransformer,
    loader: DataLoader,
    device: torch.device,
    actor_is_target_value: bool,
) -> dict[str, float]:
    model.eval()
    totals = empty_metric_totals()
    for codes, legal, is_pass in loader:
        codes = codes.to(device, non_blocking=True)
        legal = legal.to(device, non_blocking=True)
        is_pass = is_pass.to(device, non_blocking=True)
        actor = torch.full((codes.shape[0],), actor_is_target_value, device=device, dtype=torch.bool)
        target_codes = to_target_perspective(codes, actor)
        logits, _ = model(target_codes, actor)
        add_metrics(totals, logits, legal, is_pass)
    return finalize_metrics(totals)


@torch.inference_mode()
def audit_equivariance(
    model: SpatialLegalTransformer,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    model.eval()
    d4_matches = role_matches = comparisons = 0
    embedding_cosine_sum = 0.0
    seen = 0
    for codes, legal, _ in loader:
        remaining = max_samples - seen
        if remaining <= 0:
            break
        codes = codes[:remaining].to(device, non_blocking=True)
        legal = legal[:remaining].to(device, non_blocking=True)
        batch = codes.shape[0]
        actor_true = torch.ones(batch, dtype=torch.bool, device=device)
        canonical_logits, canonical_embedding = model(codes, actor_true)
        canonical_prediction = canonical_logits >= 0
        actor_false = torch.zeros(batch, dtype=torch.bool, device=device)
        swapped = to_target_perspective(codes, actor_false)
        role_logits, _ = model(swapped, actor_false)
        role_matches += ((role_logits >= 0) == canonical_prediction).all(dim=1).sum().item()
        for transform_id in range(8):
            ids = torch.full((batch,), transform_id, dtype=torch.long, device=device)
            transformed_codes, _ = apply_d4(codes, legal, ids)
            transformed_logits, transformed_embedding = model(transformed_codes, actor_true)
            expected = transform_flat(canonical_prediction, transform_id)
            d4_matches += ((transformed_logits >= 0) == expected).all(dim=1).sum().item()
            embedding_cosine_sum += F.cosine_similarity(
                transformed_embedding, canonical_embedding, dim=1
            ).sum().item()
        comparisons += batch
        seen += batch
    return {
        "samples": seen,
        "roleMaskConsistency": role_matches / max(comparisons, 1),
        "d4MaskConsistency": d4_matches / max(comparisons * 8, 1),
        "meanD4BoardEmbeddingCosine": embedding_cosine_sum / max(comparisons * 8, 1),
    }


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def ratio(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / max(warmup_steps, 1), 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum_ratio + (1 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)


def checkpoint_payload(
    model: SpatialLegalTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    raw_config: dict[str, Any],
    provenance: dict[str, Any],
    best_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "spatial-transformer-legal-checkpoint-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": raw_config,
        "provenance": provenance,
        "best_validation": best_validation,
        "torch_version": torch.__version__,
    }


def load_checkpoint(
    path: Path,
    model: SpatialLegalTransformer,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "spatial-transformer-legal-checkpoint-v1":
        raise ValueError("unsupported checkpoint format")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return payload


def provenance(config_path: Path, data_config_path: Path) -> dict[str, Any]:
    data_config = json.loads(data_config_path.read_text(encoding="utf-8"))
    base_manifest = (
        data_config_path.parent / data_config["base"]["directory"] / "manifest.json"
    ).resolve(strict=True)
    supplement = (
        data_config_path.parent / data_config["passSupplement"]["path"]
    ).resolve(strict=True)
    return {
        "config": str(config_path.resolve()),
        "configSha256": sha256_file(config_path),
        "dataConfig": str(data_config_path.resolve()),
        "dataConfigSha256": sha256_file(data_config_path),
        "baseManifest": str(base_manifest),
        "baseManifestSha256": sha256_file(base_manifest),
        "passSupplement": str(supplement),
        "passSupplementSha256": sha256_file(supplement),
        "modelSourceSha256": sha256_file(ROOT / "model.py"),
        "transformSourceSha256": sha256_file(ROOT / "transforms.py"),
        "trainingSourceSha256": sha256_file(Path(__file__)),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest = {
        "format": payload["format"],
        "checkpoint": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "epoch": payload["epoch"],
        "globalStep": payload["global_step"],
        "bestValidation": payload["best_validation"],
        "encoding": "UTF-8",
    }
    write_json(path.with_suffix(path.suffix + ".manifest.json"), manifest)
    return manifest


def train(args: argparse.Namespace) -> None:
    if not args.confirm_full_training:
        raise SystemExit("refusing formal training: pass --confirm-full-training")
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    if args.output_dir.exists() and args.resume is None:
        raise FileExistsError("output directory already exists; use a new directory or explicit --resume")
    raw_config, model_config, cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = torch.device("cuda")
    dataset = load_combined_dataset(args.data_config, "train")
    sampler = PassAwareBatchSampler(
        len(dataset.base), len(dataset.supplement), cfg.batch_size,
        cfg.pass_fraction, cfg.seed,
    )
    train_loader = DataLoader(dataset, batch_sampler=sampler, **loader_options(cfg.num_workers))
    validation_dataset = load_combined_dataset(args.data_config, "validation")
    validation_loader = DataLoader(
        validation_dataset, batch_size=cfg.batch_size, shuffle=False, **loader_options(cfg.num_workers)
    )
    model = SpatialLegalTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = make_scheduler(
        optimizer, cfg.warmup_steps, total_steps, cfg.minimum_learning_rate_ratio
    )
    use_amp = cfg.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = global_step = 0
    best: dict[str, Any] = {}
    run_provenance = provenance(args.config, args.data_config)
    if args.resume is not None:
        restored = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        if restored["config"] != raw_config or restored["provenance"] != run_provenance:
            raise ValueError("resume checkpoint configuration or data provenance differs")
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        best = dict(restored["best_validation"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    samples_processed = 0
    for epoch in range(start_epoch, cfg.epochs):
        sampler.epoch = epoch
        model.train()
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_samples = 0
        for batch_index, (codes, legal, is_pass) in enumerate(train_loader):
            target_codes, legal, actor = prepare_train_batch(codes, legal, device)
            is_pass = is_pass.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(target_codes, actor)
                loss = F.binary_cross_entropy_with_logits(logits, legal)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            samples_processed += codes.shape[0]
            epoch_loss_sum += loss.detach() * codes.shape[0]
            epoch_samples += codes.shape[0]
            if batch_index == 0 or global_step % 100 == 0:
                elapsed = max(time.time() - started, 1e-9)
                batch_totals = empty_metric_totals()
                add_metrics(batch_totals, logits.detach(), legal, is_pass)
                write_json(args.output_dir / "progress.json", {
                    "status": "training", "epoch": epoch + 1, "maxEpochs": cfg.epochs,
                    "batch": batch_index + 1, "batchesPerEpoch": len(train_loader),
                    "globalStep": global_step, "samplesProcessedThisRun": samples_processed,
                    "samplesPerSecond": samples_processed / elapsed,
                    "learningRate": optimizer.param_groups[0]["lr"],
                    "epochMeanLegalBcePerCell": epoch_loss_sum.item() / max(epoch_samples, 1),
                    "lastBatchMetrics": finalize_metrics(batch_totals),
                    "cudaPeakAllocatedBytes": torch.cuda.max_memory_allocated(),
                    "updatedAtUnix": time.time(),
                })
        train_metrics = {
            "legalBcePerCell": epoch_loss_sum.item() / max(epoch_samples, 1),
            "samples": epoch_samples,
        }
        validation_target = evaluate_role(model, validation_loader, device, True)
        validation_opponent = evaluate_role(model, validation_loader, device, False)
        audit_loader = DataLoader(
            Subset(validation_dataset, range(min(cfg.d4_audit_samples, len(validation_dataset.base)))),
            batch_size=min(cfg.batch_size, 512), shuffle=False, **loader_options(0),
        )
        equivariance = audit_equivariance(model, audit_loader, device, cfg.d4_audit_samples)
        selection = validation_target["legalBcePerCell"]
        checkpoint_path = (
            args.output_dir / "checkpoints" / f"epoch-{epoch:04d}-step-{global_step:08d}.pt"
        )
        if selection < best.get("selectionMetric", float("inf")):
            best = {
                "selectionMetric": selection, "epoch": epoch,
                "targetRole": validation_target, "opponentRole": validation_opponent,
                "equivariance": equivariance,
                "checkpoint": str(checkpoint_path.resolve()),
            }
        payload = checkpoint_payload(
            model, optimizer, scheduler, scaler, epoch, global_step,
            raw_config, run_provenance, best,
        )
        saved = save_checkpoint(checkpoint_path, payload)
        write_json(args.output_dir / "epoch_metrics" / f"epoch-{epoch:04d}.json", {
            "epoch": epoch, "train": train_metrics, "validationTargetRole": validation_target,
            "validationOpponentRole": validation_opponent, "equivariance": equivariance,
            "checkpoint": saved, "bestValidation": best,
        })
    dataset.base.close()
    validation_dataset.base.close()
    write_json(args.output_dir / "progress.json", {
        "status": "completed", "epochs": cfg.epochs, "globalStep": global_step,
        "samplesProcessedThisRun": samples_processed, "bestValidation": best,
        "elapsedSeconds": time.time() - started, "updatedAtUnix": time.time(),
    })


def benchmark(args: argparse.Namespace) -> None:
    _, model_config, _ = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    model = SpatialLegalTransformer(model_config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    codes = torch.randint(0, 3, (args.batch_size, 64), device=device)
    actor = torch.rand(args.batch_size, device=device) < 0.5
    targets = torch.zeros(args.batch_size, 64, device=device)
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits, _ = model(codes, actor)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits, _ = model(codes, actor)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "device": str(device), "batchSize": args.batch_size, "steps": args.steps,
        "samplesPerSecond": args.batch_size * args.steps / elapsed,
        "millisecondsPerStep": elapsed * 1000 / args.steps,
        "cudaPeakAllocatedBytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "parameters": parameter_count(model),
    }, ensure_ascii=False, indent=2))


def smoke(args: argparse.Namespace) -> None:
    raw_config, model_config, cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_combined_dataset(args.data_config, "train")
    indices = list(range(args.samples // 2)) + list(
        range(len(dataset.base), len(dataset.base) + args.samples - args.samples // 2)
    )
    loader = DataLoader(
        Subset(dataset, indices), batch_size=min(args.samples, 64), shuffle=False,
        **loader_options(0),
    )
    model = SpatialLegalTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    for codes, legal, _ in loader:
        target_codes, legal, actor = prepare_train_batch(codes, legal, device)
        logits, embedding = model(target_codes, actor)
        loss = F.binary_cross_entropy_with_logits(logits, legal)
        loss.backward(); optimizer.step()
        break
    output = {
        "status": "passed", "device": str(device), "loss": float(loss.item()),
        "logitShape": list(logits.shape), "embeddingShape": list(embedding.shape),
        "parameters": parameter_count(model), "config": raw_config,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    dataset.base.close()


def evaluate_command(args: argparse.Namespace) -> None:
    _, model_config, cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_combined_dataset(args.data_config, args.split)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, **loader_options(cfg.num_workers))
    model = SpatialLegalTransformer(model_config).to(device)
    payload = load_checkpoint(args.checkpoint, model)
    target = evaluate_role(model, loader, device, True)
    opponent = evaluate_role(model, loader, device, False)
    audit_loader = DataLoader(
        Subset(dataset, range(min(cfg.d4_audit_samples, len(dataset.base)))),
        batch_size=min(cfg.batch_size, 512), shuffle=False, **loader_options(0),
    )
    audit = audit_equivariance(model, audit_loader, device, cfg.d4_audit_samples)
    result = {
        "schema": "spatial-transformer-legal-evaluation-v1", "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()), "checkpointEpoch": payload["epoch"],
        "targetRole": target, "opponentRole": opponent, "equivariance": audit,
    }
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite evaluation: {args.output}")
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    dataset.base.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke_parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    smoke_parser.add_argument("--samples", type=int, default=64)
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    benchmark_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    benchmark_parser.add_argument("--batch-size", type=int, required=True)
    benchmark_parser.add_argument("--warmup-steps", type=int, default=5)
    benchmark_parser.add_argument("--steps", type=int, default=20)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--confirm-full-training", action="store_true")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate_parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate_parser.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "smoke":
        smoke(args)
    elif args.command == "benchmark":
        benchmark(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate_command(args)


if __name__ == "__main__":
    main()
