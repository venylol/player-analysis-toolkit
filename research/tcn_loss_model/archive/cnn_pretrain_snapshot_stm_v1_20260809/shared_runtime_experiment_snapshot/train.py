#!/usr/bin/env python3
"""Safe CLI. Formal optimization is impossible without the explicit `train` gate."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

import torch

from src.board_cnn_pretrain import BoardCNNTrainingConfig, benchmark_board_cnn, smoke_run, train_board_cnn
from src.checkpoint import load_transferred_model, load_transferred_profile_model, verify_checkpoint
from src.data_contract import validate_model_ready_npz, validate_raw_data, write_report
from src.inference import predict_to_csv
from src.ensemble import train_ensemble
from src.progress import initial_progress, read_progress, write_progress
from src.personal_adapter import train_adapter_ensemble
from src.training import TrainingConfig, train_cuda

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "default.json"
DEFAULT_PROFILE_CONFIG = ROOT / "config" / "profile_full.json"
DEFAULT_BOARD_CNN_CONFIG = ROOT / "config" / "board_cnn_pretrain.json"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command")
    inspect = commands.add_parser("inspect", help="validate configuration only; default safe action")
    inspect.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate = commands.add_parser("validate", help="validate raw CSV/Parquet/JSONL or model-ready NPZ")
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--context-metadata", type=Path)
    validate.add_argument("--output", type=Path)
    validate.add_argument("--require-oq-profile", action="store_true")
    validate.add_argument("--board-perspective", choices=("snapshot_side_to_move_v1", "legacy_fixed_color"))
    checkpoint = commands.add_parser("check-checkpoint", help="strictly load the official base checkpoint on CPU")
    checkpoint.add_argument("--base-checkpoint", type=Path, required=True)
    checkpoint.add_argument("--preprocessing", type=Path)
    checkpoint.add_argument("--output", type=Path)
    smoke = commands.add_parser("smoke-test", help="tiny CPU forward pass; never optimizes")
    smoke.add_argument("--base-checkpoint", type=Path, required=True)
    profile_smoke = commands.add_parser("smoke-test-profile", help="profile-branch CPU identity/shape smoke; never optimizes")
    profile_smoke.add_argument("--base-checkpoint", type=Path, required=True)
    profile_smoke.add_argument("--ablation", default="full-31")
    profile_smoke.add_argument("--device", default="cpu")
    status = commands.add_parser("status", help="read progress.json without data or GPU")
    status.add_argument("--output-dir", type=Path, required=True)
    predict = commands.add_parser("predict", help="explicit inference from a trained four-class checkpoint")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--base-checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--batch-size", type=int, default=64)
    predict_profile = commands.add_parser("predict-profile", help="explicit inference requiring OQ profile arrays/checkpoint")
    predict_profile.add_argument("--data", type=Path, required=True)
    predict_profile.add_argument("--checkpoint", type=Path, required=True)
    predict_profile.add_argument("--base-checkpoint", type=Path, required=True)
    predict_profile.add_argument("--output", type=Path, required=True)
    predict_profile.add_argument("--device", default="cpu")
    predict_profile.add_argument("--batch-size", type=int, default=64)
    train = commands.add_parser("train", help="formal CUDA training; explicit authorization flag required")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--context-metadata", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--base-checkpoint", type=Path, required=True)
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--run-name", required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--skip-test-evaluation", action="store_true",
        help="finish after validation for the seed-42 epoch probe; never opens the fixed test split",
    )
    train.add_argument("--confirm-new-data-ready", action="store_true")
    train_profile = commands.add_parser("train-profile", help="formal CUDA training with required OQ profile context")
    train_profile.add_argument("--data", type=Path, required=True)
    train_profile.add_argument("--context-metadata", type=Path)
    train_profile.add_argument("--output-dir", type=Path, required=True)
    train_profile.add_argument("--base-checkpoint", type=Path, required=True)
    train_profile.add_argument("--config", type=Path, default=DEFAULT_PROFILE_CONFIG)
    train_profile.add_argument("--run-name", required=True)
    train_profile.add_argument("--resume", type=Path)
    train_profile.add_argument("--initial-profile-checkpoint", type=Path)
    train_profile.add_argument(
        "--initial-stage-checkpoint", type=Path,
        help="strict full-model Stage A initialization for cnn-stage-b",
    )
    train_profile.add_argument(
        "--auxiliary-checkpoint", type=Path,
        help="independent LV17 checkpoint supplying audited legal/value auxiliary heads",
    )
    train_profile.add_argument("--skip-test-evaluation", action="store_true")
    train_profile.add_argument("--confirm-new-data-ready", action="store_true")
    ensemble = commands.add_parser("ensemble", help="sequential fixed-test ensemble training with resumable members")
    ensemble.add_argument("--data", type=Path, required=True)
    ensemble.add_argument("--context-metadata", type=Path)
    ensemble.add_argument("--output-dir", type=Path, required=True)
    ensemble.add_argument("--base-checkpoint", type=Path, required=True)
    ensemble.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ensemble.add_argument("--seeds", type=int, nargs="+", required=True)
    ensemble.add_argument("--confirm-new-data-ready", action="store_true")
    profile_ensemble = commands.add_parser(
        "ensemble-profile", help="sequential fixed-test ensemble training with required OQ profile context"
    )
    profile_ensemble.add_argument("--data", type=Path, required=True)
    profile_ensemble.add_argument("--context-metadata", type=Path)
    profile_ensemble.add_argument("--output-dir", type=Path, required=True)
    profile_ensemble.add_argument("--base-checkpoint", type=Path, required=True)
    profile_ensemble.add_argument("--config", type=Path, default=DEFAULT_PROFILE_CONFIG)
    profile_ensemble.add_argument("--seeds", type=int, nargs="+", required=True)
    profile_ensemble.add_argument("--fixed-split", action="store_true")
    profile_ensemble.add_argument("--warm-start-ensemble", type=Path)
    profile_ensemble.add_argument(
        "--warm-start-checkpoint-name", choices=("best.pt", "latest.pt"), default="best.pt",
        help="member checkpoint artifact used when warm-starting a new ensemble",
    )
    profile_ensemble.add_argument(
        "--extend-completed", action="store_true",
        help="resume completed members only when the new config solely extends fine-tuning epochs",
    )
    profile_ensemble.add_argument("--confirm-new-data-ready", action="store_true")
    personal = commands.add_parser("personalize-ensemble", help="frozen-base game-equal personal residual-logit adapters")
    personal.add_argument("--data", type=Path, required=True)
    personal.add_argument("--ensemble-manifest", type=Path, required=True)
    personal.add_argument("--base-checkpoint", type=Path, required=True)
    personal.add_argument("--config", type=Path, required=True)
    personal.add_argument("--output-dir", type=Path, required=True)
    board_cnn = commands.add_parser(
        "board-cnn-pretrain", help="formal independent board-CNN CUDA pretraining"
    )
    board_cnn.add_argument("--config", type=Path, default=DEFAULT_BOARD_CNN_CONFIG)
    board_cnn.add_argument("--output-dir", type=Path, required=True)
    board_cnn.add_argument("--resume", type=Path)
    board_cnn.add_argument("--confirm-full-pretraining", action="store_true")
    board_cnn_smoke = commands.add_parser(
        "board-cnn-pretrain-smoke", help="bounded CPU materialization/train/save/resume smoke"
    )
    board_cnn_smoke.add_argument("--config", type=Path, default=DEFAULT_BOARD_CNN_CONFIG)
    board_cnn_smoke.add_argument("--output-dir", type=Path, required=True)
    board_cnn_benchmark = commands.add_parser(
        "board-cnn-benchmark", help="bounded CNN compute-throughput benchmark"
    )
    board_cnn_benchmark.add_argument("--config", type=Path, default=DEFAULT_BOARD_CNN_CONFIG)
    board_cnn_benchmark.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    board_cnn_benchmark.add_argument("--batch-size", type=int, default=8)
    board_cnn_benchmark.add_argument("--steps", type=int, default=5)
    return result


def smoke_test(checkpoint_path: Path) -> dict:
    model, checkpoint = load_transferred_model(checkpoint_path)
    model.eval()
    batch, steps, dim = 2, 4, int(checkpoint["input_dim"])
    args = (
        torch.zeros(batch, steps, dim), torch.ones(batch, steps, 3, 64, dtype=torch.long),
        torch.zeros(batch, steps, 3, dtype=torch.long), torch.zeros(batch, steps, 6, dtype=torch.long),
        torch.zeros(batch, steps, 4), torch.zeros(batch, steps, 2),
    )
    with torch.no_grad():
        first = model(*args, torch.full((batch, steps), 1000.0))
        second = model(*args, torch.full((batch, steps), 9000.0))
    time_invariant = torch.equal(first.pred_time_log_seconds, second.pred_time_log_seconds)
    if not time_invariant:
        raise AssertionError("actual current time leaked into thinking-time head")
    return {
        "ok": True, "device": "cpu", "batch": batch, "steps": steps,
        "inputFeatures": dim, "timeHeadShape": list(first.pred_time_log_seconds.shape),
        "severityClassHeadShape": list(first.severity_class_probabilities.shape),
        "wldClassHeadShape": list(first.wld_probabilities.shape),
        "actualTimeDoesNotAffectTimeHead": True,
        "actualTimeAffectsOnlySeverityHead": (
            not torch.equal(first.probability_loss_ge4, second.probability_loss_ge4)
            or not torch.equal(first.probability_loss_ge10, second.probability_loss_ge10)
        ),
        "softmaxClassProbabilitySumIsOne": bool(torch.allclose(first.severity_class_probabilities.sum(dim=-1), torch.ones(batch, steps))),
        "wldSoftmaxClassProbabilitySumIsOne": bool(torch.allclose(first.wld_probabilities.sum(dim=-1), torch.ones(batch, steps))),
        "expectedWldLossInUnitInterval": bool(torch.all((first.expected_wld_loss >= 0) & (first.expected_wld_loss <= 1))),
        "probabilityGe10LeGe4": bool(torch.all(first.probability_loss_ge10 <= first.probability_loss_ge4)),
        "probabilityGe4LePositive": bool(torch.all(first.probability_loss_ge4 <= first.probability_loss_positive)),
        "optimizationSteps": 0,
    }


def profile_smoke_test(checkpoint_path: Path, ablation: str, device_name: str = "cpu") -> dict:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA profile smoke but CUDA is unavailable")
    device = torch.device(device_name)
    baseline, checkpoint = load_transferred_model(checkpoint_path)
    profile, _ = load_transferred_profile_model(checkpoint_path, ablation)
    baseline.severity_context.load_state_dict(profile.severity_context.state_dict(), strict=True)
    baseline.severity_head.load_state_dict(profile.severity_head.state_dict(), strict=True)
    baseline.wld_head.load_state_dict(profile.wld_head.state_dict(), strict=True)
    baseline.to(device).eval()
    profile.to(device).eval()
    batch, steps, dim = 2, 4, int(checkpoint["input_dim"])
    args = (
        torch.zeros(batch, steps, dim, device=device), torch.ones(batch, steps, 3, 64, dtype=torch.long, device=device),
        torch.zeros(batch, steps, 3, dtype=torch.long, device=device), torch.zeros(batch, steps, 6, dtype=torch.long, device=device),
        torch.zeros(batch, steps, 4, device=device), torch.zeros(batch, steps, 2, device=device),
        torch.full((batch, steps), 1000.0, device=device),
    )
    features = torch.randn(batch, steps, 31, device=device)
    missing = torch.rand(batch, steps, 31, device=device) > 0.5
    with torch.no_grad():
        base_output = baseline(*args)
        profile_output = profile(*args, features, missing)
    exact = (
        torch.equal(base_output.pred_time_log_seconds, profile_output.pred_time_log_seconds)
        and torch.equal(base_output.severity_logits, profile_output.severity_logits)
        and torch.equal(base_output.wld_logits, profile_output.wld_logits)
    )
    if not exact:
        raise AssertionError("zero-initialized profile branch is not exactly baseline-equivalent")
    return {
        "ok": True, "device": str(device), "optimizationSteps": 0,
        "strictOfficialBackboneLoad": True, "profileAblation": ablation,
        "profileInputShape": [batch, steps, 31], "profileMissingShape": [batch, steps, 31],
        "severityClassHeadShape": list(profile_output.severity_logits.shape),
        "wldClassHeadShape": list(profile_output.wld_logits.shape),
        "thinkingTimeHeadShape": list(profile_output.pred_time_log_seconds.shape),
        "zeroInitializedExactBaselineIdentity": True,
        "thinkingTimeHeadUnconditioned": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    command = args.command or "inspect"
    if command == "inspect":
        config_path = getattr(args, "config", DEFAULT_CONFIG)
        cfg = TrainingConfig.load(config_path)
        print(json.dumps({"ok": True, "action": "config-validation-only", "config": cfg.__dict__,
                          "formalTrainingStarted": False}, ensure_ascii=False, indent=2))
        return 0
    if command == "validate":
        if args.data.suffix.lower() == ".npz":
            report = validate_model_ready_npz(
                args.data,
                require_oq_profile=args.require_oq_profile,
                expected_board_perspective=args.board_perspective,
            )
        else:
            _nodes, report = validate_raw_data(args.data, args.context_metadata)
        if args.output:
            write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if command == "check-checkpoint":
        report = verify_checkpoint(args.base_checkpoint, args.preprocessing)
        if args.output:
            write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if command == "smoke-test":
        print(json.dumps(smoke_test(args.base_checkpoint), ensure_ascii=False, indent=2))
        return 0
    if command == "smoke-test-profile":
        print(json.dumps(profile_smoke_test(args.base_checkpoint, args.ablation, args.device), ensure_ascii=False, indent=2))
        return 0
    if command == "status":
        progress = read_progress(args.output_dir)
        view = {key: progress.get(key) for key in (
            "status", "stage", "epoch", "max_epochs", "validation_total_loss",
            "severity_classification_loss", "zero_loss_log_loss", "ge4_log_loss", "ge10_log_loss",
            "wld_classification_loss", "wld_validation_metrics",
            "best_metric", "best_epoch", "learning_rate", "elapsed_seconds", "eta_seconds", "updated_at",
            "device", "gpu_name", "cuda_version", "base_checkpoint",
        )}
        view["data_manifest"] = progress.get("data_manifest")
        print(json.dumps(view, ensure_ascii=False, indent=2))
        return 0
    if command == "predict":
        report = predict_to_csv(args.data, args.checkpoint, args.base_checkpoint, args.output, args.device, args.batch_size)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if command == "predict-profile":
        report = predict_to_csv(
            args.data, args.checkpoint, args.base_checkpoint, args.output,
            args.device, args.batch_size, use_oq_profile=True,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if command in {"train", "train-profile"}:
        if not args.confirm_new_data_ready:
            raise SystemExit("refusing formal training: user must confirm the new dataset and pass --confirm-new-data-ready")
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
            raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path, stderr_path = args.output_dir / "stdout.log", args.output_dir / "stderr.log"
        with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
            with redirect_stdout(Tee(sys.__stdout__, stdout_file)), redirect_stderr(Tee(sys.__stderr__, stderr_file)):
                try:
                    train_cuda(args.data, args.base_checkpoint, args.config, args.output_dir,
                               args.run_name, args.context_metadata, args.resume,
                               evaluate_test=not args.skip_test_evaluation,
                               use_oq_profile=command == "train-profile",
                               initial_profile_checkpoint=getattr(args, "initial_profile_checkpoint", None),
                               initial_stage_checkpoint=getattr(args, "initial_stage_checkpoint", None),
                               auxiliary_checkpoint=getattr(args, "auxiliary_checkpoint", None))
                except KeyboardInterrupt as exc:
                    traceback.print_exc()
                    try:
                        write_progress(args.output_dir, status="interrupted", stage="interrupted", error=str(exc) or "KeyboardInterrupt")
                    finally:
                        raise
                except BaseException as exc:
                    traceback.print_exc()
                    try:
                        write_progress(args.output_dir, status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}")
                    finally:
                        raise
        return 0
    if command in {"ensemble", "ensemble-profile"}:
        if not args.confirm_new_data_ready:
            raise SystemExit("refusing formal training: pass --confirm-new-data-ready")
        report = train_ensemble(
            args.data, args.base_checkpoint, args.config, args.output_dir,
            args.seeds, args.context_metadata, use_oq_profile=command == "ensemble-profile",
            extend_completed=getattr(args, "extend_completed", False),
            fixed_split=getattr(args, "fixed_split", False),
            warm_start_ensemble=getattr(args, "warm_start_ensemble", None),
            warm_start_checkpoint_name=getattr(args, "warm_start_checkpoint_name", "best.pt"),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if command == "personalize-ensemble":
        report = train_adapter_ensemble(
            args.data, args.base_checkpoint, args.ensemble_manifest,
            args.config, args.output_dir,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if command == "board-cnn-pretrain-smoke":
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty smoke output directory: {args.output_dir}")
        cfg = BoardCNNTrainingConfig.load(args.config)
        report = smoke_run(cfg, args.output_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if command == "board-cnn-benchmark":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("requested CUDA benchmark but CUDA is unavailable")
        cfg = replace(BoardCNNTrainingConfig.load(args.config), batch_size=args.batch_size)
        report = benchmark_board_cnn(cfg, torch.device(args.device), steps=args.steps)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if command == "board-cnn-pretrain":
        if not args.confirm_full_pretraining:
            raise SystemExit("refusing formal board-CNN training: pass --confirm-full-pretraining")
        if not torch.cuda.is_available():
            raise SystemExit("formal board-CNN pretraining requires CUDA")
        cfg = BoardCNNTrainingConfig.load(args.config)
        resume = args.resume
        if resume is None and cfg.resume_strategy == "latest":
            candidates = sorted((args.output_dir / "checkpoints").glob("epoch-*-step-*.pt"))
            resume = candidates[-1] if candidates else None
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and resume is None:
            raise SystemExit(f"refusing to write into non-empty output directory without resume: {args.output_dir}")
        report = train_board_cnn(cfg, args.output_dir, torch.device("cuda"), resume=resume)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(command)


if __name__ == "__main__":
    raise SystemExit(main())
