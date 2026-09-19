"""Mirror an active temporal training run into a compact progress.json."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path


SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x00000102


def process_is_running(process_id: int) -> bool:
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, process_id)
    if not handle:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def write_progress(run_dir: Path, process_id: int) -> bool:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    max_epochs = int(run_config["epochs"])
    running = process_is_running(process_id)
    early_stopping_path = run_dir / "early_stopping.json"
    if running:
        status = "running"
    elif early_stopping_path.exists() or len(metrics) == max_epochs:
        status = "complete"
    else:
        status = "process-exited-before-terminal-condition"
    latest = metrics[-1] if metrics else None
    payload = {
        "schema": "stage2-temporal-training-progress-v1",
        "status": status,
        "process_id": process_id,
        "max_epochs": max_epochs,
        "completed_epochs": len(metrics),
        "latest_epoch_zero_based": latest["epoch"] if latest else None,
        "global_step": latest["global_step"] if latest else 0,
        "latest_train_mean_batch_loss": latest["train_mean_batch_loss"] if latest else None,
        "latest_validation": latest["validation"] if latest else None,
        "epochs_without_improvement": latest.get("epochs_without_improvement") if latest else None,
        "best_validation_selection_metric_sum_mse": min(
            (item["validation"]["selection_metric_sum_mse"] for item in metrics), default=None
        ),
    }
    (run_dir / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return running


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    while write_progress(run_dir, args.process_id):
        time.sleep(args.interval_seconds)
    write_progress(run_dir, args.process_id)


if __name__ == "__main__":
    main()
