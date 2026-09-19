"""Build the sharded FP32 canonical board cache for stage 2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.offbook_detection.temporal_transformer_stage2.data import (  # noqa: E402
    GameNodes,
    canonical_target_boards,
    iter_games,
    sha256_file,
    write_json,
)
from research.offbook_detection.transformer_legal_pretrain.model import (  # noqa: E402
    SpatialLegalTransformer,
    SpatialTransformerConfig,
)


SCHEMA = "stage2-canonical-board-cache-v1"


def load_spatial_encoder(path: Path, device: torch.device) -> tuple[SpatialLegalTransformer, dict[str, object]]:
    payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if payload.get("format") != "spatial-transformer-board-encoder-v1":
        raise ValueError("unsupported spatial encoder export")
    model = SpatialLegalTransformer(SpatialTransformerConfig(**payload["model_config"]))
    incompatible = model.load_state_dict(payload["encoder_state_dict"], strict=False)
    if set(incompatible.missing_keys) != {"legal_head.weight", "legal_head.bias"} or incompatible.unexpected_keys:
        raise ValueError(f"spatial encoder state mismatch: {incompatible}")
    model.legal_head = torch.nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def encode_boards(
    model: SpatialLegalTransformer,
    codes: torch.Tensor,
    actor_is_target: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, codes.shape[0], batch_size):
            batch_codes = codes[start:start + batch_size].to(device, non_blocking=True)
            batch_actor = actor_is_target[start:start + batch_size].to(device, non_blocking=True)
            _, embedding = model(batch_codes, batch_actor)
            outputs.append(embedding.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def shard_payload(
    games: list[GameNodes],
    model: SpatialLegalTransformer,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    board_blocks: list[torch.Tensor] = []
    actor_blocks: list[torch.Tensor] = []
    offsets: list[int] = []
    total_nodes = 0
    for game in games:
        offsets.append(total_nodes)
        board_blocks.append(canonical_target_boards(game.board_before, game.transform_id))
        black_actor = torch.from_numpy(game.actor_is_black.copy())
        actor_blocks.append(torch.stack((black_actor, ~black_actor)))
        total_nodes += game.node_count
    boards_by_game = torch.cat(board_blocks, dim=1)  # 2 x total_nodes x 64
    actors_by_game = torch.cat(actor_blocks, dim=1)
    flat_embeddings = encode_boards(
        model,
        boards_by_game.reshape(-1, 64),
        actors_by_game.reshape(-1),
        batch_size,
        device,
    )
    embeddings = flat_embeddings.reshape(2, total_nodes, -1).astype(np.float32, copy=False)
    return {
        "embeddings": embeddings,
        "game_id": np.asarray([game.game_id for game in games]),
        "split": np.asarray([game.split for game in games]),
        "black_id": np.asarray([game.black_id for game in games]),
        "white_id": np.asarray([game.white_id for game in games]),
        "transform_id": np.asarray([game.transform_id for game in games], dtype=np.uint8),
        "game_node_offset": np.asarray(offsets, dtype=np.int32),
        "game_node_count": np.asarray([game.node_count for game in games], dtype=np.int16),
        "node_index": np.concatenate([game.node_index for game in games]),
        "strict_ply": np.concatenate([game.strict_ply for game in games]),
        "actor_is_black": np.concatenate([game.actor_is_black for game in games]),
        "is_pass": np.concatenate([game.is_pass for game in games]),
        "thinking_time_ms": np.concatenate([game.thinking_time_ms for game in games]),
        "black_remaining_time_ms_after": np.concatenate(
            [game.black_remaining_time_ms_after for game in games]
        ),
        "white_remaining_time_ms_after": np.concatenate(
            [game.white_remaining_time_ms_after for game in games]
        ),
        "time_control_id": np.asarray([game.time_control_id for game in games]),
    }


def save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)


def validate_dataset_hashes(data_dir: Path) -> dict[str, str]:
    source_manifest_path = data_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema") != "oq-small-transformer-change-point-dataset-v1":
        raise ValueError("unsupported source dataset manifest")
    hashes = {"manifest.json": sha256_file(source_manifest_path)}
    for name in ("games.csv", "nodes.csv", "split_manifest.csv"):
        actual = sha256_file(data_dir / name)
        expected = source_manifest["files"][name]["sha256"]
        if actual != expected:
            raise ValueError(f"source hash mismatch for {name}: expected {expected}, got {actual}")
        hashes[name] = actual
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games-per-shard", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-games", type=int)
    args = parser.parse_args()
    if args.games_per_shard <= 0 or args.batch_size <= 0:
        raise ValueError("games-per-shard and batch-size must be positive")
    if args.max_games is not None and args.max_games <= 0:
        raise ValueError("max-games must be positive")

    data_dir = args.data_dir.resolve(strict=True)
    encoder_path = args.encoder.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    if manifest_path.exists():
        raise FileExistsError(f"completed cache already exists: {manifest_path}")

    source_hashes = validate_dataset_hashes(data_dir)
    encoder_sha256 = sha256_file(encoder_path)
    contract = {
        "schema": SCHEMA,
        "source_hashes": source_hashes,
        "encoder_sha256": encoder_sha256,
        "orientation": "one whole-game D4 transform mapping first black move to f5",
        "target_views": ["black", "white"],
        "occupancy_codes": {"empty": 0, "target_player": 1, "opponent": 2},
        "dtype": "float32",
        "embedding_dim": 96,
        "games_per_shard": args.games_per_shard,
        "max_games": args.max_games,
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract") != contract:
            raise ValueError("existing partial cache contract does not match this run")
    else:
        progress = {"contract": contract, "shards": []}
        write_json(progress_path, progress)

    completed = {int(item["index"]): item for item in progress["shards"]}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, encoder_payload = load_spatial_encoder(encoder_path, device)
    if int(encoder_payload["board_embedding_dim"]) != 96:
        raise ValueError("frozen contract requires a 96-dimensional board embedding")

    pending: list[GameNodes] = []
    total_games = 0
    total_nodes = 0
    shard_index = 0

    def flush(games: list[GameNodes], index: int) -> None:
        nonlocal total_nodes
        path = output_dir / f"shard-{index:05d}.npz"
        node_count = sum(game.node_count for game in games)
        prior = completed.get(index)
        if prior is not None:
            if not path.exists() or sha256_file(path) != prior["sha256"]:
                raise ValueError(f"completed shard {index} is missing or has changed")
            if int(prior["games"]) != len(games) or int(prior["nodes"]) != node_count:
                raise ValueError(f"completed shard {index} no longer matches source grouping")
        else:
            if path.exists():
                raise FileExistsError(f"untracked shard already exists: {path}")
            save_npz_atomic(path, shard_payload(games, model, args.batch_size, device))
            item = {
                "index": index,
                "file": path.name,
                "games": len(games),
                "nodes": node_count,
                "target_view_nodes": node_count * 2,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "first_game_id": games[0].game_id,
                "last_game_id": games[-1].game_id,
            }
            progress["shards"].append(item)
            write_json(progress_path, progress)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        total_nodes += node_count

    for game in iter_games(data_dir / "nodes.csv"):
        if args.max_games is not None and total_games >= args.max_games:
            break
        pending.append(game)
        total_games += 1
        if len(pending) == args.games_per_shard:
            flush(pending, shard_index)
            pending = []
            shard_index += 1
    if pending:
        flush(pending, shard_index)

    source_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.max_games is None:
        if total_games != int(source_manifest["games"]):
            raise ValueError(f"game count mismatch: expected {source_manifest['games']}, got {total_games}")
        expected_nodes = sum(int(cohort["nodes"]) for cohort in source_manifest["cohorts"].values())
        if total_nodes != expected_nodes:
            raise ValueError(f"node count mismatch: expected {expected_nodes}, got {total_nodes}")
    manifest = {
        **contract,
        "status": "complete" if args.max_games is None else "smoke-test-subset",
        "games": total_games,
        "nodes": total_nodes,
        "target_view_sequences": total_games * 2,
        "target_view_nodes": total_nodes * 2,
        "shards": progress["shards"],
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "games": total_games, "nodes": total_nodes}, ensure_ascii=False))


if __name__ == "__main__":
    main()

