#!/usr/bin/env python3
"""Fit a first monotonic off-book evidence curve from five auditable inputs.

The model is a discrete-time first-event logistic hazard.  Its per-node hazard is
converted to a non-negative cumulative-hazard increment ``h``; ``d = cumsum(h)``
is therefore monotonic by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
MODEL_FEATURES = (
    "strict_ply",
    "log_scaled_thinking_time_ratio",
    "log_disc_loss",
    "parent_child_ratio",
    "global_rarity",
)
TIME_RATIO_LOG_SCALE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--frequency-book", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--no-offbook-quantile", type=float, default=0.95)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def transform_board(board64: str, mode: int) -> str:
    if len(board64) != 64:
        raise ValueError(f"board must contain 64 characters, got {len(board64)}")
    result = ["-"] * 64
    for row in range(8):
        for col in range(8):
            if mode == 0:
                dst_row, dst_col = row, col
            elif mode == 1:
                dst_row, dst_col = 7 - row, col
            elif mode == 2:
                dst_row, dst_col = row, 7 - col
            elif mode == 3:
                dst_row, dst_col = 7 - row, 7 - col
            elif mode == 4:
                dst_row, dst_col = col, row
            elif mode == 5:
                dst_row, dst_col = 7 - col, 7 - row
            elif mode == 6:
                dst_row, dst_col = col, 7 - row
            elif mode == 7:
                dst_row, dst_col = 7 - col, row
            else:
                raise ValueError(f"invalid symmetry mode: {mode}")
            result[dst_row * 8 + dst_col] = board64[row * 8 + col]
    return "".join(result)


def canonical_current_view(tokens: np.ndarray, side_to_move: str) -> str:
    """Convert fixed-color board tokens to the global book's current-side view."""
    flat = np.asarray(tokens).reshape(-1)
    if flat.shape != (64,):
        raise ValueError(f"expected 64 board tokens, got {flat.shape}")
    side = str(side_to_move).casefold()
    if side not in {"black", "white"}:
        raise ValueError(f"invalid side_to_move: {side_to_move!r}")
    chars: list[str] = []
    for token in flat:
        value = int(token)
        if value == 1:
            chars.append("-")
        elif value == 2:
            chars.append("X" if side == "black" else "O")
        elif value == 3:
            chars.append("O" if side == "black" else "X")
        else:
            raise ValueError(f"unexpected non-padding board token: {value}")
    board64 = "".join(chars)
    return min(transform_board(board64, mode) for mode in range(8))


def occupied_count(tokens: np.ndarray) -> int:
    flat = np.asarray(tokens).reshape(-1)
    return int(np.count_nonzero((flat == 2) | (flat == 3)))


def strict_move_ply(parent_tokens: np.ndarray) -> int:
    """Actual placement number derived from the pre-move board state."""
    return occupied_count(parent_tokens) - 3


def strict_book_node_ply(board64: str) -> int:
    """Placement number represented by a post-move global-book node."""
    return sum(char != "-" for char in board64) - 4


def step_time_ratio(thinking_time_ms: float, initial_time_limit_ms: float) -> float:
    if not math.isfinite(thinking_time_ms) or thinking_time_ms < 0:
        raise ValueError(f"invalid thinking time: {thinking_time_ms}")
    if not math.isfinite(initial_time_limit_ms) or initial_time_limit_ms <= 0:
        raise ValueError(f"invalid initial time limit: {initial_time_limit_ms}")
    ratio = thinking_time_ms / initial_time_limit_ms
    if not 0 <= ratio <= 1:
        raise ValueError(
            f"step thinking-time ratio must lie in [0, 1], got {ratio} "
            f"from {thinking_time_ms}/{initial_time_limit_ms}"
        )
    return ratio


def log_scaled_time_ratio(time_ratio: float) -> float:
    """Compress a step-time ratio relative to an interpretable 1% clock scale."""
    if not math.isfinite(time_ratio) or not 0 <= time_ratio <= 1:
        raise ValueError(f"time ratio must lie in [0, 1], got {time_ratio}")
    return math.log1p(time_ratio / TIME_RATIO_LOG_SCALE)


@dataclass(frozen=True)
class BookFeatures:
    parent_frequency: float
    child_frequency: float
    child_frequency_imputed: bool
    same_ply_frequency_sum: float
    parent_child_ratio: float
    global_ratio: float
    global_rarity: float
    status: str


@dataclass
class GlobalFrequencyBook:
    lookup: dict[tuple[int, str], int]
    frequency_sum_by_ply: dict[int, int]
    max_ply: int
    min_count: int
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "GlobalFrequencyBook":
        payload = read_json(path)
        if payload.get("schema") != "egaroucid.human_frequency_book.v1":
            raise ValueError(f"unexpected frequency book schema: {payload.get('schema')!r}")
        lookup: dict[tuple[int, str], int] = {}
        totals: dict[int, int] = {}
        for node in payload.get("nodes", []):
            if not isinstance(node, list) or len(node) < 3:
                raise ValueError("frequency book contains an invalid node")
            board, frequency, stored_ply = str(node[0]), int(node[1]), int(node[2])
            derived_ply = strict_book_node_ply(board)
            if derived_ply != stored_ply:
                raise ValueError(
                    f"book ply disagrees with board occupancy: stored={stored_ply}, derived={derived_ply}"
                )
            key = (derived_ply, board)
            if key in lookup:
                raise ValueError(f"duplicate frequency-book node at ply {derived_ply}")
            if frequency <= 0:
                raise ValueError(f"non-positive frequency at ply {derived_ply}: {frequency}")
            lookup[key] = frequency
            totals[derived_ply] = totals.get(derived_ply, 0) + frequency
        max_ply = int(payload.get("max_ply_inclusive"))
        if max(totals, default=0) > max_ply:
            raise ValueError("frequency book contains nodes beyond max_ply_inclusive")
        metadata = {key: value for key, value in payload.items() if key != "nodes"}
        return cls(
            lookup=lookup,
            frequency_sum_by_ply=totals,
            max_ply=max_ply,
            min_count=int(payload.get("min_count", 0)),
            metadata=metadata,
        )

    def features(self, parent_key: str | None, child_key: str | None, move_ply: int) -> BookFeatures:
        nan = float("nan")
        if move_ply > self.max_ply:
            return BookFeatures(nan, nan, False, nan, nan, nan, nan, "beyond_book_max_ply")
        if parent_key is None or child_key is None:
            return BookFeatures(nan, nan, False, nan, nan, nan, nan, "board_unavailable")

        parent = self.lookup.get((move_ply - 1, parent_key))
        child = self.lookup.get((move_ply, child_key))
        child_imputed = child is None
        child_for_model = 4 if child_imputed else int(child)

        parent_child_ratio = nan
        if parent is not None and parent > 0:
            parent_child_ratio = min(1.0, child_for_model / parent)

        raw_total = self.frequency_sum_by_ply.get(move_ply)
        if raw_total is None or raw_total <= 0:
            return BookFeatures(
                float(parent) if parent is not None else nan,
                float(child_for_model),
                child_imputed,
                nan,
                parent_child_ratio,
                nan,
                nan,
                "same_ply_total_unavailable",
            )
        adjusted_total = raw_total + (4 if child_imputed else 0)
        global_ratio = child_for_model / adjusted_total
        if not 0 < global_ratio <= 1:
            raise ValueError(f"invalid global ratio at ply {move_ply}: {global_ratio}")
        return BookFeatures(
            float(parent) if parent is not None else nan,
            float(child_for_model),
            child_imputed,
            float(adjusted_total),
            parent_child_ratio,
            global_ratio,
            -math.log(global_ratio),
            "ok_child_imputed_4" if child_imputed else "ok_exact_child_frequency",
        )


def load_marks(path: Path, expected_account: str) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if payload.get("schema") != "player-offbook-agent-marks-input-v1":
        raise ValueError(f"unexpected marks schema: {path}")
    if str(payload.get("account") or "").casefold() != expected_account.casefold():
        raise ValueError(f"marks account mismatch for {expected_account!r}: {path}")
    result: dict[str, dict[str, Any]] = {}
    for item in payload.get("marks", []):
        game_id = str(item.get("gameId") or "")
        judgment = str(item.get("judgment") or "")
        offbook_ply = item.get("offBookPly")
        if not game_id or game_id in result:
            raise ValueError(f"empty or duplicate game ID in {path}: {game_id!r}")
        if judgment not in {"offbook", "no_offbook"}:
            raise ValueError(f"invalid judgment for {game_id}: {judgment!r}")
        if judgment == "offbook" and not isinstance(offbook_ply, int):
            raise ValueError(f"offbook game {game_id} lacks an integer anchor ply")
        if judgment == "no_offbook" and offbook_ply is not None:
            raise ValueError(f"no_offbook game {game_id} has an anchor ply")
        result[game_id] = {"judgment": judgment, "offbook_ply": offbook_ply}
    if not result:
        raise ValueError(f"empty marks file: {path}")
    return result


def validate_npz_arrays(data: np.lib.npyio.NpzFile, path: Path) -> None:
    node_arrays = {
        "actual_thinking_time_ms",
        "raw_thinking_time_ms",
        "effective_thinking_time_ms",
        "disc_loss",
        "label_available",
        "player_id",
        "side_to_move",
        "move_index",
        "global_placement_ply",
        "board_tokens",
    }
    game_arrays = {"game_id", "source_time_limit_ms", "effective_time_limit_ms"}
    required = node_arrays | game_arrays
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{path} lacks arrays: {missing}")
    shape = data["global_placement_ply"].shape
    if data["board_tokens"].shape != (*shape, 3, 64):
        raise ValueError(f"unexpected board_tokens shape in {path}: {data['board_tokens'].shape}")
    for name in node_arrays - {"board_tokens"}:
        if data[name].shape != shape:
            raise ValueError(f"array {name} has shape {data[name].shape}, expected {shape}")
    for name in game_arrays:
        if data[name].shape != (shape[0],):
            raise ValueError(f"{name} has shape {data[name].shape}, expected {(shape[0],)}")


def materialize_cohort_nodes(
    account: str,
    data_path: Path,
    marks_path: Path,
    book: GlobalFrequencyBook,
) -> pd.DataFrame:
    marks = load_marks(marks_path, account)
    rows: list[dict[str, Any]] = []
    with np.load(data_path, allow_pickle=False) as data:
        validate_npz_arrays(data, data_path)
        game_ids = data["game_id"].astype(str)
        if set(game_ids) != set(marks):
            raise ValueError(
                f"marked games and NPZ games differ for {account}: "
                f"marks-only={sorted(set(marks) - set(game_ids))}, "
                f"data-only={sorted(set(game_ids) - set(marks))}"
            )
        for game_index, game_id in enumerate(game_ids):
            source_time_limit_ms = float(data["source_time_limit_ms"][game_index])
            effective_time_limit_ms = float(data["effective_time_limit_ms"][game_index])
            if not math.isfinite(source_time_limit_ms) or source_time_limit_ms <= 0:
                raise ValueError(f"invalid source initial time limit for {game_id}: {source_time_limit_ms}")
            if not math.isfinite(effective_time_limit_ms) or effective_time_limit_ms <= 0:
                raise ValueError(
                    f"invalid effective initial time limit for {game_id}: {effective_time_limit_ms}"
                )
            valid = np.flatnonzero(data["global_placement_ply"][game_index] > 0)
            if len(valid) == 0:
                raise ValueError(f"game {game_id} contains no actual placements")
            expected = np.arange(1, len(valid) + 1)
            actual = data["global_placement_ply"][game_index, valid].astype(int)
            if not np.array_equal(actual, expected):
                raise ValueError(f"game {game_id} has non-consecutive placement ply values")

            target_decision = 0
            anchor_count = 0
            for position, step in enumerate(valid):
                parent_tokens = data["board_tokens"][game_index, step, 0]
                move_ply = strict_move_ply(parent_tokens)
                stored_ply = int(data["global_placement_ply"][game_index, step])
                if move_ply != stored_ply:
                    raise ValueError(
                        f"strict board ply mismatch for {game_id} step {step}: {move_ply} != {stored_ply}"
                    )
                side = str(data["side_to_move"][game_index, step])
                parent_key = canonical_current_view(parent_tokens, side)

                child_key: str | None = None
                if position + 1 < len(valid):
                    next_step = valid[position + 1]
                    next_tokens = data["board_tokens"][game_index, next_step, 0]
                    if strict_move_ply(next_tokens) != move_ply + 1:
                        raise ValueError(f"child board ply mismatch for {game_id} at ply {move_ply}")
                    child_key = canonical_current_view(
                        next_tokens,
                        str(data["side_to_move"][game_index, next_step]),
                    )
                book_features = book.features(parent_key, child_key, move_ply)

                player_id = str(data["player_id"][game_index, step])
                if player_id.casefold() != account.casefold():
                    continue
                target_decision += 1
                mark = marks[game_id]
                is_anchor = mark["judgment"] == "offbook" and move_ply == mark["offbook_ply"]
                anchor_count += int(is_anchor)
                loss_available = bool(data["label_available"][game_index, step])
                disc_loss = (
                    float(data["disc_loss"][game_index, step]) if loss_available else float("nan")
                )
                raw_thinking_time_ms = float(data["raw_thinking_time_ms"][game_index, step])
                effective_thinking_time_ms = float(
                    data["effective_thinking_time_ms"][game_index, step]
                )
                actual_thinking_time_ms = float(data["actual_thinking_time_ms"][game_index, step])
                time_ratio = step_time_ratio(raw_thinking_time_ms, source_time_limit_ms)
                effective_ratio = step_time_ratio(
                    effective_thinking_time_ms,
                    effective_time_limit_ms,
                )
                if abs(time_ratio - effective_ratio) > 1e-5:
                    raise ValueError(
                        f"raw/effective time-ratio mismatch for {game_id} at ply {move_ply}: "
                        f"{time_ratio} != {effective_ratio}"
                    )
                if abs(actual_thinking_time_ms - effective_thinking_time_ms) > 1e-9:
                    raise ValueError(
                        f"actual/effective thinking-time mismatch for {game_id} at ply {move_ply}"
                    )
                rows.append({
                    "account": account,
                    "game_id": game_id,
                    "judgment": mark["judgment"],
                    "manual_offbook_ply": mark["offbook_ply"],
                    "is_manual_anchor": is_anchor,
                    "at_risk": mark["judgment"] == "no_offbook" or move_ply <= int(mark["offbook_ply"]),
                    "target_decision_number": target_decision,
                    "move_index": int(data["move_index"][game_index, step]),
                    "strict_ply": move_ply,
                    "raw_thinking_time_ms": raw_thinking_time_ms,
                    "source_initial_time_limit_ms": source_time_limit_ms,
                    "effective_thinking_time_ms": effective_thinking_time_ms,
                    "effective_initial_time_limit_ms": effective_time_limit_ms,
                    "thinking_time_ratio": time_ratio,
                    "log_scaled_thinking_time_ratio": log_scaled_time_ratio(time_ratio),
                    "disc_loss": disc_loss,
                    "loss_available": loss_available,
                    "parent_frequency": book_features.parent_frequency,
                    "child_frequency": book_features.child_frequency,
                    "child_frequency_imputed": book_features.child_frequency_imputed,
                    "same_ply_frequency_sum": book_features.same_ply_frequency_sum,
                    "parent_child_ratio": book_features.parent_child_ratio,
                    "global_ratio": book_features.global_ratio,
                    "global_rarity": book_features.global_rarity,
                    "book_status": book_features.status,
                    "log_disc_loss": math.log1p(disc_loss) if math.isfinite(disc_loss) else float("nan"),
                })
            expected_anchors = int(marks[game_id]["judgment"] == "offbook")
            if anchor_count != expected_anchors:
                raise ValueError(
                    f"manual anchor does not map exactly once for {account}/{game_id}: {anchor_count}"
                )
    return pd.DataFrame(rows)


def load_all_nodes(
    cohort_manifest_path: Path,
    book: GlobalFrequencyBook,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    manifest = read_json(cohort_manifest_path)
    if manifest.get("schema") != "manual-offbook-time-baseline-cohort-v1":
        raise ValueError("unexpected cohort manifest schema")
    frames: list[pd.DataFrame] = []
    marked_game_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for cohort in manifest.get("cohorts", []):
        account = str(cohort.get("account") or "").strip()
        data_path = resolve_repository_path(cohort["data"])
        marks_path = resolve_repository_path(cohort["marks"])
        marks = load_marks(marks_path, account)
        frame = materialize_cohort_nodes(account, data_path, marks_path, book)
        frames.append(frame)
        marked_game_rows.extend({
            "account": account,
            "game_id": game_id,
            "judgment": item["judgment"],
            "manual_offbook_ply": item["offbook_ply"],
        } for game_id, item in marks.items())
        inputs.append({
            "account": account,
            "data": str(data_path),
            "dataSha256": sha256_file(data_path),
            "marks": str(marks_path),
            "marksSha256": sha256_file(marks_path),
            "games": int(len(marks)),
            "targetNodes": int(len(frame)),
        })
    if not frames:
        raise ValueError("cohort manifest contains no cohorts")
    nodes = pd.concat(frames, ignore_index=True)
    nodes["cumulative_time_ratio"] = nodes.groupby(
        ["account", "game_id"], sort=False
    )["thinking_time_ratio"].cumsum()
    if nodes.duplicated(["account", "game_id", "target_decision_number"]).any():
        raise ValueError("materialized nodes contain duplicate target decisions")
    marked_games = pd.DataFrame(marked_game_rows)
    if marked_games.duplicated(["account", "game_id"]).any():
        raise ValueError("manual marks contain duplicate player/game evaluations")
    return nodes, marked_games, inputs


@dataclass(frozen=True)
class Preprocessor:
    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Preprocessor":
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for feature in MODEL_FEATURES:
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                raise ValueError(f"training data has no finite values for {feature}")
            mean = float(np.mean(finite))
            std = float(np.std(finite))
            means[feature] = mean
            stds[feature] = std if math.isfinite(std) and std > 1e-12 else 1.0
        return cls(means, stds)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for feature in MODEL_FEATURES:
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, self.means[feature])
            columns.append((values - self.means[feature]) / self.stds[feature])
        return np.column_stack(columns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "features": list(MODEL_FEATURES),
            "means": self.means,
            "stds": self.stds,
            "missingPolicy": "training finite mean; standardized contribution is zero; no missing indicator",
        }


@dataclass
class FittedHazardModel:
    model: LogisticRegression
    preprocessing: Preprocessor

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.preprocessing.transform(frame))[:, 1]

    def as_dict(self) -> dict[str, Any]:
        standardized = {
            feature: float(value)
            for feature, value in zip(MODEL_FEATURES, self.model.coef_[0], strict=True)
        }
        raw_scale = {
            feature: float(standardized[feature] / self.preprocessing.stds[feature])
            for feature in MODEL_FEATURES
        }
        raw_intercept = float(self.model.intercept_[0]) - sum(
            raw_scale[feature] * self.preprocessing.means[feature] for feature in MODEL_FEATURES
        )
        return {
            "model": "L2-regularized discrete-time logistic first-event hazard",
            "interceptStandardized": float(self.model.intercept_[0]),
            "standardizedCoefficients": standardized,
            "rawScaleCoefficients": raw_scale,
            "rawScaleIntercept": raw_intercept,
            "preprocessing": self.preprocessing.as_dict(),
        }


def fit_hazard_model(frame: pd.DataFrame, regularization_c: float) -> FittedHazardModel:
    if regularization_c <= 0:
        raise ValueError("regularization C must be positive")
    risk = frame.loc[frame["at_risk"]].copy()
    y = risk["is_manual_anchor"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("hazard training data must contain both event and non-event rows")
    preprocessing = Preprocessor.fit(risk)
    model = LogisticRegression(
        l1_ratio=0.0,
        C=regularization_c,
        solver="lbfgs",
        max_iter=5000,
        fit_intercept=True,
    )
    model.fit(preprocessing.transform(risk), y)
    return FittedHazardModel(model, preprocessing)


def hazard_increment(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return -np.log1p(-clipped)


def add_curve_columns(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["hazard_probability"] = probabilities
    result["h_increment"] = hazard_increment(probabilities)
    result["d_cumulative"] = np.nan
    result["a_second_difference"] = np.nan
    for _, index in result.groupby(["account", "game_id"], sort=False).groups.items():
        ordered = result.loc[index].sort_values("target_decision_number", kind="stable")
        h = ordered["h_increment"].to_numpy(dtype=float)
        result.loc[ordered.index, "d_cumulative"] = np.cumsum(h)
        result.loc[ordered.index, "a_second_difference"] = np.diff(h, prepend=h[0])
    if (result["d_cumulative"].groupby([result["account"], result["game_id"]]).diff().dropna() < -1e-12).any():
        raise AssertionError("d_cumulative is not monotonic")
    return result


def training_no_offbook_threshold(
    model: FittedHazardModel,
    training_frame: pd.DataFrame,
    training_games: pd.DataFrame,
    quantile: float,
) -> float:
    if not 0 < quantile < 1:
        raise ValueError("no-offbook quantile must lie strictly between zero and one")
    no_offbook_games = training_games.loc[training_games["judgment"].eq("no_offbook")]
    if no_offbook_games.empty:
        raise ValueError("training fold contains no no_offbook games")
    no_offbook = training_frame.loc[training_frame["judgment"].eq("no_offbook")].copy()
    maxima_by_game: dict[tuple[str, str], float] = {}
    if not no_offbook.empty:
        no_offbook["h"] = hazard_increment(model.predict_probability(no_offbook))
        maxima_by_game = no_offbook.groupby(["account", "game_id"], sort=False)["h"].max().to_dict()
    maxima = np.asarray([
        maxima_by_game.get((str(row.account), str(row.game_id)), 0.0)
        for row in no_offbook_games.itertuples(index=False)
    ])
    return float(np.quantile(maxima, quantile, method="higher"))


def cross_validated_curves(
    nodes: pd.DataFrame,
    marked_games: pd.DataFrame,
    regularization_c: float,
    no_offbook_quantile: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    accounts = sorted(nodes["account"].unique(), key=str.casefold)
    for held_out in accounts:
        train = nodes.loc[nodes["account"].ne(held_out)].copy()
        test = nodes.loc[nodes["account"].eq(held_out)].copy()
        train_games = marked_games.loc[marked_games["account"].ne(held_out)].copy()
        test_games = marked_games.loc[marked_games["account"].eq(held_out)].copy()
        model = fit_hazard_model(train, regularization_c)
        threshold = training_no_offbook_threshold(model, train, train_games, no_offbook_quantile)
        predicted = add_curve_columns(test, model.predict_probability(test))
        predicted["held_out_account"] = held_out
        predicted["fold_h_threshold"] = threshold
        outputs.append(predicted)
        folds.append({
            "heldOutAccount": held_out,
            "trainingPlayers": int(train["account"].nunique()),
            "trainingGames": int(len(train_games)),
            "testGames": int(len(test_games)),
            "trainingAtRiskNodes": int(train["at_risk"].sum()),
            "testNodes": int(len(test)),
            "hThresholdFromTrainingNoOffbookMaxima": threshold,
            "model": model.as_dict(),
        })
    combined = pd.concat(outputs, ignore_index=True)
    return combined.sort_values(["account", "game_id", "target_decision_number"], kind="stable"), folds


def game_predictions(nodes: pd.DataFrame, marked_games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (account, game_id), game in nodes.groupby(["account", "game_id"], sort=True):
        ordered = game.sort_values("target_decision_number", kind="stable")
        threshold = float(ordered["fold_h_threshold"].iloc[0])
        top = ordered.sort_values(
            ["h_increment", "target_decision_number"], ascending=[False, True], kind="stable"
        ).iloc[0]
        candidates = ordered.loc[ordered["h_increment"] >= threshold]
        first = candidates.iloc[0] if not candidates.empty else None
        anchor_rows = ordered.loc[ordered["is_manual_anchor"]]
        anchor = anchor_rows.iloc[0] if len(anchor_rows) == 1 else None
        top_distance = (
            abs(int(top["target_decision_number"]) - int(anchor["target_decision_number"]))
            if anchor is not None else np.nan
        )
        first_distance = (
            abs(int(first["target_decision_number"]) - int(anchor["target_decision_number"]))
            if first is not None and anchor is not None else np.nan
        )
        rows.append({
            "account": account,
            "game_id": game_id,
            "judgment": ordered["judgment"].iloc[0],
            "manual_offbook_ply": ordered["manual_offbook_ply"].iloc[0],
            "target_nodes": int(len(ordered)),
            "top_h_ply": int(top["strict_ply"]),
            "top_h_target_decision": int(top["target_decision_number"]),
            "top_h": float(top["h_increment"]),
            "top_h_distance_from_anchor": top_distance,
            "threshold": threshold,
            "has_threshold_candidate": first is not None,
            "first_candidate_ply": int(first["strict_ply"]) if first is not None else np.nan,
            "first_candidate_target_decision": (
                int(first["target_decision_number"]) if first is not None else np.nan
            ),
            "first_candidate_distance_from_anchor": first_distance,
        })
    existing = {(str(row["account"]), str(row["game_id"])) for row in rows}
    threshold_by_account = nodes.groupby("account", sort=False)["fold_h_threshold"].first().to_dict()
    for mark in marked_games.itertuples(index=False):
        key = (str(mark.account), str(mark.game_id))
        if key in existing:
            continue
        if mark.judgment != "no_offbook":
            raise ValueError(f"anchored game has no target nodes: {key}")
        rows.append({
            "account": mark.account,
            "game_id": mark.game_id,
            "judgment": mark.judgment,
            "manual_offbook_ply": mark.manual_offbook_ply,
            "target_nodes": 0,
            "top_h_ply": np.nan,
            "top_h_target_decision": np.nan,
            "top_h": np.nan,
            "top_h_distance_from_anchor": np.nan,
            "threshold": float(threshold_by_account[mark.account]),
            "has_threshold_candidate": False,
            "first_candidate_ply": np.nan,
            "first_candidate_target_decision": np.nan,
            "first_candidate_distance_from_anchor": np.nan,
        })
    return pd.DataFrame(rows)


def finite_rate(condition: pd.Series) -> float | None:
    return float(condition.mean()) if len(condition) else None


def build_summary(
    nodes: pd.DataFrame,
    games: pd.DataFrame,
    folds: list[dict[str, Any]],
    final_model: FittedHazardModel,
    inputs: list[dict[str, Any]],
    cohort_manifest_path: Path,
    frequency_book_path: Path,
    book: GlobalFrequencyBook,
    regularization_c: float,
    no_offbook_quantile: float,
) -> dict[str, Any]:
    anchored = games.loc[games["judgment"].eq("offbook")]
    no_offbook = games.loc[games["judgment"].eq("no_offbook")]
    at_risk = nodes.loc[nodes["at_risk"]]
    y = at_risk["is_manual_anchor"].to_numpy(dtype=int)
    score = at_risk["h_increment"].to_numpy(dtype=float)
    candidate_anchored = anchored.loc[anchored["has_threshold_candidate"]]
    return {
        "schema": "monotonic-offbook-evidence-fit-v1",
        "status": "completed",
        "featurePolicy": {
            "strictPly": "occupied squares in pre-move parent board minus 3",
            "thinkingTimeRatio": "raw_thinking_time_ms / source initial time limit in milliseconds",
            "thinkingTimeModelInput": (
                "log1p(thinking_time_ratio / 0.01); fixed 1% initial-clock scale"
            ),
            "cumulativeTimeRatio": (
                "reported for interpretation only; cumulative_time_ratio is not a model input"
            ),
            "discLoss": "log1p(lossClipped); unavailable loss is neutral mean-imputed",
            "parentChildRatio": "min(1, child node frequency / parent node frequency)",
            "missingParent": "parent_child_ratio is NaN",
            "missingChildWithinBook": "child frequency is 4",
            "globalRatio": "child frequency / sum of retained node frequencies at strict child ply",
            "missingChildGlobalDenominator": "add the imputed child frequency 4 to same-ply denominator",
            "globalRarity": "-log(global_ratio)",
            "afterBookMaxPly": "book-derived features are NaN and have neutral standardized contribution",
            "shapeConstraint": "none in v1; human average time/loss curves are not hard targets for d",
        },
        "modelPolicy": {
            "eventModel": "discrete-time logistic first-event hazard",
            "atRiskOffbookGame": "target decisions through and including exact manual anchor",
            "atRiskNoOffbookGame": "all target decisions",
            "h": "-log(1 - hazard_probability), always non-negative",
            "d": "within-game cumulative sum of h, monotonic by construction",
            "v": "h",
            "a": "first difference of h",
            "regularizationC": regularization_c,
            "crossValidation": "leave one player out",
            "threshold": (
                f"fold-training {no_offbook_quantile:.3f} quantile of per-game maximum h, method=higher"
            ),
        },
        "counts": {
            "players": int(nodes["account"].nunique()),
            "playerGameEvaluations": int(len(games)),
            "uniqueGames": int(games["game_id"].nunique()),
            "offbookGames": int(len(anchored)),
            "noOffbookGames": int(len(no_offbook)),
            "targetNodes": int(len(nodes)),
            "atRiskNodes": int(nodes["at_risk"].sum()),
            "manualAnchors": int(nodes["is_manual_anchor"].sum()),
            "bookExactChildFrequencyNodes": int(nodes["book_status"].eq("ok_exact_child_frequency").sum()),
            "bookImputedChildFrequencyNodes": int(nodes["book_status"].eq("ok_child_imputed_4").sum()),
            "bookBeyondMaxPlyNodes": int(nodes["book_status"].eq("beyond_book_max_ply").sum()),
            "bookBoardUnavailableNodes": int(nodes["book_status"].eq("board_unavailable").sum()),
        },
        "crossValidatedMetrics": {
            "nodeAnchorRocAuc": float(roc_auc_score(y, score)),
            "nodeAnchorAveragePrecision": float(average_precision_score(y, score)),
            "topHExactAnchorRate": finite_rate(anchored["top_h_distance_from_anchor"].eq(0)),
            "topHWithinOneTargetDecisionRate": finite_rate(anchored["top_h_distance_from_anchor"].le(1)),
            "topHWithinTwoTargetDecisionsRate": finite_rate(anchored["top_h_distance_from_anchor"].le(2)),
            "anchoredGameCandidateRate": finite_rate(anchored["has_threshold_candidate"]),
            "candidateExactAnchorRateAllAnchored": float(
                candidate_anchored["first_candidate_distance_from_anchor"].eq(0).sum() / len(anchored)
            ),
            "candidateWithinOneRateAllAnchored": float(
                candidate_anchored["first_candidate_distance_from_anchor"].le(1).sum() / len(anchored)
            ),
            "candidateWithinTwoRateAllAnchored": float(
                candidate_anchored["first_candidate_distance_from_anchor"].le(2).sum() / len(anchored)
            ),
            "noOffbookSpecificity": finite_rate(~no_offbook["has_threshold_candidate"]),
        },
        "finalAllPlayerModel": final_model.as_dict(),
        "folds": folds,
        "inputs": {
            "cohortManifest": str(cohort_manifest_path),
            "cohortManifestSha256": sha256_file(cohort_manifest_path),
            "cohorts": inputs,
            "frequencyBook": str(frequency_book_path),
            "frequencyBookSha256": sha256_file(frequency_book_path),
            "frequencyBookMetadata": book.metadata,
        },
        "limitations": [
            "The global Frequency Book retains only nodes with frequency at least 5 through ply 30.",
            "A missing child within ply 30 is assigned frequency 4 by the frozen v1 policy.",
            "Node-frequency ratios are descriptive proxies, not exact parent-to-child transition probabilities.",
            "Manual anchors were primarily selected from timing continuity, so this is not independent validation of psychological book knowledge.",
            "No multiscale detector is fitted in this first curve-fitting stage; top-h and threshold results are initial diagnostics.",
        ],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    metrics = summary["crossValidatedMetrics"]
    coefficients = summary["finalAllPlayerModel"]["standardizedCoefficients"]
    lines = [
        "# 第一版单调累计脱谱证据曲线拟合",
        "",
        "## 样本与方法",
        "",
        f"- {counts['players']} 位选手，{counts['playerGameEvaluations']} 个选手－对局；{counts['offbookGames']} 局有人工锚点，{counts['noOffbookGames']} 局为 `no_offbook`。",
        f"- 共 {counts['targetNodes']} 个目标决策节点，其中首次事件风险集 {counts['atRiskNodes']} 个、人工锚点 {counts['manualAnchors']} 个。",
        "- 使用五项输入：严格棋盘 `ply`、`log(1 + 本步耗时比例/1%)`、子损、全局节点母子频率比、同 `ply` 全局罕见度。",
        "- 原始本步耗时比例仍逐行输出；时间变换以初始总时间的 1% 为固定尺度，并压缩极长思考的影响。",
        "- 另输出截至当前的累计用时比例用于解释；它不是模型输入，避免在 `d` 中二次累计时间。",
        "- 采用按棋手留一预测；`h=-log(1-p)` 非负，`d` 为 `h` 的局内累计和。",
        "",
        "## 留一验证初步结果",
        "",
        f"- 节点锚点 ROC-AUC / AP：{metrics['nodeAnchorRocAuc']:.3f} / {metrics['nodeAnchorAveragePrecision']:.3f}。",
        f"- 每局最大 `h` 精确命中 / 前后1步 / 前后2步：{metrics['topHExactAnchorRate']:.1%} / {metrics['topHWithinOneTargetDecisionRate']:.1%} / {metrics['topHWithinTwoTargetDecisionsRate']:.1%}。",
        f"- 按训练折 `no_offbook` 整局最大值阈值，锚点局候选覆盖率：{metrics['anchoredGameCandidateRate']:.1%}；`no_offbook` 特异度：{metrics['noOffbookSpecificity']:.1%}。",
        "",
        "## 全样本最终模型的标准化系数",
        "",
    ]
    for feature in MODEL_FEATURES:
        lines.append(f"- `{feature}`：{coefficients[feature]:.6f}")
    lines.extend([
        "",
        "## 边界",
        "",
        "这是第一版曲线拟合与留一验证，不是正式脱谱判定器。Frequency Book 的节点频率比不是严格转移概率，且 Book 只覆盖到 `ply 30`。本阶段尚未加入多尺度定位或曲线形状硬约束。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cohort_manifest_path = args.cohort_manifest.resolve()
    frequency_book_path = args.frequency_book.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    book = GlobalFrequencyBook.load(frequency_book_path)
    nodes, marked_games, inputs = load_all_nodes(cohort_manifest_path, book)
    cv_nodes, folds = cross_validated_curves(
        nodes,
        marked_games,
        args.regularization_c,
        args.no_offbook_quantile,
    )
    games = game_predictions(cv_nodes, marked_games)
    final_model = fit_hazard_model(nodes, args.regularization_c)
    summary = build_summary(
        cv_nodes,
        games,
        folds,
        final_model,
        inputs,
        cohort_manifest_path,
        frequency_book_path,
        book,
        args.regularization_c,
        args.no_offbook_quantile,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    cv_nodes.to_csv(output_dir / "cross_validated_node_curves.csv", index=False, encoding="utf-8")
    games.to_csv(output_dir / "cross_validated_game_summary.csv", index=False, encoding="utf-8")
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "outputDir": str(output_dir),
        "counts": summary["counts"],
        "crossValidatedMetrics": summary["crossValidatedMetrics"],
        "standardizedCoefficients": summary["finalAllPlayerModel"]["standardizedCoefficients"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
