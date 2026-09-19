"""Verify every pass supplement record with the retained independent C++ rules tool."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<8sHHQ12x")
RECORD_SIZE = 25
BOARD_BYTES = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_board(packed: np.ndarray) -> str:
    symbols = "-XO?"
    cells = [symbols[(int(packed[index // 4]) >> ((index % 4) * 2)) & 3] for index in range(64)]
    if "?" in cells:
        raise ValueError("packed board contains the reserved 2-bit code 3")
    return "".join(cells)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--cpp-source", type=Path, required=True)
    parser.add_argument("--cpp-exe", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()

    supplement = args.supplement.resolve(strict=True)
    cpp_source = args.cpp_source.resolve(strict=True)
    cpp_exe = args.cpp_exe.resolve(strict=True)
    audit_dir = args.audit_dir.resolve()
    input_dir = audit_dir / "input"
    output_dir = audit_dir / "output"
    input_path = input_dir / "0000000.txt"
    output_path = output_dir / "0000000.bcnn"
    audit_manifest = audit_dir / "verification_manifest.json"
    existing = [path for path in (input_path, output_path, audit_manifest) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing audit artifacts: {existing}")

    with np.load(supplement, allow_pickle=False) as data:
        packed_boards = np.ascontiguousarray(data["packed_boards"])
        sides = data["side_to_move"].tolist()
        game_ids = data["game_id"].tolist()
        move_indices = data["move_index"].tolist()
    if packed_boards.ndim != 2 or packed_boards.shape[1] != BOARD_BYTES:
        raise ValueError("supplement packed_boards must have shape Nx16")
    if any(side not in ("black", "white") for side in sides):
        raise ValueError("supplement contains an unsupported current side")

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open("w", encoding="utf-8", newline="\n") as handle:
        for packed in packed_boards:
            handle.write(f"{unpack_board(packed)} 0\n")

    command = [
        str(cpp_exe),
        "--source-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--threads", "1",
        "--first-index", "0",
        "--last-index", "0",
    ]
    completed = subprocess.run(
        command,
        cwd=cpp_exe.parent,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )

    with output_path.open("rb") as handle:
        magic, version, record_size, records = HEADER.unpack(handle.read(HEADER.size))
        raw = np.frombuffer(handle.read(), dtype=np.uint8).reshape(records, RECORD_SIZE)
    if (magic, version, record_size) != (b"BCNNDS01", 1, RECORD_SIZE):
        raise ValueError("C++ verifier emitted an unsupported shard header")
    if records != packed_boards.shape[0]:
        raise ValueError(f"C++ verifier record count differs: {records} != {packed_boards.shape[0]}")
    cpp_boards = np.ascontiguousarray(raw[:, :BOARD_BYTES])
    cpp_legal = np.ascontiguousarray(raw[:, BOARD_BYTES:BOARD_BYTES + 8]).view("<u8").reshape(records)
    board_mismatches = np.flatnonzero(np.any(cpp_boards != packed_boards, axis=1))
    nonzero_legal = np.flatnonzero(cpp_legal != 0)
    if board_mismatches.size or nonzero_legal.size:
        sample_indices = np.unique(np.concatenate((board_mismatches[:5], nonzero_legal[:5])))
        samples = [
            {
                "recordIndex": int(index),
                "gameId": game_ids[int(index)],
                "moveIndex": int(move_indices[int(index)]),
                "sideToMove": sides[int(index)],
                "cppLegalBitboard": int(cpp_legal[int(index)]),
            }
            for index in sample_indices
        ]
        raise RuntimeError(
            f"C++ pass verification failed: boardMismatches={board_mismatches.size}, "
            f"nonzeroLegal={nonzero_legal.size}, samples={samples}"
        )

    manifest = {
        "schema": "transformer-legal-pass-cpp-verification-v1",
        "status": "passed",
        "records": int(records),
        "currentSideContract": "each fixed-color board is normalized so X is the recorded pass side to move",
        "cppRuleResult": "all legal-move uint64 bitboards are zero",
        "boardRoundTripMismatches": int(board_mismatches.size),
        "nonzeroLegalMoveRecords": int(nonzero_legal.size),
        "supplement": str(supplement),
        "supplementSha256": sha256_file(supplement),
        "cppSource": str(cpp_source),
        "cppSourceSha256": sha256_file(cpp_source),
        "cppExecutable": str(cpp_exe),
        "cppExecutableSha256": sha256_file(cpp_exe),
        "cppInput": str(input_path),
        "cppInputSha256": sha256_file(input_path),
        "cppOutput": str(output_path),
        "cppOutputSha256": sha256_file(output_path),
        "cppStdout": completed.stdout.strip(),
        "cppStderr": completed.stderr.strip(),
        "encoding": "UTF-8",
    }
    write_json(audit_manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
