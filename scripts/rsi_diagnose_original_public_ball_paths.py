"""Diagnose authenticated four public mesh controls without new physics."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi import first_touch_ball_path_diagnostics as diagnostics
from rosclaw_soccer.rsi import first_touch_outcome
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("exact clean frozen diagnostic source and unused output required")
    cfg = json.loads(args.config.read_text())
    review_path, complete_path = Path(cfg["nominal_readback"]), Path(cfg["nominal_complete"])
    binding = ReviewedContactWorld(review_path, cfg["nominal_readback_file_hash"]).verify(
        cfg["mesh_task_contact_world_hash"]
    )
    review = json.loads(review_path.read_text())
    complete = json.loads(complete_path.read_text())
    if (
        complete["report_hash"]
        != hash_json({k: v for k, v in complete.items() if k != "report_hash"})
        or review["complete_hash"] != complete["report_hash"]
        or len(complete["results"]) != 4
        or [r["group"] for r in complete["results"]] != [4, 8, 32, 56]
    ):
        raise ValueError("entire independently read four-draw control bank required")
    pins = {
        str(p): digest(p)
        for p in (
            Path(__file__).resolve(),
            Path(diagnostics.__file__),
            Path(first_touch_outcome.__file__),
            args.config,
            review_path,
            complete_path,
        )
    }
    rows = []
    for receipt, audited in zip(complete["results"], review["rows"], strict=True):
        if (
            receipt["receipt_hash"]
            != hash_json({k: v for k, v in receipt.items() if k != "receipt_hash"})
            or receipt["receipt_hash"] != audited["receipt_hash"]
            or receipt["outcome"] != audited["outcome"]
        ):
            raise ValueError("exact independent receipt and primitive labels required")
        witnesses = dict(receipt["witnesses"])
        witnesses.update({k: receipt[k] for k in ("intent_witness", "encoder_witness")})
        for witness in witnesses.values():
            path = Path(witness["path"])
            if path.is_symlink() or not path.is_file() or digest(path) != witness["sha256"]:
                raise ValueError("all original independently reviewed witness bytes required")
            pins[str(path)] = witness["sha256"]
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as archive:
            result = diagnostics.diagnose(dict(archive))
        if result["outcome"] != receipt["outcome"]:
            raise ValueError("measured physical outcome changed")
        rows.append(
            dict(group=receipt["group"], receipt_hash=receipt["receipt_hash"], diagnostics=result)
        )
    if any(digest(Path(p)) != value for p, value in pins.items()):
        raise ValueError("diagnostic source/input changed")
    result = dict(
        schema="soccer.rsi.original_public_ball_path_diagnostics.v1",
        source_head=args.source_head,
        numerical_review_binding=binding,
        whole_reviewed_witness_bytes_authenticated=True,
        physical_proof_logical_seals_reopened_here=False,
        input_file_hashes=pins,
        rows=rows,
        new_physics_steps=0,
        new_optimizer_updates=0,
        causal_intervention_performed=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
