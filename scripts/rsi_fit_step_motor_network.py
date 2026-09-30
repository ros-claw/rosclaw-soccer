"""Fit an explicitly unqualified per-frame motor warm start from sealed physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.step_motor_network import fit_model
from rosclaw_soccer.sim.contracts import hash_bytes
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    manifest = _sealed(args.bank_root / "manifest.json")
    path = args.bank_root / "step_motor_teachers.npz"
    if hash_bytes(path.read_bytes()) != manifest["data_hash"]:
        parser.error("teacher data differs from sealed physical manifest")
    with np.load(path, allow_pickle=False) as arrays:
        model = fit_model(manifest, arrays, epochs=args.epochs)
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_once(args.output_root / "model.json", model)
    print(
        json.dumps(
            dict(
                model_hash=model["model_hash"],
                metrics=model["metrics"],
                physics_qualified=False,
                online_rl_qualified=False,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
