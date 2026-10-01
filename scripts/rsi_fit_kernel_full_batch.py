"""Consumed-only optimizer comparison on an already audited physical bank.

The bank is inherited evidence, not newly collected/re-reviewed physics in this
command. Gaussian likelihood is revalidated against the actual current parent;
the resulting candidate requires a new independent physical evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.kernel_full_batch_learning import fit_update
from rosclaw_soccer.sim.contracts import hash_bytes
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-model", "rollout-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = _sealed(args.rollout_root / "rollout_manifest.json")
    path = args.rollout_root / "rollouts.npz"
    if (
        manifest["partition"] != "TRAIN_CONSUMED"
        or manifest["physical_rollout_count"] != 64
        or manifest["frame_sample_count"] != 17280
        or manifest["independent_contexts"] != 4
        or manifest["backend_count"] != 2
        or manifest["data_hash"] != hash_bytes(path.read_bytes())
        or any(
            manifest.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
        )
        or args.output.exists()
    ):
        parser.error("exact complete inherited consumed physical bank and fresh output required")
    parent = json.loads(args.parent_model.read_text())
    with np.load(path, allow_pickle=False) as data:
        candidate = fit_update(parent, data, batch_hash=manifest["report_hash"])
    write_once(args.output, candidate)
    print(
        json.dumps(dict(model_hash=candidate["model_hash"], receipt=candidate["learning_receipt"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
