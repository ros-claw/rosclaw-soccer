"""Consumed AWR-inspired comparison from the independently reconstructed bank.

This command rechecks the saved bank binding, not every physical execution.
It does not open a fresh pool, select a deployed policy or grant authority.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.kernel_replay_motor import fit_update
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.sim.contracts import hash_bytes
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rollout-root", "zero-model", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = _sealed(args.rollout_root / "rollout_manifest.json")
    path = args.rollout_root / "rollouts.npz"
    parent = json.loads(args.zero_model.read_text())
    records = manifest["records"]
    if (
        manifest["schema"] != "soccer.rsi.expanded_failure_motor_rollout_bank.v1"
        or manifest["partition"] != "TRAIN_CONSUMED"
        or manifest["data_hash"] != hash_bytes(path.read_bytes())
        or manifest["parent_model_hash"] != parent["model_hash"]
        or manifest["physical_rollout_count"] != len(records)
        or manifest["frame_sample_count"] != len(records) * 270
        or manifest["independent_contexts"] != len({(r["seed"], r["lane"]) for r in records})
        or manifest["backend_count"] != 2
        or {r["backend"] for r in records} != {"IsaacLab", "MuJoCo"}
        or any(
            manifest.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("sealed complete consumed actual replay bank required")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    for i, record in enumerate(records):
        ids = slice(i * 270, (i + 1) * 270)
        if (
            record["group"] != i
            or not np.all(arrays["trajectory_index"][ids] == i)
            or not np.all(arrays["terminal_return"][ids] == terminal_return(record["outcome"]))
        ):
            raise ValueError("measured returns or trajectory grouping changed")
    model = fit_update(parent, arrays, batch_hash=manifest["report_hash"])
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_once(args.output_root / "model.json", model)
    print(
        json.dumps(dict(model_hash=model["model_hash"], receipt=model["learning_receipt"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
