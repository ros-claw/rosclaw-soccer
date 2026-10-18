"""Read a declared receipt prefix; reconstruct PD projection, never physics.

Checks complete original witness bytes and primitive outcome consistency.
Historical physical replay reports are authenticated, not re-executed here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.cerebellar_torque_transmission import diagnose_transmission
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.count <= 44:
        raise ValueError("unused output and explicit bounded receipt prefix required")
    root = args.root.resolve()
    pins = {}

    def pin(path):
        if not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError("ordinary original evidence required")
        actual = digest(path)
        if str(path) in pins and pins[str(path)] != actual:
            raise ValueError("input changed during diagnosis")
        pins[str(path)] = actual
        return actual

    def sealed(path, key="report_hash"):
        pin(path)
        value = json.loads(path.read_text())
        if value.get(key) != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("original diagnostic source seal differs")
        return value

    for relative in (
        "scripts/rsi_diagnose_torque_transmission.py",
        "src/rosclaw_soccer/rsi/cerebellar_torque_transmission.py",
        "src/rosclaw_soccer/rsi/cerebellar_torque_policy.py",
        "src/rosclaw_soccer/rsi/first_touch_outcome.py",
        "src/rosclaw_soccer/sim/contracts.py",
    ):
        pin(Path(__file__).resolve().parents[1] / relative)
    declaration = sealed(root / "declaration.json")
    rows = []
    for index in range(args.count):
        receipt = sealed(root / f"receipt-{index:04d}.json", "receipt_hash")
        if receipt["job"] != declaration["jobs"][index]["job"]:
            raise ValueError("entire ordered prefix including failed outcomes required")
        witnesses = dict(receipt["witnesses"])
        for key in ("intent_witness", "encoder_witness", "model_witness"):
            witnesses[key] = receipt[key]
        for witness in witnesses.values():
            path = Path(witness["path"])
            if not path.resolve().is_relative_to(root) or pin(path) != witness["sha256"]:
                raise ValueError("whole experiment-owned original witness bytes differ")
        commitment = json.loads(Path(witnesses["commitment"]["path"]).read_text())
        independent = sealed(Path(witnesses["independent_review"]["path"]))
        if (
            independent["report_hash"] != receipt["independent_review_hash"]
            or independent.get("actual_mujoco_dynamics_replayed") is not True
            or independent.get("actual_pd_torque_reconstructed") is not True
            or commitment["compiled_model_hash"] != declaration["mesh_task_contact_world_hash"]
        ):
            raise ValueError("existing independent PD/physics replay required")
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        outcome = label_first_touch(trace).to_dict()
        if outcome != receipt["outcome"]:
            raise ValueError("unchanged primitive outcome required")
        measured = diagnose_transmission(
            trace,
            kp=commitment["physics"]["canonical_pd_kp"],
            kd=commitment["physics"]["canonical_pd_kd"],
        )
        rows.append(
            dict(
                index=index,
                job=receipt["job"],
                receipt_hash=receipt["receipt_hash"],
                outcome=outcome,
                sampling_draws=receipt["cerebellar_sampling_draws"],
                model_optimizer_steps=receipt["cerebellar_model_optimizer_steps"],
                measured=measured,
            )
        )
        print("PD_TRANSMISSION_RECONSTRUCTED", index, flush=True)
        del trace
    if any(digest(Path(name)) != expected for name, expected in pins.items()):
        raise ValueError("frozen inputs changed during diagnosis")
    result = dict(
        schema="soccer.rsi.cerebellar_torque_transmission_prefix_readback.v1",
        declaration_hash=declaration["report_hash"],
        prefix_count=args.count,
        all_prefix_receipts_including_failures_retained=True,
        rows=rows,
        input_file_hashes=pins,
        historical_replay_report_bytes_checked_not_physics_reexecuted=True,
        full_physical_report_logical_payload_reconstructed_here=False,
        parameter_to_outcome_causality_established=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
