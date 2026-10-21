"""TRAIN-only exposure diagnostic; zero physics, inference, gradients or selection."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exam-root", required=True, type=Path)
    p.add_argument("--exam-hash", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    receipt = a.exam_root / "complete.json"
    if a.output.exists() or digest(receipt) != a.exam_hash:
        raise ValueError("unused output and exact completed exam required")
    r = json.loads(receipt.read_text())
    if (
        r["task_set"] != "train64"
        or r["checkpoint_intent"] != "bounded-recovery-continuation"
        or r["all_strict_replays_exact"] is not True
    ):
        raise ValueError("complete native TRAIN64 continuation exam required")
    pins = {**r["input_file_hashes"], str(receipt): a.exam_hash}
    pins[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    banks = [Path(x) for x in pins if x.endswith("/precontact-resets.npz")]
    if len(banks) != 1:
        raise ValueError("one exact measured reset bank required")
    check_pins(pins)
    with np.load(banks[0], allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in z.files}
    rows = sorted(
        [x for x in r["results"] if x["arm"] == "final" and x["repeat"] == 0],
        key=lambda x: x["world"],
    )
    if [x["world"] for x in rows] != list(range(64)):
        raise ValueError("full64 actual final trajectories required")
    outcomes = []
    for row in rows:
        w = row["world"]
        ids = np.flatnonzero((bank["world"] == w) & (bank["lead_controls"] == 10))
        if len(ids) != 1:
            raise ValueError("unambiguous source course reset required")
        index = int(ids[0])
        frame = int(bank["reference_frame"][index])
        path = a.exam_root / f"world{w}-final-repeat0/native.npz"
        if digest(path) != row["raw_hash"]:
            raise ValueError("raw actual exam changed")
        pins[str(path)] = row["raw_hash"]
        with np.load(path, allow_pickle=False) as z:
            obs, recovery = z["observation"].copy(), z["recovery_phase"].copy()
        enabled = (
            (np.linalg.norm(obs[:, 168:170], axis=1) < 0.7)
            & (np.abs(obs[:, 168:177]).sum(axis=1) > 0)
            & (recovery == 0)
        )
        comparable = frame < len(obs) and recovery[frame] == 0
        outcomes.append(
            dict(
                world=w,
                training_reset_frame=frame,
                near_gate_controls=int(enabled.sum()),
                pre_reset_near_gate_controls=int(enabled[:frame].sum()),
                reset_frame_already_recovery_or_ended=not comparable,
                reset_frame_observation_max_abs_difference=(
                    float(np.abs(obs[frame] - bank["observation"][index]).max())
                    if comparable
                    else None
                ),
            )
        )
    check_pins(pins)
    total = sum(x["near_gate_controls"] for x in outcomes)
    early = sum(x["pre_reset_near_gate_controls"] for x in outcomes)
    write_once(
        a.output,
        dict(
            schema="soccer.rsi.option_prefix_exposure_diagnostic.v1",
            input_file_hashes=pins,
            outcomes=outcomes,
            near_gate_controls=total,
            pre_reset_near_gate_controls=early,
            pre_reset_fraction=early / total if total else None,
            worlds_with_pre_reset_gate=sum(x["pre_reset_near_gate_controls"] > 0 for x in outcomes),
            new_physics_steps=0,
            new_neural_rows=0,
            optimizer_updates=0,
            exposure_mismatch_not_proof_of_sole_failure_cause=True,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
