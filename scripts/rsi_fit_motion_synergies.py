"""Fit a SIM motion subspace from explicitly successful executed public B traces."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--pilot-hash", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    cp = a.pilot / "complete.json"
    if a.output.exists() or digest(cp) != a.pilot_hash:
        raise ValueError("unused output and exact closed public pilot required")
    report = json.loads(cp.read_text())
    if report["status"] != "BOUNDED_SURFACE_CONTACT_PILOT_COMPLETE":
        raise ValueError("closed native paired pilot required")
    pins = dict(report["input_file_hashes"])
    pins[str(cp)] = a.pilot_hash
    import numpy as np

    from rosclaw_soccer.rsi.motion_synergy_reference import fit_motion_synergies

    samples, records = [], []
    for pair in report["results"]:
        ref = pair["arms"]["B"]
        path = a.pilot / ref["record"]
        if digest(path) != ref["hash"]:
            raise ValueError("changed B record")
        row = json.loads(path.read_text())
        pins[str(path)] = ref["hash"]
        if not row["outcome"]["task_pass"] or not row["healthy_full_horizon"]:
            continue
        raw = path.parent / row["artifact"]["path"]
        if digest(raw) != row["artifact"]["hash"]:
            raise ValueError("changed executed native trajectory")
        pins[str(raw)] = row["artifact"]["hash"]
        with np.load(raw, allow_pickle=False) as data:
            targets = data["joint_targets"][row["entry_control"] : 300].copy()
        innovations = np.diff(targets, axis=0)
        samples.append(innovations)
        records.append(
            dict(
                course_index=pair["course_index"],
                record_hash=ref["hash"],
                samples=len(innovations),
                entry_control=row["entry_control"],
            )
        )
    if len(records) < 4:
        raise ValueError("multiple distinct successful public motions required")
    basis, fit = fit_motion_synergies(np.concatenate(samples).astype(np.float64), rank=6)
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    artifact = a.output / "synergy-basis.npz"
    np.savez_compressed(artifact, basis_hg=basis)
    pins[str(artifact)] = digest(artifact)
    write_once(
        a.output / "complete.json",
        dict(
            status="PUBLIC_EXECUTED_MOTION_SYNERGIES_FITTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            artifact=dict(path=artifact.name, hash=digest(artifact)),
            fit=fit,
            records=records,
            training_data_is_consumed_public=True,
            independent_fresh_cases=0,
            neural_actor_updates=0,
            physics_qualified=False,
            new_physical_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
