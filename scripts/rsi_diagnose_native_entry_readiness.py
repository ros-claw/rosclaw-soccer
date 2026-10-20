"""Authenticate the closed calibrated pilot's failed initiation, not retrain it."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-pilot-hash", "pilot-unit"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("clean frozen read-only diagnostic source required")
    wait_for_terminal_collection(a.pilot_unit)
    if digest(a.pilot / "complete.json") != a.expected_pilot_hash:
        raise ValueError("exact full repeated pilot receipt required")
    c = json.loads((a.pilot / "complete.json").read_text())
    d = json.loads((a.pilot / "declaration.json").read_text())
    if c["status"] != "NATIVE_ENTRY_FIT_THREE_INTENTS_REPRODUCED" or c["native_steps"] != 18000:
        raise ValueError("full actual pilot required")
    pins = dict(d["input_file_hashes"])
    pins.update({str(f): digest(f) for f in a.pilot.rglob("*") if f.is_file()})
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.entry_readiness_diagnostic import diagnose_readiness
    from rosclaw_soccer.rsi.motion_entry_profile import build_motion_entry
    from rosclaw_soccer.rsi.native_entry_calibration import (
        NativeEntryCalibration,
        bind_native_entry,
    )
    from scripts.rsi_pilot_motion_entry import compare_episode
    from scripts.rsi_probe_compact_source import load_arrays
    from scripts.rsi_review_motion_entry_pilot import review_episode

    world_path = next(Path(k) for k in pins if k.endswith(".mjb"))
    motion_path = next(Path(k) for k in pins if k.endswith(".csv"))
    prior_path = next(Path(k) for k in pins if Path(k).name == "entry-affordance.json")
    world = mujoco.MjModel.from_binary_path(str(world_path))
    poses = np.loadtxt(motion_path, delimiter=",", skiprows=1, dtype=np.float64)
    saved = json.loads(prior_path.read_text())
    prior = SkillEntryAffordance(
        tuple(saved["ball_offset_xy_m"]),
        saved["historical_frame"],
        saved["historical_forward_speed_mps"],
        saved["historical_artifact_hash"],
    )
    fields = (
        "motion_configuration_hash",
        "parent_affordance_hash",
        "audit_hash",
        "sample_hashes",
        "mean_displacement_xy_m",
        "maximum_training_residual_m",
    )
    values = {k: d["fit"][k] for k in fields}
    for key in ("sample_hashes", "mean_displacement_xy_m"):
        values[key] = tuple(values[key])
    fit = NativeEntryCalibration(**values)
    if fit.contract_hash != d["fit_hash"] or fit.contract_hash != c["fit_hash"]:
        raise ValueError("fit lineage differs")
    b, e, source = build_motion_entry(
        poses,
        prior,
        world_hash=digest(world_path),
        motion_artifact_hash=digest(motion_path),
        reference_side="right",
        source_window=(466, 767),
        transport=False,
    )
    b, e = bind_native_entry(b, e, fit)
    b.validate(
        source, e, world_hash=digest(world_path), source_window=(466, 767), reference_side="right"
    )
    results = []
    for law, row in zip((0, 3, 4), c["results"], strict=True):
        folder = f"right-law{law}"
        raw = load_arrays(a.pilot / "produce" / folder / "native300.npz")
        repeat = json.loads((a.pilot / "repeat" / folder / "complete.json").read_text())
        compare_episode(a.pilot / "repeat" / folder, a.pilot / "produce" / folder, repeat)
        if (
            row != json.loads((a.pilot / "produce" / folder / "complete.json").read_text())
            or row["law_index"] != law
        ):
            raise ValueError("actual complete episode order differs")
        review_episode(world, raw, row)
        results.append(dict(law_index=law, **diagnose_readiness(raw, e, law_index=law)))
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="CALIBRATED_ENTRY_INITIATION_FAILURE_DIAGNOSED",
            source_head=a.expected_head,
            pilot_hash=a.expected_pilot_hash,
            input_file_hashes=pins,
            results=results,
            source_kick_geometry_evaluated=False,
            new_native_steps=0,
            new_neural_calls=0,
            optimizer_steps=0,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
