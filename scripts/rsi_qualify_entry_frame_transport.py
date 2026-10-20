"""Root-frame entry-point geometry qualification, zero NN/native integration.

The result is an UNADMITTED estimate, not a skill, actor, motion or physical
contact certificate. No body state, actuator, robot transport or world changes.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("compact-review", "inputs", "motion-csv", "prior", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "expected-compact-review-hash", "compact-review-unit"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or digest(a.compact_review / "complete.json") != a.expected_compact_review_hash
    ):
        raise ValueError("frozen geometry source and exact closed compact failure required")
    wait_for_terminal_collection(a.compact_review_unit)
    review = json.loads((a.compact_review / "complete.json").read_text())
    if (
        review["status"] != "COMPACT_SOURCE_LEAD24_RAW_REPRODUCTION_AUTHENTICATED"
        or review["summary"]["old_success"] != 4
        or review["summary"]["new_success"] != 0
        or review["summary"]["contexts"] != 24
    ):
        raise ValueError("whole24 declared failure context required")
    pins = dict(review["input_file_hashes"])
    for f in (a.inputs, a.motion_csv, a.prior):
        if pins.get(str(f)) != digest(f):
            raise ValueError("exact previously bound source/prior identities required")
    pins[str(a.compact_review / "complete.json")] = a.expected_compact_review_hash
    pins.update(
        {
            str(f): digest(f)
            for root in (Path.cwd() / "src", Path.cwd() / "scripts")
            for f in root.rglob("*.py")
        }
    )
    check_pins(pins)
    import numpy as np

    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.bilateral_entry import ReflectedEntryAffordance
    from rosclaw_soccer.rsi.entry_frame_transport import transport_entry_frame

    source = np.loadtxt(a.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
    roots = source[:, :7].copy()
    stored = json.loads(a.prior.read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("past actual motion entry contract changed")
    right = transport_entry_frame(
        prior,
        roots,
        original_source_frame=286,
        target_source_frame=466,
        source_motion_artifact_hash=digest(a.motion_csv),
    )
    reflected = roots.copy()
    reflected[:, 1] *= -1
    reflected[:, [4, 6]] *= -1
    left_prior = ReflectedEntryAffordance(
        (prior.ball_offset_xy_m[0], -prior.ball_offset_xy_m[1]),
        prior.historical_frame,
        prior.historical_forward_speed_mps,
        prior.historical_artifact_hash,
        prior.contract_hash,
    )
    left = transport_entry_frame(
        left_prior,
        reflected,
        original_source_frame=286,
        target_source_frame=466,
        source_motion_artifact_hash=digest(a.motion_csv),
    )
    np.testing.assert_allclose(
        left.transported_ball_offset_xy_m,
        [right.transported_ball_offset_xy_m[0], -right.transported_ball_offset_xy_m[1]],
        atol=1e-12,
        rtol=0,
    )
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="ENTRY_FRAME_POINT_TRANSPORT_GEOMETRY_QUALIFIED_NOT_DYNAMIC",
            source_head=a.expected_head,
            input_file_hashes=pins,
            right=dict(report=right.report(), contract_hash=right.contract_hash),
            left=dict(report=left.report(), contract_hash=left.contract_hash),
            source_reference_window=[466, 767],
            original_reference_window=[286, 767],
            nominal_source_frequency_hz=120,
            entry_intervals=50,
            nominal_reference_cue_after_handoff_frames=75,
            left_root_reflection_qualified_not_29_joint_admission=True,
            source_reference_admission_reused_not_reperformed=True,
            actual_body_or_ball_changed=False,
            controller_or_policy_changed=False,
            new_native_steps=0,
            new_neural_calls=0,
            optimizer_steps=0,
            dynamic_feasibility_proven=False,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
