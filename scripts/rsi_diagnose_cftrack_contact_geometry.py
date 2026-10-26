"""Compare old local and new absolute reference IK on one OWN predicted contact.

Endpoint FK only, zero new NN or physical steps; not a usable teacher yet.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--future", type=Path, required=True)
    p.add_argument("--future-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.future / "complete.json"
    if a.output.exists() or complete.is_symlink() or digest(complete) != a.future_hash:
        raise ValueError("unused output and exact qualified own-future source required")
    report = json.loads(complete.read_text())
    if report["status"] != "CFTRACK_OWN_NN_NATIVE_FUTURE_EXACT":
        raise ValueError("qualified private native current-body NN future required")
    pins = dict(report["input_file_hashes"])
    pins[str(complete)] = a.future_hash
    raw_path = a.future / "owned-future.npz"
    if raw_path.is_symlink() or pins.get(str(raw_path)) != digest(raw_path):
        raise ValueError("source-bound original own prediction required")
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.contact_reference_ik import (
        first_contact_foot,
        solve_absolute_contact_pose,
        solve_contact_reference_pose,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene

    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique standard native body required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    with np.load(raw_path, allow_pickle=False) as raw:
        contact = first_contact_foot(raw["ball_forces"][:, 0, :2])
        if contact is None:
            raise ValueError("predicted first-foot event required")
        side, native = contact
        peak = native // 10 - 1
        if peak < 1:
            raise ValueError("pre-contact seed required")
        d = mujoco.MjData(model)
        d.qpos[:], d.qvel[:] = raw["qpos"][peak * 10 - 1], raw["qvel"][peak * 10 - 1]
    goal = np.array([5.0, 0.0, 0.11])
    local = solve_contact_reference_pose(model, d, layout, goal=goal, active_foot=side)
    absolute = solve_absolute_contact_pose(model, d, layout, goal=goal, active_foot=side)
    q = np.array(absolute["reference_joints_hg"])
    # Changed reference leg configurations must remain inside native ranges;
    # actual upper-body/root/ball seed is never repaired or changed.
    limits = model.jnt_range[layout.joints][:12]
    if (q[:12] < limits[:, 0]).any() or (q[:12] > limits[:, 1]).any():
        raise ValueError("absolute leg reference violates native joint range")
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_CONTACT_ENDPOINT_GEOMETRY_ONLY",
            source_head=a.expected_head,
            input_file_hashes=pins,
            active_foot=side,
            first_foot_native=native,
            reference_peak_control=peak,
            old_local_geometry=local,
            new_absolute_geometry=absolute,
            new_neural_calls=0,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            taskspace_reference_or_actual_policy_not_yet_modified=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
