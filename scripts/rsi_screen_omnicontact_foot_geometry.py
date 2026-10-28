"""Measure reference-only foot/ball geometry in the existing MuJoCo body.

Kinematics are evaluated on copied source frames. No dynamics, controller,
source-frame adjustment, learning or real-robot claim is made.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.training.omnicontact_reference import LAB_TO_MUJOCO
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset-root", "screen", "scene", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("screen-hash", "scene-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.screen.is_symlink()
        or a.scene.is_symlink()
        or digest(a.screen) != a.screen_hash
        or digest(a.scene) != a.scene_hash
    ):
        raise ValueError("unused external output and pinned screen/scene required")
    screen = json.loads(a.screen.read_text())
    if (
        screen.get("status") != "OMNICONTACT_TRAIN_RECEIVING_REFERENCE_SCREEN_NOT_PHYSICS"
        or screen.get("validation_or_test_motion_opened") is not False
        or screen.get("rejected_inputs") != []
        or screen.get("promotion_authorized") is not False
    ):
        raise ValueError("complete training-only source screen required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    ball_geoms = np.flatnonzero(model.geom_bodyid == layout.ball)
    if len(ball_geoms) != 1 or model.geom_type[ball_geoms[0]] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError("one declared spherical standard soccer ball required")
    radius = float(model.geom_size[ball_geoms[0], 0])
    mass = float(model.body_mass[layout.ball])
    if not (0.10 <= radius <= 0.12 and 0.40 <= mass <= 0.46):
        raise ValueError("unexpected scene ball geometry or mass")
    root = a.dataset_root.resolve()
    state = mujoco.MjData(model)
    rows, pins = [], []
    excluded = 0
    for item in screen["rows"]:
        events = [
            event for event in item["events"] if event["reference_ball_ankle_distance_m"] <= 0.2
        ]
        excluded += len(item["events"]) - len(events)
        if not events:
            continue
        path = root / item["source_relative_path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root / "npz" / "soccer")
            or digest(path) != item["source_hash"]
        ):
            raise ValueError("unchanged confined screened training motion required")
        with np.load(path, allow_pickle=False) as data:
            fps = int(data["fps"][0])
            n = len(data["joint_pos"])
            for event in events:
                frame, side = event["frame"], event["foot"]
                if side not in ("left", "right") or not 0 <= frame < n:
                    raise ValueError("valid screened source frame required")
                before = max(0, frame - round(0.1 * fps))
                after = min(n - 1, frame + round(0.1 * fps))
                samples = []
                for index in (before, frame, after):
                    q = np.asarray(data["body_quat_w"][index, 0], np.float64)
                    bq = np.asarray(data["object_quat_w"][index], np.float64)
                    if abs(np.linalg.norm(q) - 1) > 1e-5 or abs(np.linalg.norm(bq) - 1) > 1e-5:
                        raise ValueError("source unit root and ball quaternions required")
                    state.qpos[layout.rq : layout.rq + 3] = data["body_pos_w"][index, 0]
                    state.qpos[layout.rq + 3 : layout.rq + 7] = q
                    state.qpos[layout.qi] = data["joint_pos"][index, list(LAB_TO_MUJOCO)]
                    state.qpos[layout.bq : layout.bq + 3] = data["object_pos_w"][index]
                    state.qpos[layout.bq + 3 : layout.bq + 7] = bq
                    mujoco.mj_forward(model, state)
                    surfaces = measure_ball_surfaces(model, state, layout)["surfaces"]
                    foot = surfaces[side + "_foot"]
                    if not foot["available"]:
                        raise ValueError("actual named foot geometry required")
                    src_index = 25 if side == "left" else 26
                    link = model.body("b/" + side + "_ankle_roll_link").id
                    fk_error = float(
                        np.linalg.norm(state.xpos[link] - data["body_pos_w"][index, src_index])
                    )
                    if fk_error > 0.02:
                        raise ValueError("source/body ankle mapping differs by more than 2 cm")
                    samples.append(
                        dict(
                            frame=index,
                            foot_surface_gap_m=float(foot["gap_m"]),
                            reference_ankle_fk_error_m=fk_error,
                            ball_floor_clearance_m=float(state.qpos[layout.bq + 2] - radius),
                        )
                    )
                rows.append(
                    dict(
                        capture=item["capture"],
                        category=item["category"],
                        source_relative_path=item["source_relative_path"],
                        source_hash=item["source_hash"],
                        foot=side,
                        event_frame=frame,
                        fps=fps,
                        screening_event=event,
                        samples=samples,
                    )
                )
        if digest(path) != item["source_hash"]:
            raise ValueError("source motion changed during geometry check")
        pins.append((path, item["source_hash"]))
    if (
        not rows
        or digest(a.screen) != a.screen_hash
        or digest(a.scene) != a.scene_hash
        or any(digest(path) != expected for path, expected in pins)
    ):
        raise ValueError("unchanged nonempty source and scene required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="OMNICONTACT_REFERENCE_FOOT_GEOMETRY_NOT_PHYSICS",
            source_head=a.expected_head,
            screen_hash=a.screen_hash,
            scene_hash=a.scene_hash,
            reference_ankle_screen_max_m=0.2,
            candidate_events=len(rows),
            excluded_events=excluded,
            ball_radius_m=radius,
            ball_mass_kg=mass,
            samples=rows,
            physics_steps=0,
            sampled_source_poses_not_state_transitions=True,
            contact_labels_not_contact_forces=True,
            noncommercial_source=True,
            commercial_publicity_authorized=False,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
