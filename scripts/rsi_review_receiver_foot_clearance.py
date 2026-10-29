"""Measured 50 Hz ball-foot geometry on replayed receiver trajectories."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external evidence output required")
    source, _, source_hash = report(
        a.source,
        a.audit,
        a.audit_hash,
        "NEURAL_RELAY_CHILD_VELOCITY_CANDIDATE_SAMPLING32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    scene = a.source / "scene.xml"
    scene_hash = digest(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    data = mujoco.MjData(model)
    rows = []
    for row in source["records"]:
        if row["course"]["arm"]["exploration_seed"] is None:
            continue
        path = bind(a.source, row, "physics")
        if path is None:
            raise ValueError("actual own-history physics required")
        with np.load(path, allow_pickle=False) as z:
            force = z["ball_forces"].reshape(-1, 2, 3)
            a_foot = np.flatnonzero((force[:, 0, :2] > 10).any(axis=1))
            b_foot = np.flatnonzero((force[:, 1, :2] > 10).any(axis=1))
            if not a_foot.size:
                rows.append(dict(index=row["index"], passer_foot_observed=False))
                continue
            best = None
            start = int(a_foot[0]) // 10
            for control in range(start, len(z["qpos"])):
                data.qpos[:] = z["qpos"][control, -1]
                data.qvel[:] = z["qvel"][control, -1]
                data.time = float(z["time_before"][control] + 0.02)
                surfaces = measure_ball_surfaces(model, data, layout)["surfaces"]
                for side in ("left", "right"):
                    surface = surfaces[side + "_foot"]
                    if surface["available"]:
                        candidate = (
                            float(surface["gap_m"]),
                            control,
                            side,
                            bool(surface["smooth_separated"]),
                        )
                        if best is None or candidate[0] < best[0]:
                            best = candidate
            rows.append(
                dict(
                    index=row["index"],
                    course=row["course"],
                    termination=row["termination"],
                    passer_first_foot_substep=int(a_foot[0]),
                    receiver_first_foot_substep=int(b_foot[0]) if b_foot.size else None,
                    nearest_sampled_foot_gap_m=best[0] if best else None,
                    nearest_sampled_control=best[1] if best else None,
                    nearest_sampled_foot=best[2] if best else None,
                    nearest_sampled_gap_smooth=best[3] if best else None,
                    physics_hash=row["artifacts"]["physics"]["hash"],
                )
            )
    if len(rows) != 24 or digest(scene) != scene_hash:
        raise ValueError("all sampled cases and unchanged scene required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="REPLAYED_50HZ_FOOT_GEOMETRY_DEVELOPMENT_ONLY",
            source_hash=source_hash,
            audit_hash=a.audit_hash,
            scene_hash=scene_hash,
            code_head=a.expected_head,
            records=rows,
            causal_opportunity_or_reachability_proven=False,
            physical_improvement_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
