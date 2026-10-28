"""Inspect post-step foot-ball geometry at measured nonfoot contact in SIM."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    source_root, audit = a.root / "source", a.root / "audit/complete.json"
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external analysis output required")
    ah = digest(audit)
    source, _, sh = report(
        source_root, audit, ah, "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY", worlds=16
    )
    if source["source_head"] != a.source_head:
        raise ValueError("exact independently replayed physical source required")
    scene = source_root / "scene.xml"
    scene_hash = digest(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    ball_geoms = np.flatnonzero(model.geom_bodyid == layout.ball)
    feet = [
        i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            ("b/left_foot", "b/right_foot")
        )
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_collision")
    ]
    if len(ball_geoms) != 1 or len(feet) != 14:
        raise ValueError("exact single sphere and fourteen physical foot geoms required")
    rows = []
    for row in source["records"]:
        path = bind(source_root, row, "physics")
        if path is None:
            raise ValueError("all original physical records required")
        with np.load(path, allow_pickle=False) as f:
            forces = f["ball_forces"].reshape(-1, 2, 3)
            nonfoot = np.flatnonzero(forces[:, 1, 2] > 1)
            foot = np.flatnonzero(np.max(forces[:, 1, :2], axis=1) > 10)
            if len(nonfoot):
                first = int(nonfoot[0])
                states = f["qpos"].reshape(-1, model.nq)
                if first >= len(states) or not np.isfinite(states[first]).all():
                    raise ValueError("measured finite post-step state required")
                data.qpos[:] = states[first]
                mujoco.mj_forward(model, data)
                distances = [
                    float(mujoco.mj_geomDistance(model, data, int(ball_geoms[0]), i, 4.0, None))
                    for i in feet
                ]
                min_dist = min(distances)
                if not np.isfinite(distances).all():
                    raise ValueError("finite physical geometry distances required")
                nearest = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, feet[int(np.argmin(distances))]
                )
            else:
                first = min_dist = nearest = None
            rows.append(
                dict(
                    context=row["course"]["context"],
                    side=row["course"]["lateral_m"],
                    learned=row["course"]["arm"]["receiver_learned"],
                    first_nonfoot_substep=first,
                    first_foot_substep=int(foot[0]) if len(foot) else None,
                    foot_ball_signed_distance_after_nonfoot_m=min_dist,
                    nearest_foot_geom=nearest,
                    body_failure=row["termination"] == "BODY_FAILURE",
                )
            )
    if (
        len(rows) != 16
        or digest(source_root / "complete.json") != sh
        or digest(audit) != ah
        or digest(scene) != scene_hash
    ):
        raise ValueError("all pinned pilot worlds and unchanged scene required")
    summary = {}
    for learned in (False, True):
        group = [r for r in rows if r["learned"] is learned]
        distances = [
            r["foot_ball_signed_distance_after_nonfoot_m"]
            for r in group
            if r["foot_ball_signed_distance_after_nonfoot_m"] is not None
        ]
        summary["candidate" if learned else "baseline"] = dict(
            worlds=len(group),
            nonfoot_worlds=len(distances),
            median_foot_ball_signed_distance_after_nonfoot_m=float(np.median(distances))
            if distances
            else None,
            nonfoot_before_foot=sum(
                r["first_nonfoot_substep"] is not None
                and (
                    r["first_foot_substep"] is None
                    or r["first_nonfoot_substep"] < r["first_foot_substep"]
                )
                for r in group
            ),
        )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_NONFOOT_POSTSTEP_GEOMETRY_DIAGNOSED",
            source_head=a.expected_head,
            physical_source_hash=sh,
            audit_hash=ah,
            scene_hash=scene_hash,
            summary=summary,
            records=rows,
            geometry_is_poststep_not_exact_contact_instant=True,
            no_counterfactual_or_success_claim=True,
            independent_fresh_contexts=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
