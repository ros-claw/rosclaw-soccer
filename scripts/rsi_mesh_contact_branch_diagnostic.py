"""Paired 20ms contact branches at recorded states, never a terminal policy exam."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-world", type=Path, required=True)
    parser.add_argument("--new-world", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh evidence output required")
    from rosclaw_soccer.physics import contact_wrench

    paths = (
        args.old_world,
        args.new_world,
        args.trace,
        Path(__file__),
        Path(contact_wrench.__file__),
    )
    pins = {str(p): sha(p) for p in paths}
    models = [mujoco.MjModel.from_binary_path(str(p)) for p in (args.old_world, args.new_world)]
    if any((m.nq, m.nv, m.nu) != (43, 41, 29) for m in models):
        raise ValueError("canonical G1 simulation worlds required")
    if models[0].npair != 0 or models[1].npair != 2:
        raise ValueError("explicit original zero-pair and two-pair candidate required")
    datas = [mujoco.MjData(m) for m in models]
    ai = []
    for model in models:
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        mapping = [np.flatnonzero(model.actuator_trnid[:, 0] == j) for j in joints]
        if any(len(a) != 1 for a in mapping):
            raise ValueError("unique canonical actuators required")
        ai.append([int(a[0]) for a in mapping])
    with np.load(args.trace, allow_pickle=False) as trace:
        qpos = trace["canonical_qpos"][:, 0].copy()
        qvel = trace["canonical_qvel"][:, 0].copy()
        torques = trace["torque_nm"][:, 0].copy()
    if qpos.shape != (300, 43) or qvel.shape != (300, 41) or torques.shape != (300, 10, 29):
        raise ValueError("exact original full recorded trace required")
    if not all(np.isfinite(a).all() for a in (qpos, qvel, torques)):
        raise ValueError("finite original trace required")
    rows = []
    events = []
    for frame in range(300):
        branches = []
        for label, model, data, actuators in zip(("old", "mesh"), models, datas, ai, strict=True):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = qpos[frame]
            data.qvel[:] = qvel[frame]
            data.time = frame * 0.02
            mujoco.mj_forward(model, data)
            body = model.body("ball").id
            bg = int(model.geom("ball_geom").id)
            bq = model.jnt_qposadr[model.body_jntadr[body]]
            bv = model.jnt_dofadr[model.body_jntadr[body]]
            maximum_foot_force = 0.0
            maximum_nonfoot_force = 0.0
            for substep in range(10):
                data.ctrl[actuators] = torques[frame, substep]
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("nonfinite diagnostic branch")
                for contact in read_geom_contact_wrenches(model, data, geom_id=bg):
                    other_body = int(model.geom_bodyid[contact.other_geom_id])
                    if other_body == 0:
                        continue
                    name = model.body(other_body).name
                    force = float(np.linalg.norm(contact.force_world_n))
                    foot = name in ("left_ankle_roll_link", "right_ankle_roll_link")
                    if foot:
                        maximum_foot_force = max(maximum_foot_force, force)
                    else:
                        maximum_nonfoot_force = max(maximum_nonfoot_force, force)
                    events.append(
                        dict(
                            world=label,
                            frame=frame,
                            substep=substep,
                            other_body=name,
                            other_geom=model.geom(contact.other_geom_id).name,
                            **dataclasses.asdict(contact),
                        )
                    )
            branches.append(
                dict(
                    world=label,
                    maximum_foot_force_n=maximum_foot_force,
                    maximum_nonfoot_force_n=maximum_nonfoot_force,
                    terminal_ball_position_m=data.qpos[bq : bq + 3].tolist(),
                    terminal_ball_velocity_m_s=data.qvel[bv : bv + 3].tolist(),
                )
            )
        rows.append(dict(frame=frame, branches=branches))
    if any(sha(Path(p)) != h for p, h in pins.items()):
        raise ValueError("input/source changed during diagnostic")
    result = dict(
        schema="soccer.physics.mesh_contact_branch_diagnostic.v1",
        pins=pins,
        rows=rows,
        events=events,
        new_physics_steps=6000,
        paired_recorded_states=300,
        independent_courses=0,
        branch_substeps=10,
        warmstart="reset per branch in both worlds",
        future_foundation_recomputed=False,
        optimizer_updates=0,
        terminal_task_success_inferred=False,
        football_skill_qualified=False,
        cached_contact_before_integrated_qpos=True,
        hardware_authorized=False,
    )
    result["report_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(report_sha256=result["report_sha256"], events=len(events), new_physics_steps=6000)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
