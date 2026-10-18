"""Offline old/new contact-world torque replay; not a football policy exam.

The ball is deliberately moved away. Replaying recorded torques tests whether
the task-only contact change disturbs locomotion dynamics, not adaptive skill.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.task_contact_surface import with_mesh_task_contacts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh evidence directory required")
    pins = {str(p): sha(p) for p in (args.scene, args.trace, args.source_world, Path(__file__))}
    from rosclaw_soccer.physics import task_contact_surface

    helper = Path(task_contact_surface.__file__)
    pins[str(helper)] = sha(helper)
    spec = mujoco.MjSpec.from_file(str(args.scene))
    initial = spec.compile()
    if mujoco.mj_name2id(initial, mujoco.mjtObj.mjOBJ_BODY, "ball") >= 0:
        raise ValueError("this explicit diagnostic requires the original no-ball scene")
    body = spec.worldbody.add_body(name="ball", pos=(2.5, 0, 0.13))
    body.add_freejoint(name="ball_free")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.11, 0, 0),
        mass=0.43,
        friction=(0.6, 0.005, 0.0001),
    )
    if not np.any(initial.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):
        spec.worldbody.add_geom(
            name="transfer_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0, 0, 0.01),
            friction=(1, 0.005, 0.0001),
        )
    candidate, construction = with_mesh_task_contacts(
        spec,
        task_geom_name="ball_geom",
        surface_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    )
    models = [spec.compile(), candidate.compile()]
    datas = []
    for model in models:
        model.opt.timestep = 0.002
        ball = model.body("ball").id
        joint = model.body_jntadr[ball]
        bq, bv = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        model.geom_size[model.geom("ball_geom").id, 0] = 0.11
        model.body_mass[ball] = 0.43
        model.body_inertia[ball] = 0.4 * 0.43 * 0.11**2
        model.dof_damping[bv : bv + 6] = 0
        box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
        if box >= 0:
            mask = model.geom_bodyid == box
            model.geom_contype[mask] = 0
            model.geom_conaffinity[mask] = 0
        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        datas.append(data)
    args.output.mkdir(parents=True, exist_ok=False)
    for label, model in zip(("rebuilt_source", "mesh_candidate"), models, strict=True):
        mujoco.mj_saveModel(model, str(args.output / f"{label}.mjb"), None)
    if sha(args.output / "rebuilt_source.mjb") != sha(args.source_world):
        raise ValueError("reconstructed source must match the exact legacy compiled world")
    with np.load(args.trace, allow_pickle=False) as trace:
        qpos = trace["canonical_qpos"][0, 0].copy()
        qvel = trace["canonical_qvel"][0, 0].copy()
        torques = trace["torque_nm"][:, 0].copy().reshape(-1, 29)
    if qpos.shape != (43,) or qvel.shape != (41,) or torques.shape != (3000, 29):
        raise ValueError("exact six-second recorded G1 diagnostic required")
    if not all(np.isfinite(a).all() for a in (qpos, qvel, torques)):
        raise ValueError("finite original data required")
    for model, data in zip(models, datas, strict=True):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.qpos[bq : bq + 3] = (100, 100, 0.13)
        data.qvel[bv : bv + 6] = 0
        mujoco.mj_forward(model, data)
    fields = ("qpos", "qvel", "qacc", "qacc_warmstart", "qfrc_constraint", "actuator_force")
    digest = hashlib.sha256()
    minimum_height = float("inf")
    for step, torque in enumerate(torques):
        for model, data in zip(models, datas, strict=True):
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            ball_geom = int(model.geom("ball_geom").id)
            if any(
                ball_geom in c.geom
                and any(
                    model.geom_bodyid[g] != 0 and model.geom_bodyid[g] != model.body("ball").id
                    for g in c.geom
                )
                for c in data.contact[: data.ncon]
            ):
                raise ValueError("unexpected ball/player contact in ground-only diagnostic")
        for field in fields:
            values = [getattr(d, field) for d in datas]
            if not all(np.isfinite(v).all() for v in values):
                raise ValueError(f"nonfinite {field} at substep {step}")
            if values[0].tobytes() != values[1].tobytes():
                raise ValueError(f"ground parity mismatch {field} at substep {step}")
            digest.update(values[0].tobytes())
        minimum_height = min(minimum_height, float(datas[0].qpos[2]))
    if any(sha(Path(p)) != h for p, h in pins.items()):
        raise ValueError("source/input changed during diagnostic")
    result = dict(
        schema="soccer.physics.mesh_ground_parity_diagnostic.v1",
        pins=pins,
        construction=construction,
        substeps_per_world=3000,
        new_physics_steps=6000,
        worlds=2,
        checked_fields=list(fields),
        byte_equal=True,
        common_state_sha256=digest.hexdigest(),
        minimum_root_height_m=minimum_height,
        candidate_world_sha256=sha(args.output / "mesh_candidate.mjb"),
        original_world_sha256=sha(args.source_world),
        adaptive_policy_rollouts=0,
        optimizer_updates=0,
        football_skill_qualified=False,
        hardware_authorized=False,
    )
    result["report_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
