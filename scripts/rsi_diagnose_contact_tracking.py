"""Decompose an audited native reference failure using FK only, no new NN/steps."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_teacher_physics_context import load_walk_kick_context

    clean(a.expected_head)
    if a.output.exists() or a.audit.is_symlink() or digest(a.audit) != a.audit_hash:
        raise ValueError("unused output and committed closed independent readback required")
    audit = json.loads(a.audit.read_text())
    complete = a.qualification / "complete.json"
    if (
        audit["status"] != "CONTACT_REFERENCE_ACTUAL_NN547_AND_GENERATION_VERIFIED"
        or complete.is_symlink()
        or digest(complete) != audit["qualification_hash"]
    ):
        raise ValueError("exact independently verified producer required")
    report = json.loads(complete.read_text())
    pins = dict(audit["input_file_hashes"])
    pins[str(a.audit)] = a.audit_hash
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces

    context = load_walk_kick_context(pins)
    model, lo = context["model"], context["layout"]

    def raw(name):
        path = a.qualification / name
        if path.is_symlink() or pins.get(str(path)) != digest(path):
            raise ValueError("producer-bound ordinary raw required")
        with np.load(path, allow_pickle=False) as saved:
            return {k: saved[k].copy() for k in saved.files}

    actual, parent = raw("actual-unvalidated.npz"), raw("parent-future.npz")
    entry = report["actual_record"]["entry_control"]
    peak = report["generation"]["peak_control"]
    ik = report["generation"]["ik"]
    waypoint = np.array(ik["reference_joints_hg"])
    before = parent["qpos"][peak * 10 - 1]
    measured = actual["qpos"][(entry + peak) * 10 - 1]
    foot = report["generation"]["contact_foot"]
    body = int(model.body("a/" + foot + "_ankle_roll_link").id)
    local, target = np.array(ik["contact_point_local"]), np.array(ik["contact_goal_world"])
    combinations = {}
    for name, base, joints in (
        ("parent_root_ik_joints", before, waypoint),
        ("parent_root_actual_joints", before, measured[lo.qi]),
        ("actual_root_ik_joints", measured, waypoint),
        ("actual_root_actual_joints", measured, measured[lo.qi]),
    ):
        d = mujoco.MjData(model)
        d.qpos[:] = base
        d.qpos[lo.qi] = joints
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        point = d.xpos[body] + d.xmat[body].reshape(3, 3) @ local
        combinations[name] = dict(
            foot_point_world=point.tolist(),
            error_to_planned_point_m=float(np.linalg.norm(point - target)),
        )
    # Distances are not additive causal percentages. This is a factorial FK
    # decomposition, not evidence a counterfactual body trajectory is feasible.
    forces = actual["ball_forces"][:, 0]
    nf = np.flatnonzero(forces[:, 2] > 1)
    contacts = []
    for label, index in (
        ("first_nonfoot", int(nf[0]) if len(nf) else None),
        ("peak_nonfoot", int(forces[:, 2].argmax())),
    ):
        if index is None:
            continue
        d = mujoco.MjData(model)
        d.qpos[:], d.qvel[:] = actual["qpos"][index], actual["qvel"][index]
        surfaces = measure_ball_surfaces(model, d, lo)["surfaces"]
        contacts.append(
            dict(
                event=label,
                native_index=index,
                control=index // 10,
                ball_forces_n=forces[index].tolist(),
                post_step_surface_gaps_m={k: v.get("gap_m") for k, v in surfaces.items()},
                force_is_original_raw_not_recomputed=True,
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="AUDITED_CONTACT_TRACKING_FK_DECOMPOSITION",
            source_head=a.expected_head,
            input_file_hashes=pins,
            qualification_hash=audit["qualification_hash"],
            audit_hash=a.audit_hash,
            combinations=combinations,
            root_translation_change_m=(
                measured[lo.rq : lo.rq + 3] - before[lo.rq : lo.rq + 3]
            ).tolist(),
            actual_minus_ik_leg12_rad=(measured[lo.qi][:12] - waypoint[:12]).tolist(),
            contacts=contacts,
            decomposition_distances_are_not_additive=True,
            counterfactual_root_joint_combinations_are_not_dynamic_proofs=True,
            new_neural_calls=0,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
