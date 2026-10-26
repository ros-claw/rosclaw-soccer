"""Actual NN547/reference readback and causal foot-tracking diagnostic, no stepping."""

import argparse
import contextlib
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--qualification-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_teacher_physics_context import load_walk_kick_context

    clean(a.expected_head)
    complete = a.qualification / "complete.json"
    if a.output.exists() or complete.is_symlink() or digest(complete) != a.qualification_hash:
        raise ValueError("unused output and exact closed native reference qualification required")
    report = json.loads(complete.read_text())
    if report["status"] != "CONTACT_REFERENCE_NATIVE_NN_EXACT":
        raise ValueError("full healthy actual/reference interface qualification required")
    pins = dict(report["input_file_hashes"])
    pins[str(complete)] = a.qualification_hash
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_new_nn_calls=128,
            additional_original_constructor_warmups_source_bound=5,
            total_new_nn_ceiling_including_warmups=133,
            new_physical_steps=0,
            optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.kick_measured_history import (
        KickMeasuredSample,
        install_measured_kick_history,
    )
    from rosclaw_soccer.providers.g1.kick_warmstart import rebase_kick_reference
    from rosclaw_soccer.rsi.absolute_contact_reference import (
        AbsoluteContactReferencePlan,
        condition_absolute_contact_reference,
    )
    from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
    from rosclaw_soccer.rsi.contact_reference_ik import (
        first_contact_foot,
        solve_contact_reference_pose,
    )
    from rosclaw_soccer.rsi.contact_reference_plan import (
        ContactReferencePlan,
        condition_contact_reference,
    )
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    context = load_walk_kick_context(pins)
    model, lo = context["model"], context["layout"]

    def raw(name):
        path = a.qualification / name
        if path.is_symlink() or pins.get(str(path)) != digest(path):
            raise ValueError("exact producer-bound raw required")
        with np.load(path, allow_pickle=False) as saved:
            data = {k: saved[k].copy() for k in saved.files}
        if any(not np.isfinite(v).all() for v in data.values()):
            raise ValueError("finite raw required")
        return data

    actual, parent, future = (
        raw("actual-unvalidated.npz"),
        raw("parent-future.npz"),
        raw("contact-future.npz"),
    )
    row = report["actual_record"]
    entry = row["entry_control"]
    if (
        row["native_steps"] != 3600
        or row["controls"] != 360
        or hash_json(relabel_arrays(actual, controls=360, ball_address=lo.bq, root_address=lo.rq))
        != hash_json(row["outcome"])
    ):
        raise ValueError("whole actual original H300 score differs")
    if (abs(actual["torque"][:, lo.ai]) > 0.85 * context["hard_limits"] + 1e-12).any():
        raise ValueError("hard actual torque caps violated")
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        if not np.array_equal(future[key], actual[key][entry * 10 : (entry + 128) * 10]):
            raise ValueError("full conditioned future differs fromactual")
    if not np.array_equal(future["joint_targets"], actual["joint_targets"][entry : entry + 128]):
        raise ValueError("conditioned target future differs fromactual")
    contact = first_contact_foot(parent["ball_forces"][:, 0, :2])
    if contact is None:
        raise ValueError("parent contact seed absent")
    foot, first_native = contact
    peak = first_native // 10 - 1
    if not 1 <= peak < 48:
        raise ValueError("current-own parent IK seed out ofrange")
    seed = mujoco.MjData(model)
    seed.qpos[:], seed.qvel[:] = parent["qpos"][peak * 10 - 1], parent["qvel"][peak * 10 - 1]
    seed.time = 0.02 * (entry + peak)
    ik = solve_contact_reference_pose(
        model, seed, lo, goal=np.array([8.0, 0.0, 0.11]), active_foot=foot
    )
    if hash_json(ik) != hash_json(report["generation"]["ik"]):
        raise ValueError("reference IK does not regenerate fromown parent raw")
    d = mujoco.MjData(model)
    torso = int(model.body("a/torso_link").id)

    def observe(tick):
        d.qpos[:], d.qvel[:] = actual["qpos"][tick * 10 - 1], actual["qvel"][tick * 10 - 1]
        q, v = lo.canonical(d)
        d.qpos[:] = actual["qpos"][tick * 10 - 2]
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        return q, v

    samples = tuple(
        KickMeasuredSample.from_numeric_record(np.array(s, np.float64))
        for s in row["entry_state"]["measured_history"]
    )
    if tuple(s.frame for s in samples) != tuple(range(entry - 4, entry + 1)):
        raise ValueError("actual consecutive five-frame history required")
    for s in samples:
        q, v = observe(s.frame)
        rebuilt = KickMeasuredSample(
            s.frame,
            s.time_sec,
            s.frame - 1,
            tuple(map(float, q[7:36])),
            tuple(map(float, v[6:35])),
            tuple(map(float, v[3:6])),
            tuple(map(float, q[:3])),
            tuple(map(float, q[3:7])),
            tuple(map(float, d.xpos[lo.ball])),
            tuple(map(float, actual["joint_targets"][s.frame - 1])),
        )
        if s != rebuilt:
            raise ValueError("entry history does not match actual source")
    st, ot, pt, mapping = context["types"]
    with contextlib.redirect_stdout(io.StringIO()):
        policy = pt(st(29), ot(29))

    def policy_observe(tick):
        q, v = observe(tick)
        s = policy.state_cmd
        s.q, s.dq = q[7:36].copy(), v[6:35].copy()
        s.tau_est = actual["torque"][tick * 10 - 1, lo.ai].copy()
        s.root_lin_vel_b, s.root_ang_vel_b = v[:3].copy(), v[3:6].copy()
        s.torso_pos_w, s.torso_quat_w = d.xpos[torso].copy(), d.xquat[torso].copy()
        s.pelvis_pos_w, s.pelvis_quat_w = q[:3].copy(), q[3:7].copy()
        s.ball_pos_w, s.ball_vel_w = d.xpos[lo.ball].copy(), v[35:38].copy()
        s.ball_valid = True

    policy_observe(entry)
    policy.target_pos_w = np.array([8.0, 0.0, 0.11], np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        policy.enter()
    receipt = install_measured_kick_history(
        policy, samples, frame=entry, time_sec=samples[-1].time_sec
    )
    rebase_kick_reference(policy, entry_frame=row["admitted_reference_phase"])
    policy.time_step = int(policy.WARMUP_STEPS) + row["admitted_reference_phase"]
    if receipt != row["measured_history_receipt"]:
        raise ValueError("measured history receipt differs")
    delta = np.array(ik["reference_joints_hg"]) - seed.qpos[lo.qi]
    absolute = report.get("reference_contract") == "absolute_contact_waypoint_v1"
    plan = (
        AbsoluteContactReferencePlan(
            row["admitted_reference_phase"],
            tuple(int(v) for v in mapping),
            tuple(float(v) for v in ik["reference_joints_hg"]),
            peak,
        )
        if absolute
        else ContactReferencePlan(
            row["admitted_reference_phase"],
            tuple(int(v) for v in mapping),
            tuple(float(v) for v in delta),
            peak,
        )
    )
    old_reference_endpoint = (
        policy.motion_joint_pos[plan.start_frame + peak, mapping].astype(np.float64).copy()
    )
    condition = condition_absolute_contact_reference if absolute else condition_contact_reference
    installed = condition(policy, plan, limits_hg=model.jnt_range[lo.joints].astype(np.float64))
    if hash_json(installed) != hash_json(report["installation"]):
        raise ValueError("exact reference generation/conditioning receipt differs")
    endpoint = policy.motion_joint_pos[plan.start_frame + peak, mapping].astype(np.float64).copy()
    for tick in range(entry, entry + 128):
        policy_observe(tick)
        obs = policy._build_obs()
        out = policy.ort_session.run(["actions"], {"obs": obs[None]})[0][0]
        clipped = np.clip(out, policy.action_clip_lo_il, policy.action_clip_hi_il)
        target = (policy.default_q_mj + policy.action_scale_mj * clipped[mapping]).astype(
            np.float64
        )
        if not np.array_equal(target, actual["joint_targets"][tick]):
            raise ValueError("actual547/history reference or original NN decoded target differs")
        policy.last_action_il = clipped.copy()
        policy.time_step += 1
    # Diagnose three different configurations; these are FK comparisons only.
    active_body = int(model.body("a/" + foot + "_ankle_roll_link").id)
    support = "right" if foot == "left" else "left"
    support_body = int(model.body("a/" + support + "_ankle_roll_link").id)
    goal = np.array(ik["contact_goal_world"])
    local = np.array(ik["contact_point_local"])
    measured_joint = actual["qpos"][(entry + peak) * 10 - 1, lo.qi]
    tracking = dict(
        reference_endpoint_error_to_ik_leg12_max_rad=float(
            abs(endpoint[:12] - np.array(ik["reference_joints_hg"])[:12]).max()
        ),
        nominal_reference_tracking_bias_leg12_max_rad=float(
            abs(old_reference_endpoint[:12] - seed.qpos[lo.qi][:12]).max()
        ),
        reference_endpoint_error_to_ik_max_rad=float(
            abs(endpoint - np.array(ik["reference_joints_hg"])).max()
        ),
        nominal_reference_tracking_bias_max_rad=float(
            abs(old_reference_endpoint - seed.qpos[lo.qi]).max()
        ),
        actual_endpoint_error_to_ik_max_rad=float(
            abs(measured_joint - np.array(ik["reference_joints_hg"])).max()
        ),
        planned_peak_control=entry + peak,
    )
    for name, qpos in (
        ("ik_configuration", seed.qpos.copy()),
        ("reference_joints_on_parent_root", seed.qpos.copy()),
        ("actual_at_peak", actual["qpos"][(entry + peak) * 10 - 1].copy()),
    ):
        geom = mujoco.MjData(model)
        geom.qpos[:] = qpos
        if name == "ik_configuration":
            geom.qpos[lo.qi] = np.array(ik["reference_joints_hg"])
        elif name == "reference_joints_on_parent_root":
            geom.qpos[lo.qi] = endpoint
        mujoco.mj_kinematics(model, geom)
        mujoco.mj_comPos(model, geom)
        point = geom.xpos[active_body] + geom.xmat[active_body].reshape(3, 3) @ local
        surfaces = measure_ball_surfaces(model, geom, lo)["surfaces"]
        tracking[name] = dict(
            point_error_to_planned_world_goal_m=float(np.linalg.norm(point - goal)),
            minimum_shin_gap_m=min(surfaces[side + "_shin"]["gap_m"] for side in ("left", "right")),
            support_position_world=geom.xpos[support_body].tolist(),
            foot_surface_gap_m=surfaces[foot + "_foot"]["gap_m"],
        )
    actual_contact = first_contact_foot(actual["ball_forces"][:, 0, :2])
    nf = np.flatnonzero(actual["ball_forces"][:, 0, 2] > 1)
    tracking["actual_first_foot"] = actual_contact
    tracking["actual_first_nonfoot_control"] = None if not len(nf) else int(nf[0] // 10)
    if d.time != 0:
        raise ValueError("readback unexpectedly advanced physical time")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CONTACT_REFERENCE_ACTUAL_NN547_AND_GENERATION_VERIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            qualification_hash=a.qualification_hash,
            measured_history_samples=5,
            actual_nn547_readbacks=128,
            new_neural_calls=128,
            additional_constructor_warmups_source_bound=5,
            new_physical_steps=0,
            optimizer_updates=0,
            original_math_and_ik_solver_reused=True,
            tracking_diagnostic=tracking,
            task_outcome=row["outcome"],
            task_effectiveness_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
