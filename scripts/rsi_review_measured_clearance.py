"""Actual-state NN547, measured geometry and exact target/history codec readback."""

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace


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
    from scripts.rsi_review_joint_response_pilot import verify_target_edits
    from scripts.rsi_teacher_physics_context import load_walk_kick_context

    clean(a.expected_head)
    complete = a.qualification / "complete.json"
    if a.output.exists() or complete.is_symlink() or digest(complete) != a.qualification_hash:
        raise ValueError("unused output and exact closed qualification required")
    report = json.loads(complete.read_text())
    if report["status"] != "MEASURED_CLEARANCE_G1_ACTUAL_NN_FUTURE_EXACT":
        raise ValueError("closed healthy measured-feedback parity required")
    pins = dict(report["input_file_hashes"])
    pins[str(complete)] = a.qualification_hash
    # Early producer versions omitted these JSON commitments. Pin here, then
    # reconstruct their contents from producer-bound actual arrays and NN.
    newly_pinned_receipts = []
    for name in ("actual-edits.json", "future.json", "declaration.json"):
        path = a.qualification / name
        if path.is_symlink():
            raise ValueError("ordinary raw receipt required")
        if str(path) not in pins:
            newly_pinned_receipts.append(name)
            pins[str(path)] = digest(path)
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            qualification_hash=a.qualification_hash,
            input_file_hashes=pins,
            maximum_original_nn_readbacks=128,
            original_constructor_warmups_source_bound=5,
            total_new_nn_call_ceiling=133,
            new_physical_steps=0,
            optimizer_updates=0,
            receipts_first_pinned_at_reader=newly_pinned_receipts,
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
    from rosclaw_soccer.rsi.ball_clearance_control import clearance_offset
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    context = load_walk_kick_context(pins)
    model, lo = context["model"], context["layout"]
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("same qualified native body required")

    def raw(name):
        path = a.qualification / name
        if pins.get(str(path)) != digest(path) or path.is_symlink():
            raise ValueError("producer-bound raw arrays required")
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k].copy() for k in f.files}

    actual, future = raw("actual-unvalidated.npz"), raw("future.npz")
    row = report["actual_record"]
    entry = row["entry_control"]
    edits = json.loads((a.qualification / "actual-edits.json").read_text())["edits"][:128]
    if (
        len(edits) != 128
        or row["native_steps"] != 3600
        or hash_json(relabel_arrays(actual, controls=360, ball_address=lo.bq, root_address=lo.rq))
        != hash_json(row["outcome"])
    ):
        raise ValueError("full actual outcome and128 target receipts required")
    if any(not np.isfinite(v).all() for v in actual.values()):
        raise ValueError("finite actual evidence required")
    if (abs(actual["torque"][:, lo.ai]) > 0.85 * context["hard_limits"] + 1e-12).any():
        raise ValueError("actual original hard torque caps exceeded")
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        if not np.array_equal(future[key], actual[key][entry * 10 : (entry + 128) * 10]):
            raise ValueError("future differs fromactual " + key)
    if not np.array_equal(future["joint_targets"], actual["joint_targets"][entry : entry + 128]):
        raise ValueError("prospective/actual target mismatch")
    d, geometry = mujoco.MjData(model), mujoco.MjData(model)
    torso = int(model.body("a/torso_link").id)

    def observe(tick):
        d.qpos[:], d.qvel[:] = actual["qpos"][tick * 10 - 1], actual["qvel"][tick * 10 - 1]
        q, v = lo.canonical(d)
        d.qpos[:] = actual["qpos"][tick * 10 - 2]
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        return q, v

    samples = tuple(
        KickMeasuredSample.from_numeric_record(np.array(r, np.float64))
        for r in row["entry_state"]["measured_history"]
    )
    if tuple(s.frame for s in samples) != tuple(range(entry - 4, entry + 1)):
        raise ValueError("actual consecutive entry history required")
    for s in samples:
        q, v = observe(s.frame)
        expected = KickMeasuredSample(
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
        if s != expected or abs(s.time_sec - 0.02 * s.frame) > 1e-8:
            raise ValueError("measured entry does not match actual raw")
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
        raise ValueError("actual original measured-history receipt differs")
    previous = np.zeros(29)
    offsets = []
    ballgeom = int(np.flatnonzero(model.geom_bodyid == lo.ball)[0])
    anticipatory = json.loads((a.qualification / "declaration.json").read_text()).get(
        "anticipatory_velocity", False
    )
    for local, edit in enumerate(edits):
        tick = entry + local
        policy_observe(tick)
        obs = policy._build_obs()
        out = policy.ort_session.run(["actions"], {"obs": obs[None]})[0][0]
        clipped = np.clip(out, policy.action_clip_lo_il, policy.action_clip_hi_il)
        if not np.array_equal(clipped, np.array(edit["original_raw_action_il"], np.float32)):
            raise ValueError("actual-state original547 NN/action history differs")
        actual_q = actual["qpos"][tick * 10 - 1]
        projected = actual_q.copy()
        if anticipatory:
            mujoco.mj_integratePos(model, projected, actual["qvel"][tick * 10 - 1], 0.16)
        distances, risks, grads = np.ones(2), np.ones(2), np.zeros((2, 12))
        for side, name in enumerate(("left", "right")):
            group = np.flatnonzero(model.geom_bodyid == model.body("a/" + name + "_knee_link").id)
            options = []
            for pose in (actual_q, projected) if anticipatory else (actual_q,):
                geometry.qpos[:] = pose
                mujoco.mj_kinematics(model, geometry)
                geom = int(
                    min(
                        group,
                        key=lambda g: mujoco.mj_geomDistance(
                            model, geometry, ballgeom, int(g), 1.0, None
                        ),
                    )
                )
                options.append(
                    (mujoco.mj_geomDistance(model, geometry, ballgeom, geom, 1.0, None), geom, pose)
                )
            distances[side] = options[0][0]
            distance, geom, pose = min(options, key=lambda x: x[0])
            risks[side] = distance
            if distance >= 0.08:
                continue
            for axis, qi in enumerate(lo.qi[:12]):
                values = []
                for sign in (-1, 1):
                    geometry.qpos[:] = pose
                    geometry.qpos[qi] += sign * 1e-4
                    mujoco.mj_kinematics(model, geometry)
                    values.append(
                        mujoco.mj_geomDistance(model, geometry, ballgeom, geom, 1.0, None)
                    )
                grads[side, axis] = (values[1] - values[0]) / 2e-4
        delta = clearance_offset(risks, grads, previous)
        if local >= 120:
            delta = previous + np.clip(-previous, -0.008, 0.008)
        for recorded, expected in (
            (edit["distances_m"], distances),
            (edit["gradient_hg12"], grads),
            (edit["requested_offset_hg_rad"], delta),
            (edit["previous_requested_offset"], previous),
        ):
            if not np.array_equal(np.asarray(recorded), expected):
                raise ValueError("actual geometry/gradient/request feedback differs")
        if "decision_distances_m" in edit and not np.array_equal(
            np.asarray(edit["decision_distances_m"]), risks
        ):
            raise ValueError("kinematic risk differs")
        offsets.append(delta.copy())
        previous = delta.copy()
        policy.last_action_il = np.array(
            edit.get("executed_raw_action_il", edit["original_raw_action_il"]), np.float32
        )
        policy.time_step += 1
    codec = (
        policy.default_q_mj,
        policy.action_scale_mj,
        mapping,
        policy.action_clip_lo_il,
        policy.action_clip_hi_il,
    )
    plan = SimpleNamespace(
        start_control=entry, projection_mode="per_axis", offsets=lambda: np.array(offsets)
    )
    verify_target_edits(
        edits,
        actual["joint_targets"][entry : entry + 128],
        start=entry,
        plan=plan,
        codec=codec,
        limits=model.jnt_range[lo.joints],
    )
    if previous.any() or d.time != 0 or geometry.time != 0:
        raise ValueError("end fade or zero physical advancement violated")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="ACTUAL_MEASURED_CLEARANCE_NN547_GEOMETRY_CODEC_VERIFIED",
            source_head=a.expected_head,
            qualification_hash=a.qualification_hash,
            input_file_hashes=pins,
            measured_entry_samples=5,
            actual_nn547_readbacks=128,
            actual_geometry_and_codec_reconstructions=128,
            new_nn_calls=128,
            additional_constructor_warmups_source_bound=5,
            new_physical_steps=0,
            optimizer_updates=0,
            receipts_first_pinned_at_reader=newly_pinned_receipts,
            original_math_codecs_reused=True,
            task_effectiveness_proven=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
