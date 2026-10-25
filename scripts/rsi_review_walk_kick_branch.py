"""Independent raw/body-history and original NN547 readback; no physics replay.

Reuses original codecs and five source-bound constructor warmup calls. Executes
one velocity and128 kick NN readbacks on reconstructed ACTUAL observations.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification", "reference-record", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--qualification-hash", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_teacher_physics_context import load_walk_kick_context

    clean(a.expected_head)
    cp = a.qualification / "complete.json"
    if a.output.exists() or cp.is_symlink() or digest(cp) != a.qualification_hash:
        raise ValueError("unused audit and exact completed composed qualification required")
    report = json.loads(cp.read_text())
    if (
        report["status"] != "CURRENT_WALK_TO_KICK_NN_AND_NATIVE_FUTURE_EXACT"
        or report["actual_worlds"] != 1
        or report["actual_native_steps"] != 3600
        or report["future_admission_control"] != 77
        or report["source_prefix_controls_not_new"] != 76
        or report["original_whole_actual_trajectory_unchanged"] is not True
        or report["future_measured_entry_receipt_matches_original"] is not True
        or report["neural_optimizer_updates"] != 0
        or any(
            report[k] is not False
            for k in (
                "task_effectiveness_proven",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
        or report["forecasts"]
        != dict(
            forecast_calls=1,
            attempted_native_steps=1290,
            integrated_native_steps=1290,
            maximum_native_steps=1290,
            future_velocity_nn_calls=1,
            future_kick_nn_calls=128,
            attempted_future_kick_nn_calls=128,
        )
    ):
        raise ValueError("one current walk-to-kick component qualification required, not a teacher")
    pins = dict(report["input_file_hashes"])
    pins[str(cp)] = a.qualification_hash
    dp = a.qualification / "declaration.json"
    if digest(dp) != report["declaration_hash"]:
        raise ValueError("original before-physics declaration changed")
    pins[str(dp)] = report["declaration_hash"]
    declaration = json.loads(dp.read_text())
    if pins.get(str(a.reference_record)) != digest(a.reference_record):
        raise ValueError("exact original comparison record required")
    reference = json.loads(a.reference_record.read_text())
    if reference["arm"] != "onnx" or reference["entry_control"] != 77:
        raise ValueError("same original actual admission77 required")
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.walk_kick_original_nn_readback.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            qualification_hash=a.qualification_hash,
            new_physical_step_ceiling=0,
            maximum_velocity_inference_readbacks=1,
            maximum_kick_inference_readbacks=128,
            kick_constructor_warmup_calls_from_pinned_source=5,
            total_nn_call_ceiling_including_source_bound_warmups=134,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )

    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.kick_measured_history import (
        KickMeasuredSample,
        install_measured_kick_history,
    )
    from rosclaw_soccer.providers.g1.kick_warmstart import rebase_kick_reference
    from rosclaw_soccer.providers.g1.velocity_policy import (
        G1VelocityPolicy,
        observation98,
        phase_delta,
        target29,
    )
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    context = load_walk_kick_context(pins)
    model, lo = context["model"], context["layout"]
    if compiled_model_hash(model) != declaration["model_hash"]:
        raise ValueError("same compiled native body required")

    def raw(path):
        if path.is_symlink() or pins.get(str(path)) != digest(path):
            raise ValueError("saved qualified actual/future raw data changed")
        with np.load(path, allow_pickle=False) as data:
            result = {k: data[k].copy() for k in data.files}
        if any(not np.isfinite(v).all() for v in result.values()):
            raise ValueError("nonfinite raw trajectory or neural record")
        return result

    actual = raw(a.qualification / "actual-unvalidated.npz")
    future = raw(a.qualification / "future.npz")
    original = raw(a.reference_record.parent / reference["artifact"]["path"])
    fp = a.qualification / "future.json"
    if pins.get(str(fp)) != digest(fp):
        raise ValueError("exact prospective future metadata required")
    meta = json.loads(fp.read_text())
    row = report["actual_record"]
    root = actual["qpos"][:, lo.rq : lo.rq + 7]
    if (
        set(actual) != set(original)
        or any(not np.array_equal(v, original[k]) for k, v in actual.items())
        or row["controls"] != 360
        or row["native_steps"] != 3600
        or row["healthy_full_horizon"] is not True
        or (root[:, 2] < 0.65).any()
        or (1 - 2 * (root[:, 4] ** 2 + root[:, 5] ** 2) < 0.9).any()
        or (abs(actual["torque"][:, lo.ai]) > 0.85 * context["hard_limits"] + 1e-12).any()
        or hash_json(relabel_arrays(actual, controls=360, ball_address=lo.bq, root_address=lo.rq))
        != hash_json(row["outcome"])
        or any(
            meta[k] != v
            for k, v in dict(
                controls=129,
                native_steps=1290,
                requested_controls=129,
                donor_control_frame=76,
                admission_control=77,
                source_prefix_controls_not_new=76,
                body_failed=False,
                admission_rejection=None,
            ).items()
        )
    ):
        raise ValueError("original whole task/body or consumed future metadata differs")
    for k in ("qpos", "qvel", "torque", "ball_forces"):
        if not np.array_equal(future[k], actual[k][760:2050]):
            raise ValueError("composed native future differs from actual: " + k)
    for k in ("joint_targets", "commands"):
        if not np.array_equal(future[k], actual[k][76:205]):
            raise ValueError("composed actual neural target/command differs: " + k)
    if (
        future["velocity_observations"].shape != (1, 98)
        or future["kick_observations"].shape != (128, 547)
        or future["kick_raw_actions"].shape != (128, 29)
        or future["kick_clipped_actions"].shape != (128, 29)
        or not np.array_equal(future["policy_ticks"], np.arange(76, 205))
    ):
        raise ValueError(
            "all separate true neural dimensions and continuous absolute clocks required"
        )

    d = mujoco.MjData(model)
    torso = int(model.body("a/torso_link").id)

    def observe(tick):
        d.qpos[:], d.qvel[:] = actual["qpos"][tick * 10 - 1], actual["qvel"][tick * 10 - 1]
        q, v = lo.canonical(d)
        # Preserve original Euler observation semantics: cached PRE-last-step
        # geometry alongside latest measured q/v; no mj_forward on actual state.
        d.qpos[:] = actual["qpos"][tick * 10 - 2]
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        return q, v

    samples = tuple(
        KickMeasuredSample.from_numeric_record(r) for r in future["measured_entry_samples"]
    )
    if len(samples) != 5 or tuple(s.frame for s in samples) != tuple(range(73, 78)):
        raise ValueError("five actual consecutive absolute entry samples required")
    for sample in samples:
        q, v = observe(sample.frame)
        reconstructed = KickMeasuredSample(
            sample.frame,
            sample.time_sec,
            sample.frame - 1,
            tuple(map(float, q[7:36])),
            tuple(map(float, v[6:35])),
            tuple(map(float, v[3:6])),
            tuple(map(float, q[:3])),
            tuple(map(float, q[3:7])),
            tuple(map(float, d.xpos[lo.ball])),
            tuple(map(float, actual["joint_targets"][sample.frame - 1])),
        )
        if sample != reconstructed or abs(sample.time_sec - 0.02 * sample.frame) > 1e-8:
            raise ValueError("entry sample does not match the actual completed body/target")
    if not np.array_equal(
        future["measured_entry_samples"], np.array(row["entry_state"]["measured_history"])
    ):
        raise ValueError("future entry history differs from actual original measured history")
    assets = context["assets"]
    velocity = G1VelocityPolicy(
        assets["velocity_model"],
        assets["velocity_config"],
        expected_model_hash=assets["velocity_model_hash"],
        expected_config_hash=assets["velocity_config_hash"],
    )
    q, v = observe(76)
    last = future["velocity_observations"][0, 69:].copy()
    if not np.array_equal(target29(last, velocity.params), actual["joint_targets"][75]):
        raise ValueError("actual previous velocity action history codec differs")
    phase = phase_delta()
    for _ in range(76):
        phase = np.float32(np.fmod(np.float32(phase + phase_delta()), np.float32(1)))
    obs, after = observation98(
        q, v, lo.angular_velocity_body(d), actual["commands"][76], last, phase, velocity.params
    )
    action = velocity.session.run(["actions"], {"obs": obs[None]})[0][0]
    if (
        not np.array_equal(obs, future["velocity_observations"][0])
        or not np.array_equal(action, future["velocity_actions"][0])
        or after != future["velocity_phases"][0]
        or not np.array_equal(target29(action, velocity.params), actual["joint_targets"][76])
    ):
        raise ValueError("real velocity NN readback or actual98 observation differs")
    state_type, output_type, policy_type, mapping = context["types"]
    with contextlib.redirect_stdout(io.StringIO()):
        policy = policy_type(state_type(29), output_type(29))  # five original constructor warmups

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

    policy_observe(77)
    policy.target_pos_w = np.array([8.0, 0.0, 0.11], np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        policy.enter()
    receipt = install_measured_kick_history(
        policy, samples, frame=77, time_sec=samples[-1].time_sec
    )
    rebase_kick_reference(policy, entry_frame=row["admitted_reference_phase"])
    policy.time_step = int(policy.WARMUP_STEPS) + row["admitted_reference_phase"]
    if (
        receipt != meta["measured_entry_receipt"]
        or receipt != reference["measured_history_receipt"]
    ):
        raise ValueError("independently reconstructed measured entry receipt differs")
    for local, tick in enumerate(range(77, 205)):
        policy_observe(tick)
        obs = policy._build_obs()  # original pure codec, updates only this readback's history
        out = policy.ort_session.run(["actions"], {"obs": obs[None]})[0][0]
        clipped = np.clip(out, policy.action_clip_lo_il, policy.action_clip_hi_il)
        target = (policy.default_q_mj + policy.action_scale_mj * clipped[mapping]).astype(
            np.float64
        )
        if (
            not np.array_equal(obs, future["kick_observations"][local])
            or not np.array_equal(out, future["kick_raw_actions"][local])
            or not np.array_equal(clipped, future["kick_clipped_actions"][local])
            or not np.array_equal(target, actual["joint_targets"][tick])
            or future["kick_policy_ticks"][local] != policy.time_step
        ):
            raise ValueError(
                "actual547/history, original frozen kick NN output or executed target differs"
            )
        policy.last_action_il = clipped.copy()
        policy.time_step += 1
    if d.time != 0:
        raise ValueError("offline native geometry readback advanced physical time")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CURRENT_WALK_KICK_NATIVE_NN547_AND_HISTORY_READBACK_VERIFIED",
            source_head=a.expected_head,
            qualification_hash=a.qualification_hash,
            input_file_hashes=pins,
            actual_native_steps_readback=3600,
            forecast_native_steps_readback=1290,
            measured_entry_samples_reconstructed=5,
            velocity_observations98_reconstructed=1,
            kick_observations547_reconstructed=128,
            new_nn_inference_readback_calls=129,
            kick_constructor_warmup_calls_from_pinned_source=5,
            constructor_warmup_count_source_bound_not_runtime_intercepted=True,
            total_new_nn_calls_including_source_bound_warmups=134,
            original_codec_math_reused=True,
            source_nonmutation_is_pinned_producer_assertion=True,
            zero_change_composed_chain_verified=True,
            task_effectiveness_proven=False,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
