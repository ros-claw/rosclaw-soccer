"""Read saved velocity-NN/native futures and reconstruct actual98 observations.

No new NN inference or physics integration. Original policy codec math is
reused, not independently reimplemented. Compiled model and raw data are pinned.
"""

import argparse
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

    clean(a.expected_head)
    cp = a.qualification / "complete.json"
    if a.output.exists() or cp.is_symlink() or digest(cp) != a.qualification_hash:
        raise ValueError("unused audit and exact completed velocity qualification required")
    report = json.loads(cp.read_text())
    trajectory = report.get("trajectory_profile", False)
    if type(trajectory) is not bool:
        raise ValueError("explicit boolean velocity trajectory scope required")
    horizons = {frame: 16 if trajectory and frame != 76 else 1 for frame in (0, 20, 40, 60, 76)}
    future_controls = sum(horizons.values())
    if (
        report["status"]
        != (
            "CURRENT_VELOCITY_NN_MULTICONTROL_NATIVE_FUTURES_EXACT"
            if trajectory
            else "CURRENT_VELOCITY_NN_ONE_CONTROL_NATIVE_FUTURES_EXACT"
        )
        or report["actual_worlds"] != 1
        or report["actual_native_steps"] != 3600
        or report["original_whole_actual_trajectory_unchanged"] is not True
        or report["phase_or_action_history_reset"] is not False
        or report["future_skill_switch_prediction_qualified"] is not False
        or report["task_effectiveness_proven"] is not False
        or report["neural_optimizer_updates"] != 0
        or any(
            report[k] is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
        or report["forecasts"]
        != dict(
            forecast_calls=5,
            attempted_native_steps=10 * future_controls,
            integrated_native_steps=10 * future_controls,
            maximum_native_steps=10 * future_controls,
            future_velocity_nn_calls=future_controls,
        )
        or [r["frame"] for r in report["probes"]] != [0, 20, 40, 60, 76]
    ):
        raise ValueError("only the exact component qualification, not task effectiveness required")
    pins = dict(report["input_file_hashes"])
    pins[str(cp)] = a.qualification_hash
    dp = a.qualification / "declaration.json"
    if digest(dp) != report["declaration_hash"]:
        raise ValueError("original pre-physics declaration changed")
    pins[str(dp)] = report["declaration_hash"]
    declaration = json.loads(dp.read_text())
    if trajectory:
        from scripts.rsi_native_fresh_common import receipt

        if (
            declaration["protocol"]
            != "soccer.current_velocity_nn_native_trajectory_qualification.v1"
            or declaration["probe_horizons"] != {str(k): v for k, v in horizons.items()}
            or declaration["forecast_native_step_ceiling"] != 650
            or declaration["maximum_future_velocity_nn_calls"] != 65
        ):
            raise ValueError("separate65-control trajectory qualification declaration required")
        _, component = receipt(pins, declaration["component_audit_hash"])
        if component["status"] != "CURRENT_VELOCITY_RAW_NATIVE_AND_ACTUAL98_OBSERVATIONS_VERIFIED":
            raise ValueError("independently verified one-control prerequisite required")
    if a.reference_record.is_symlink() or pins.get(str(a.reference_record)) != digest(
        a.reference_record
    ):
        raise ValueError("pinned original whole execution required")
    ref = json.loads(a.reference_record.read_text())
    if ref["arm"] != "onnx" or ref["entry_control"] != 77:
        raise ValueError("same original pre-handoff comparison required")
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("one pinned original body/codec required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.velocity_policy import (
        VelocityPolicyParameters,
        observation98,
        phase_delta,
        target29,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    if compiled_model_hash(model) != report["model_hash"]:
        raise ValueError("compiled original body differs")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    cfg = one("/velocity/v0/params/deploy.yaml")
    params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])

    def arrays(path):
        if pins.get(str(path)) != digest(path):
            raise ValueError("saved whole trajectory or velocity forecast changed")
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k].copy() for k in saved.files}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("nonfinite native data")
        return values

    actual = arrays(a.qualification / "actual-unvalidated.npz")
    old = arrays(a.reference_record.parent / ref["artifact"]["path"])
    row = report["actual_record"]
    root = actual["qpos"][:, lo.rq : lo.rq + 7]
    if (
        set(actual) != set(old)
        or any(not np.array_equal(v, old[k]) for k, v in actual.items())
        or row["controls"] != 360
        or row["native_steps"] != 3600
        or row["healthy_full_horizon"] is not True
        or any(row[k] != ref[k] for k in ("course", "parameters", "entry_control"))
        or (root[:, 2] < 0.65).any()
        or (1 - 2 * (root[:, 4] ** 2 + root[:, 5] ** 2) < 0.9).any()
        or (abs(actual["torque"][:, lo.ai]) > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12).any()
        or hash_json(relabel_arrays(actual, controls=360, ball_address=lo.bq, root_address=lo.rq))
        != hash_json(row["outcome"])
    ):
        raise ValueError("original whole execution or full native health/task label differs")
    # Reconstruct the declared initial state, without a physics step.
    d = mujoco.MjData(model)
    d.qpos[lo.rq : lo.rq + 7] = [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]
    d.qpos[lo.qi] = params.defaults
    x, y, vx = row["course"]
    d.qpos[lo.bq : lo.bq + 7] = [x, y, 0.11, 1.0, 0.0, 0.0, 0.0]
    d.qvel[lo.bv], d.qvel[lo.bv + 4] = vx, vx / 0.11
    mujoco.mj_forward(model, d)
    initial_q, initial_v = d.qpos.copy(), d.qvel.copy()
    phases = [phase_delta()]
    for _ in range(77):
        phases.append(np.float32(np.fmod(np.float32(phases[-1] + phase_delta()), np.float32(1))))
    for record in report["probes"]:
        frame = record["frame"]
        n = horizons[frame]
        forecast = arrays(a.qualification / record["artifact"])
        if any(
            not np.array_equal(forecast[k], actual[k][frame * 10 : (frame + n) * 10])
            for k in ("qpos", "qvel", "torque", "ball_forces")
        ):
            raise ValueError("saved copied velocity future differs from actual native execution")
        targets = forecast["joint_targets"] if trajectory else forecast["joint_target"][None]
        actions = forecast["actions"] if trajectory else forecast["action"][None]
        observations = forecast["observations"] if trajectory else forecast["observation"][None]
        if (
            targets.shape != (n, 29)
            or actions.shape != (n, 29)
            or observations.shape != (n, 98)
            or not np.array_equal(targets, actual["joint_targets"][frame : frame + n])
            or (
                trajectory
                and (
                    not np.array_equal(forecast["commands"], actual["commands"][frame : frame + n])
                    or not np.array_equal(forecast["policy_ticks"], np.arange(frame, frame + n))
                    or forecast["phases"].shape != (n,)
                    or record["controls"] != n
                    or record["requested_controls"] != n
                )
            )
        ):
            raise ValueError("own future command, target, horizon or absolute policy clock differs")
        for local in range(n):
            tick = frame + local
            if not np.array_equal(target29(actions[local], params), targets[local]):
                raise ValueError("saved copied NN action does not decode to actual executed target")
            if tick:
                d.qpos[:], d.qvel[:] = actual["qpos"][tick * 10 - 1], actual["qvel"][tick * 10 - 1]
            else:
                d.qpos[:], d.qvel[:] = initial_q, initial_v
            q, v = lo.canonical(d)
            # Euler caches are PRE-integration; measured q/v are POST-integration.
            d.qpos[:] = actual["qpos"][tick * 10 - 2] if tick else initial_q
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)
            angular = lo.angular_velocity_body(d)
            last = observations[local, 69:].copy()
            expected_last_target = (
                actual["joint_targets"][tick - 1]
                if tick
                else target29(np.zeros(29, np.float32), params)
            )
            if not np.array_equal(target29(last, params), expected_last_target):
                raise ValueError("copied prior action is not bound to actual previous target")
            obs, phase = observation98(
                q, v, angular, actual["commands"][tick], last, phases[tick], params
            )
            if not np.array_equal(obs, observations[local]) or (
                trajectory and forecast["phases"][local] != phase
            ):
                raise ValueError("reconstructed actual98/phase differs from own future")
        if (
            record["phase_after"] != float(phase)
            or record["actual_completed_control"] != frame
            or record["copied_next_frame"] != frame + n
        ):
            raise ValueError("reconstructed actual98 observation or uninterrupted phase differs")
    if d.time != 0:
        raise ValueError("raw-data observation reconstruction advanced physical time")
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="CURRENT_VELOCITY_TRAJECTORY_RAW_AND_ACTUAL98_OBSERVATIONS_VERIFIED"
            if trajectory
            else "CURRENT_VELOCITY_RAW_NATIVE_AND_ACTUAL98_OBSERVATIONS_VERIFIED",
            trajectory_profile=trajectory,
            source_head=a.expected_head,
            qualification_hash=a.qualification_hash,
            input_file_hashes=pins,
            actual_native_steps_readback=3600,
            forecast_native_steps_readback=10 * future_controls,
            full_observation98_reconstructions=future_controls,
            actual_action_history_codec_reconstructions=future_controls,
            source_stream_nonmutation_is_pinned_producer_assertion=True,
            original_observation_codec_math_reused=True,
            future_skill_switch_prediction_qualified=False,
            task_effectiveness_proven=False,
            new_physical_steps=0,
            new_neural_calls=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
