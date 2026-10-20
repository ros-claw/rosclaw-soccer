"""Four bounded native mobility cases on the new, unchanged training body.

Two initial conditions x idle/pulse, each independently repeated in full.
Same qualified ONNX/98D codec; no learning, gain changes, or football claim.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("export-root", "qualification-root", "policy-reference-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "export-unit",
        "export-hash",
        "qualification-unit",
        "qualification-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("frozen source and unused native mobility evidence output required")
    for root, unit, expected in (
        (a.export_root, a.export_unit, a.export_hash),
        (a.qualification_root, a.qualification_unit, a.qualification_hash),
    ):
        wait_for_terminal_collection(unit)
        if digest(root / "complete.json") != expected:
            raise ValueError("immutable actual terminal receipt required")
    export = json.loads((a.export_root / "complete.json").read_text())
    qual = json.loads((a.qualification_root / "complete.json").read_text())
    if (
        export["status"] != "MJLAB_NATIVE_EXPORT_ONE_CONTROL_DIAGNOSTIC_NOT_LONG_RUN_QUALIFIED"
        or qual["status"] != "G1_VELOCITY_POLICY_CPP_CODEC_NN_QUALIFIED_NOT_PHYSICS"
        or not qual["cpp_float32_target_exact"]
    ):
        raise ValueError("actual compiled-model export and qualified deployment codec required")
    policy_dir = a.policy_reference_root / "deploy/robots/g1/config/policy/velocity/v0"
    actor_path = policy_dir / "exported/policy.onnx"
    config_path = policy_dir / "params/deploy.yaml"
    # The terminal qualification binds assets through input_file_hashes;
    # model_hash/config_hash are declaration fields, not terminal fields.
    model_hash = qual["input_file_hashes"][str(actor_path)]
    config_hash = qual["input_file_hashes"][str(config_path)]
    pins = {
        str(a.export_root / "complete.json"): a.export_hash,
        str(a.qualification_root / "complete.json"): a.qualification_hash,
        str(a.export_root / "compiled_model.mjb"): export["model_hash"],
        str(a.export_root / "native-gpu-control1.npz"): export["state_artifact_hash"],
        str(actor_path): model_hash,
        str(config_path): config_hash,
    }
    for directory in (Path.cwd() / "src", Path.cwd() / "scripts"):
        for source in directory.rglob("*.py"):
            pins[str(source)] = digest(source)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            max_native_steps=24000,
            max_neural_calls=2400,
            cases=4,
            repeats=2,
            physical_horizon_controls=300,
            pulse_onset=50,
            pulse_frames=100,
            pulse_speed_mps=0.3,
            original_exported_dynamics_unchanged=True,
            not_inherited_old_body_certificate=True,
            new_optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    actor = None
    rows = []
    steps = 0
    calls = 0
    attempted_steps = 0
    prior_actor_calls = 0
    data = None
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.physics.g1_velocity_observation import world_command_to_body
        from rosclaw_soccer.providers.g1.velocity_policy import (
            G1VelocityPolicy,
            observation98,
            phase_delta,
            target29,
        )
        from rosclaw_soccer.rsi.locomotion_response_protocol import LocomotionResponseCourse
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        if mujoco.__version__ != "3.5.0":
            raise ValueError("exact original exported engine required")
        model = mujoco.MjModel.from_binary_path(str(a.export_root / "compiled_model.mjb"))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        write_once(a.output / "body-profile.json", layout.profile())
        ai = layout.ai
        kp = model.actuator_gainprm[ai, 0]
        bias = model.actuator_biasprm[ai]
        if (
            not np.all(model.actuator_forcelimited[ai])
            or np.any(kp <= 0)
            or not np.array_equal(bias[:, 1], -kp)
            or np.any(bias[:, 2] >= 0)
            or np.any(model.actuator_gear[ai, 0] != 1)
            or np.any(model.actuator_gaintype[ai] != int(mujoco.mjtGain.mjGAIN_FIXED))
            or np.any(model.actuator_biastype[ai] != int(mujoco.mjtBias.mjBIAS_AFFINE))
            or mujoco.get_mjcb_control() is not None
        ):
            raise ValueError("qualified limited native position-actuator encoding required")
        with np.load(a.export_root / "native-gpu-control1.npz", allow_pickle=False) as z:
            recorded_q = z["qpos"][0].copy()
            recorded_v = z["qvel"][0].copy()
        robot_bodies = set()
        for b in range(model.nbody):
            parent = b
            while parent not in (0, layout.pelvis):
                parent = int(model.body_parentid[parent])
            if parent == layout.pelvis:
                robot_bodies.add(b)
        results = []
        for case in range(4):
            first_raw = None
            first_result = None
            for repeat in range(2):
                prior_actor_calls = calls
                actor = G1VelocityPolicy(
                    actor_path,
                    config_path,
                    expected_model_hash=model_hash,
                    expected_config_hash=config_hash,
                )
                if actor.contract_hash != qual["policy_contract_hash"]:
                    raise ValueError("unchanged qualified actor and parameter contract required")
                data = mujoco.MjData(model)
                if case < 2:
                    data.qpos[layout.rq : layout.rq + 7] = [0, 0, 0.793, 1, 0, 0, 0]
                    data.qpos[layout.qi] = actor.params.defaults
                    data.qvel[:] = 0
                    data.qpos[layout.bq : layout.bq + 7] = [3, 3, 0.11, 1, 0, 0, 0]
                else:
                    data.qpos[:] = recorded_q
                    data.qvel[:] = recorded_v
                mujoco.mj_forward(model, data)
                q, _ = layout.canonical(data)
                w, x, y, z = q[3:7]
                yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                xy = (
                    (float(0.3 * np.cos(yaw)), float(0.3 * np.sin(yaw))) if case % 2 else (0.0, 0.0)
                )
                course = LocomotionResponseCourse(case, "new_body_probe", xy, 50)
                warning = tuple(int(w.number) for w in data.warning)
                rows = []
                phase = phase_delta()
                last = np.zeros(29, dtype=np.float32)
                for frame in range(300):
                    mujoco.mj_forward(model, data)
                    q, v = layout.canonical(data)
                    cmd = world_command_to_body(q, course.command(frame))
                    omega = layout.angular_velocity_body(data)
                    target = actor.propose(q, v, omega, cmd, frame=frame)
                    calls += 1
                    obs, phase = observation98(q, v, omega, cmd, last, phase, actor.params)
                    if not np.array_equal(obs, actor.last_observation) or not np.array_equal(
                        target, target29(actor.last_action, actor.params)
                    ):
                        raise ValueError(
                            "new native layout changed the qualified neural feature/action codec"
                        )
                    last = actor.last_action.copy()
                    row = dict(
                        qpos_before=q.copy(),
                        qvel_before=v.copy(),
                        observation=obs.copy(),
                        action=last.copy(),
                        target=target.copy(),
                        command_body=cmd.copy(),
                        command_world=np.array(course.command(frame)),
                        omega=omega.copy(),
                        phase=phase,
                    )
                    data.ctrl[ai] = target
                    qs = []
                    vs = []
                    forces = []
                    lengths = []
                    velocities = []
                    ball_forces = []
                    for _ in range(10):
                        before = data.time
                        attempted_steps += 1
                        mujoco.mj_step(model, data)
                        steps += 1
                        if (
                            not np.isclose(data.time - before, 0.002, atol=1e-10, rtol=0)
                            or tuple(int(w.number) for w in data.warning) != warning
                        ):
                            raise ValueError("native body warning or implicit integration restart")
                        cq, cv = layout.canonical(data)
                        qs.append(cq)
                        vs.append(cv)
                        force = data.actuator_force[ai].copy()
                        expected = (
                            kp * target
                            + bias[:, 0]
                            + bias[:, 1] * data.actuator_length[ai]
                            + bias[:, 2] * data.actuator_velocity[ai]
                        )
                        expected = np.clip(
                            expected,
                            model.actuator_forcerange[ai, 0],
                            model.actuator_forcerange[ai, 1],
                        )
                        if not np.allclose(force, expected, rtol=0, atol=1e-8):
                            raise ValueError(
                                "actual native actuator force differs from compiled PD encoding"
                            )
                        forces.append(force)
                        lengths.append(data.actuator_length[ai].copy())
                        velocities.append(data.actuator_velocity[ai].copy())
                        ball_force = 0.0
                        for contact in range(data.ncon):
                            c = data.contact[contact]
                            b1 = int(model.geom_bodyid[c.geom1])
                            b2 = int(model.geom_bodyid[c.geom2])
                            if (b1 == layout.ball and b2 in robot_bodies) or (
                                b2 == layout.ball and b1 in robot_bodies
                            ):
                                wrench = np.zeros(6)
                                mujoco.mj_contactForce(model, data, contact, wrench)
                                ball_force = max(ball_force, float(np.linalg.norm(wrench[:3])))
                        ball_forces.append(ball_force)
                    row.update(
                        native_qpos=np.asarray(qs),
                        native_qvel=np.asarray(vs),
                        actual_force=np.asarray(forces),
                        actuator_length=np.asarray(lengths),
                        actuator_velocity=np.asarray(velocities),
                        ball_robot_force=np.asarray(ball_forces),
                    )
                    rows.append(row)
                if actor.neural_calls != 300 or actor.next_frame != 300:
                    raise ValueError("whole independent H300 neural session required")
                raw = {k: np.array([row[k] for row in rows]) for k in rows[0]}
                if any(not np.isfinite(value).all() for value in raw.values()):
                    raise ValueError("all actual native and neural evidence must be finite")
                if not np.array_equal(
                    raw["qpos_before"][1:], raw["native_qpos"][:-1, -1]
                ) or not np.array_equal(raw["qvel_before"][1:], raw["native_qvel"][:-1, -1]):
                    raise ValueError("continuous actual body required")
                upright = 1 - 2 * (
                    raw["native_qpos"][:, :, 4] ** 2 + raw["native_qpos"][:, :, 5] ** 2
                )
                height = float(raw["native_qpos"][:, :, 2].min())
                ball_force = float(raw["ball_robot_force"].max())
                mean = raw["qvel_before"][100:150, :2].mean(0)
                desired = np.asarray(xy)
                error = float(np.linalg.norm(mean - desired))
                terminal = float(np.linalg.norm(raw["qvel_before"][250:300, :2].mean(0)))
                drift = float(
                    np.linalg.norm(raw["qpos_before"][-1, :2] - raw["qpos_before"][150, :2])
                )
                stable = bool(height >= 0.65 and upright.min() >= 0.9 and ball_force <= 1)
                result = dict(
                    case=case,
                    initial="policy_home" if case < 2 else "recorded_motion",
                    world_pulse=xy,
                    minimum_height_m=height,
                    minimum_upright=float(upright.min()),
                    max_robot_ball_force_n=ball_force,
                    pulse_mean_velocity_mps=mean.tolist(),
                    pulse_velocity_error_mps=error,
                    terminal_mean_speed_mps=terminal,
                    stop_drift_m=drift,
                    physical_response_guard=stable,
                    command_response_research_gate=bool(
                        stable and error <= 0.08 and terminal <= 0.05 and drift <= 0.15
                    ),
                    native_steps=3000,
                    neural_calls=300,
                    new_optimizer_steps=0,
                    football_skill_proven=False,
                )
                folder = a.output / f"case{case}-repeat{repeat}"
                folder.mkdir()
                np.savez_compressed(folder / "native300.npz", **raw)
                write_once(
                    folder / "complete.json",
                    dict(**result, raw_hash=digest(folder / "native300.npz")),
                )
                if first_raw is None:
                    first_raw = raw
                    first_result = result
                elif result != first_result or any(
                    not np.array_equal(raw[k], first_raw[k]) for k in raw
                ):
                    raise ValueError("independent full neural/native repeat differs")
                print(json.dumps(dict(**result, repeat=repeat)), flush=True)
                rows = []
            results.append(dict(**first_result, independent_full_repeat_exact=True))
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="VELOCITY_PRIOR_NEW_BODY_MOBILITY_PROBE_NOT_FOOTBALL_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                native_steps=steps,
                neural_calls=calls,
                attempted_native_steps=attempted_steps,
                new_optimizer_steps=0,
                unchanged_exported_model=True,
                all_four_independent_full_repeats_exact=True,
                football_skill_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        if data is not None:
            import numpy as np

            np.savez_compressed(
                a.output / "failed-live-state.npz",
                qpos=data.qpos.copy(),
                qvel=data.qvel.copy(),
                actuator_force=data.actuator_force.copy(),
            )
        if rows:
            import numpy as np

            np.savez_compressed(
                a.output / "partial-failed.npz",
                **{k: np.array([r[k] for r in rows]) for k in rows[0]},
            )
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                native_steps=steps,
                attempted_native_steps=attempted_steps,
                completed_neural_calls=calls,
                current_actor_neural_calls=actor.neural_calls if actor is not None else 0,
                actual_neural_calls=prior_actor_calls
                + (actor.neural_calls if actor is not None else 0),
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
