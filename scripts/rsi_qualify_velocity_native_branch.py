"""Qualify separately budgeted one/multi-control velocity futures, not a teacher."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("effect-review", "reference-record", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--effect-review-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--trajectory-profile", action="store_true")
    p.add_argument("--component-audit", type=Path)
    p.add_argument("--component-audit-hash")
    a = p.parse_args()
    if a.trajectory_profile != (
        a.component_audit is not None and a.component_audit_hash is not None
    ) or (
        not a.trajectory_profile
        and (a.component_audit is not None or a.component_audit_hash is not None)
    ):
        raise ValueError("separate trajectory profile requires exact one-control component audit")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if (
        a.output.exists()
        or a.effect_review.is_symlink()
        or digest(a.effect_review) != a.effect_review_hash
    ):
        raise ValueError("unused output and exact original native-effect lineage required")
    effect = json.loads(a.effect_review.read_text())
    if effect["status"] != "TASK_SPACE_NATIVE_EFFECT_READBACK_COMPLETE":
        raise ValueError("original closed native task-space effect review required")
    pins = dict(effect["input_file_hashes"])
    pins[str(a.effect_review)] = a.effect_review_hash
    if a.trajectory_profile:
        from scripts.rsi_native_fresh_common import receipt
        from scripts.rsi_stable_plastic_sources import merge_pins

        if a.component_audit.is_symlink() or digest(a.component_audit) != a.component_audit_hash:
            raise ValueError("exact closed independent one-control audit required")
        component = json.loads(a.component_audit.read_text())
        if (
            component["status"] != "CURRENT_VELOCITY_RAW_NATIVE_AND_ACTUAL98_OBSERVATIONS_VERIFIED"
            or component["full_observation98_reconstructions"] != 5
            or component["new_physical_steps"] != 0
            or component["new_neural_calls"] != 0
            or component["future_skill_switch_prediction_qualified"] is not False
        ):
            raise ValueError("independent current-stream actual98/native component proof required")
        merge_pins(pins, component["input_file_hashes"])
        pins[str(a.component_audit)] = a.component_audit_hash
        _, qualified = receipt(pins, component["qualification_hash"])
        if qualified["status"] != "CURRENT_VELOCITY_NN_ONE_CONTROL_NATIVE_FUTURES_EXACT":
            raise ValueError("same one-control qualification lineage required")
    if a.reference_record.is_symlink() or pins.get(str(a.reference_record)) != digest(
        a.reference_record
    ):
        raise ValueError("exact pinned original ONNX whole execution required")
    reference = json.loads(a.reference_record.read_text())
    if reference["arm"] != "onnx" or reference["entry_control"] != 77:
        raise ValueError("same preregistered original velocity-to-kick source boundary required")
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        rows = [Path(k) for k in pins if k.endswith(suffix)]
        if len(rows) != 1:
            raise ValueError("unique original pinned native asset required: " + suffix)
        return rows[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.shared_velocity_forecast import SharedVelocityForecaster
    from rosclaw_soccer.rsi.shared_velocity_trajectory import SharedVelocityTrajectoryForecaster
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    cfgs = [
        Path(k)
        for k in pins
        if k.endswith(".json")
        and Path(k).stat().st_size < 2048
        and json.loads(Path(k).read_text()).get("schema")
        == "soccer.native_single_finisher_experimental_approach.v1"
    ]
    if len(cfgs) != 1:
        raise ValueError("one original experimental-envelope ancestry required")
    assets, _ = admit_heading_evidence(cfgs[0], pins[str(cfgs[0])])
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    assets.update(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
        experimental_fast=reference["experimental_fast_approach"],
    )
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    model_hash = compiled_model_hash(model)
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    frames = (0, 20, 40, 60, 76)
    horizons = {frame: 16 if a.trajectory_profile and frame != 76 else 1 for frame in frames}
    future_controls = sum(horizons.values())
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.current_velocity_nn_native_trajectory_qualification.v1"
            if a.trajectory_profile
            else "soccer.current_velocity_nn_native_one_control_qualification.v1",
            trajectory_profile=a.trajectory_profile,
            component_audit_hash=a.component_audit_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            model_hash=model_hash,
            probes=frames,
            controls_per_probe=None if a.trajectory_profile else 1,
            probe_horizons=horizons,
            actual_world_ceiling=1,
            actual_native_step_ceiling=3600,
            forecast_native_step_ceiling=10 * future_controls,
            maximum_future_velocity_nn_calls=future_controls,
            future_skill_switch_prediction_qualified=False,
            task_effectiveness_proven=False,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    forecaster = None
    saved = []
    last_command = np.zeros(3, np.float32)
    observed_commands = 0

    def probe(*, stream, world, command, frame):
        nonlocal forecaster, last_command, observed_commands
        if frame != observed_commands:
            raise ValueError(
                "one consecutive actual command observation before each velocity proposal"
            )
        prior = last_command.copy()
        last_command = command.copy()
        observed_commands += 1
        if frame not in frames:
            return
        if forecaster is None:
            cls = (
                SharedVelocityTrajectoryForecaster
                if a.trajectory_profile
                else SharedVelocityForecaster
            )
            forecaster = cls(
                world,
                stream,
                expected_model_hash=model_hash,
                native_step_budget=10 * future_controls,
            )
        if forecaster.world is not world or forecaster.stream is not stream:
            raise ValueError("same original current-world velocity actor required")
        try:
            predicted = (
                forecaster.forecast(
                    controls=horizons[frame],
                    previous_command=prior,
                    parameters=reference["parameters"],
                )
                if a.trajectory_profile
                else forecaster.forecast(command)
            )
        except BaseException:
            if a.trajectory_profile and forecaster.last_attempt is not None:
                failed = forecaster.last_attempt
                np.savez_compressed(
                    a.output / f"forecast-{frame:03d}-failed-partial.npz",
                    **{k: v for k, v in failed.items() if isinstance(v, np.ndarray)},
                )
                write_once(
                    a.output / f"forecast-{frame:03d}-failed-partial.json",
                    {k: v for k, v in failed.items() if not isinstance(v, np.ndarray)},
                )
            raise
        artifact = a.output / f"forecast-{frame:03d}.npz"
        np.savez_compressed(
            artifact,
            **{
                k: predicted[k]
                for k in (
                    (
                        "qpos",
                        "qvel",
                        "torque",
                        "ball_forces",
                        "joint_targets",
                        "commands",
                        "observations",
                        "actions",
                        "phases",
                        "policy_ticks",
                    )
                    if a.trajectory_profile
                    else (
                        "qpos",
                        "qvel",
                        "torque",
                        "ball_forces",
                        "joint_target",
                        "observation",
                        "action",
                    )
                )
            },
        )
        pins[str(artifact)] = digest(artifact)
        saved.append(
            dict(
                frame=frame,
                artifact=artifact.name,
                phase_after=float(predicted["phases"][-1])
                if a.trajectory_profile
                else predicted["phase_after"],
                controls=predicted["controls"] if a.trajectory_profile else 1,
                requested_controls=horizons[frame],
                copied_next_frame=predicted["copied_next_frame"],
                actual_completed_control=predicted["actual_completed_control"],
                donor_stream_hash=predicted["donor_stream_hash"],
                copied_stream_after_hash=predicted["copied_stream_after_hash"],
            )
        )
        write_once(a.output / f"forecast-{frame:03d}.json", saved[-1])

    row, actual = evaluate_episode(
        model=model,
        assets=assets,
        policy_types=types,
        course=tuple(reference["course"]),
        parameters=reference["parameters"],
        hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
        velocity_boundary_probe=probe,
    )
    artifact = a.output / "actual-unvalidated.npz"
    np.savez_compressed(artifact, **actual)
    pins[str(artifact)] = digest(artifact)
    write_once(a.output / "actual-unvalidated.json", row)
    if (
        row["controls"] != 360
        or row["native_steps"] != 3600
        or row["healthy_full_horizon"] is not True
        or row["entry_control"] != 77
        or [r["frame"] for r in saved] != list(frames)
    ):
        raise ValueError("same full healthy actual episode and five pre-kick NN probes required")
    original = a.reference_record.parent / reference["artifact"]["path"]
    if pins.get(str(original)) != digest(original):
        raise ValueError("original comparator arrays changed")
    with np.load(original, allow_pickle=False) as old:
        if set(old.files) != set(actual) or any(
            not np.array_equal(old[k], v) for k, v in actual.items()
        ):
            raise ValueError("velocity NN forecasts changed original whole actual trajectory")
    for record in saved:
        frame = record["frame"]
        with np.load(a.output / record["artifact"], allow_pickle=False) as forecast:
            n = horizons[frame]
            if any(
                not np.array_equal(forecast[k], actual[k][frame * 10 : (frame + n) * 10])
                for k in ("qpos", "qvel", "torque", "ball_forces")
            ):
                raise ValueError("copied velocity NN/native future differs from actual execution")
            if a.trajectory_profile:
                if (
                    not np.array_equal(
                        forecast["joint_targets"], actual["joint_targets"][frame : frame + n]
                    )
                    or not np.array_equal(
                        forecast["commands"], actual["commands"][frame : frame + n]
                    )
                    or not np.array_equal(forecast["policy_ticks"], np.arange(frame, frame + n))
                ):
                    raise ValueError("own-state future command, neural target or clock differs")
            elif not np.array_equal(
                forecast["joint_target"], actual["joint_targets"][frame]
            ) or not np.array_equal(forecast["observation"][6:9], actual["commands"][frame]):
                raise ValueError("copied current target or command differs from actual control")
        if (
            record["actual_completed_control"] != frame
            or record["copied_next_frame"] != frame + horizons[frame]
        ):
            raise ValueError("velocity NN forecast reset or fabricated the absolute gait clock")
    assert forecaster is not None
    counters = forecaster.counters()
    if counters != dict(
        forecast_calls=5,
        attempted_native_steps=10 * future_controls,
        integrated_native_steps=10 * future_controls,
        maximum_native_steps=10 * future_controls,
        future_velocity_nn_calls=future_controls,
    ):
        raise ValueError("all five copied velocity NN and native calls must be charged")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CURRENT_VELOCITY_NN_MULTICONTROL_NATIVE_FUTURES_EXACT"
            if a.trajectory_profile
            else "CURRENT_VELOCITY_NN_ONE_CONTROL_NATIVE_FUTURES_EXACT",
            trajectory_profile=a.trajectory_profile,
            source_head=a.expected_head,
            input_file_hashes=pins,
            declaration_hash=digest(a.output / "declaration.json"),
            model_hash=model_hash,
            actual_record=row,
            probes=saved,
            actual_worlds=1,
            actual_native_steps=3600,
            forecasts=counters,
            original_whole_actual_trajectory_unchanged=True,
            phase_or_action_history_reset=False,
            future_skill_switch_prediction_qualified=False,
            task_effectiveness_proven=False,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
