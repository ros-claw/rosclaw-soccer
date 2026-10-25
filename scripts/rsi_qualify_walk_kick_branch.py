"""Prospective one-walk +128-kick native/NN parity; not a skill gain experiment."""

import argparse
import contextlib
import io
import json
from collections import deque
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-audit", "reference-record", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--source-audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_teacher_physics_context import load_walk_kick_context

    clean(a.expected_head)
    if (
        a.output.exists()
        or a.source_audit.is_symlink()
        or digest(a.source_audit) != a.source_audit_hash
    ):
        raise ValueError("unused output and exact closed multi-control locomotion audit required")
    source = json.loads(a.source_audit.read_text())
    if (
        source["status"] != "CURRENT_VELOCITY_TRAJECTORY_RAW_AND_ACTUAL98_OBSERVATIONS_VERIFIED"
        or source["full_observation98_reconstructions"] != 65
        or source["new_physical_steps"] != 0
        or source["new_neural_calls"] != 0
        or source["future_skill_switch_prediction_qualified"] is not False
    ):
        raise ValueError("independently verified own-state locomotion future required")
    pins = dict(source["input_file_hashes"])
    pins[str(a.source_audit)] = a.source_audit_hash
    if a.reference_record.is_symlink() or pins.get(str(a.reference_record)) != digest(
        a.reference_record
    ):
        raise ValueError("exact original whole ONNX trajectory required")
    reference = json.loads(a.reference_record.read_text())
    if reference["arm"] != "onnx" or reference["entry_control"] != 77:
        raise ValueError("same preregistered original admission77 required")
    source_pins(pins)
    check_pins(pins)
    context = load_walk_kick_context(pins)
    context["assets"]["experimental_fast"] = reference["experimental_fast_approach"]
    import numpy as np

    from rosclaw_soccer.rsi.shared_kick_history import measured_kick_sample
    from rosclaw_soccer.rsi.shared_nn_forecast import policy_state_hash
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.shared_walk_kick_forecast import SharedWalkKickForecaster
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    state_type, output_type, policy_type, _ = context["types"]
    with contextlib.redirect_stdout(io.StringIO()):
        cold = policy_type(state_type(29), output_type(29))
    cold_hash = policy_state_hash(cold)
    model_hash = compiled_model_hash(context["model"])
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.current_walk_one_then_kick128_native_qualification.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            model_hash=model_hash,
            source_audit_hash=a.source_audit_hash,
            cold_template_state_hash=cold_hash,
            forecast_from_actual_control=76,
            velocity_controls=1,
            kick_controls=128,
            expected_admission_control=77,
            actual_world_ceiling=1,
            actual_native_step_ceiling=3600,
            forecast_native_step_ceiling=1290,
            maximum_future_velocity_nn_calls=1,
            maximum_future_kick_nn_calls=128,
            prepared_template_constructor_warmup_calls_from_pinned_source=5,
            actual_kick_constructor_warmup_calls_from_pinned_source=5,
            constructor_warmups_are_not_physical_control_or_optimizer_updates=True,
            source_future_arrays_used_for_control=False,
            task_effectiveness_proven=False,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    forecaster = None
    future = None
    history = deque(maxlen=5)
    previous = np.zeros(3, np.float32)
    next_frame = 0
    contacted = False

    def save_future(value):
        ap = a.output / "future.npz"
        np.savez_compressed(ap, **{k: v for k, v in value.items() if isinstance(v, np.ndarray)})
        write_once(
            a.output / "future.json",
            {k: v for k, v in value.items() if not isinstance(v, np.ndarray)},
        )
        pins[str(ap)] = digest(ap)
        pins[str(a.output / "future.json")] = digest(a.output / "future.json")

    def probe(*, stream, world, command, frame):
        nonlocal forecaster, future, previous, next_frame, contacted
        if frame != next_frame or world.control_steps != frame:
            raise ValueError("own consecutive actual velocity boundary required")
        next_frame += 1
        if frame:
            history.append(measured_kick_sample(world, "a/"))
            contacted |= bool(world.last_ball_contact_bodies["a/"])
        prior = previous.copy()
        previous = command.copy()
        if frame != 76:
            return
        forecaster = SharedWalkKickForecaster(
            world, stream, cold, expected_model_hash=model_hash, native_step_budget=1290
        )
        try:
            future = forecaster.forecast(
                velocity_controls=1,
                kick_controls=128,
                previous_command=prior,
                parameters=reference["parameters"],
                history=tuple(history),
                prior_contact=contacted,
            )
        except BaseException:
            if forecaster.last_attempt is not None:
                save_future(forecaster.last_attempt)
            raise
        save_future(future)  # Persist BEFORE the donor continues its actual future.

    row, actual = evaluate_episode(
        model=context["model"],
        assets=context["assets"],
        policy_types=context["types"],
        course=tuple(reference["course"]),
        parameters=reference["parameters"],
        hard_limits=context["hard_limits"],
        velocity_boundary_probe=probe,
    )
    ap = a.output / "actual-unvalidated.npz"
    np.savez_compressed(ap, **actual)
    pins[str(ap)] = digest(ap)
    write_once(a.output / "actual-unvalidated.json", row)
    if future is None or forecaster is None:
        raise ValueError("one prospective composed future must run atactual76")
    if (
        row["controls"] != 360
        or row["native_steps"] != 3600
        or not row["healthy_full_horizon"]
        or row["entry_control"] != 77
        or future["admission_control"] != 77
        or future["admission_rejection"] is not None
        or future["body_failed"]
        or future["controls"] != 129
        or future["native_steps"] != 1290
        or future["source_prefix_controls_not_new"] != 76
        or future["measured_entry_receipt"] != reference["measured_history_receipt"]
        or not np.array_equal(
            future["measured_entry_samples"], np.array(row["entry_state"]["measured_history"])
        )
    ):
        raise ValueError(
            "same future measured history/admission and complete healthy chain required"
        )
    original = a.reference_record.parent / reference["artifact"]["path"]
    if pins.get(str(original)) != digest(original):
        raise ValueError("original trajectory changed")
    with np.load(original, allow_pickle=False) as old:
        if set(old.files) != set(actual) or any(
            not np.array_equal(v, old[k]) for k, v in actual.items()
        ):
            raise ValueError("composed prediction modified original whole actual execution")
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        if not np.array_equal(future[key], actual[key][760:2050]):
            raise ValueError("composed full NN/native future differs fromactual: " + key)
    for key in ("joint_targets", "commands"):
        if not np.array_equal(future[key], actual[key][76:205]):
            raise ValueError("composed actual command or NN target differs: " + key)
    if (
        future["kick_observations"].shape != (128, 547)
        or future["velocity_observations"].shape != (1, 98)
        or not np.array_equal(future["policy_ticks"], np.arange(76, 205))
        or forecaster.counters()
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
        raise ValueError(
            "every future neural call, dimensional observation and native step required"
        )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CURRENT_WALK_TO_KICK_NN_AND_NATIVE_FUTURE_EXACT",
            source_head=a.expected_head,
            input_file_hashes=pins,
            declaration_hash=digest(a.output / "declaration.json"),
            actual_record=row,
            actual_worlds=1,
            actual_native_steps=3600,
            forecasts=forecaster.counters(),
            future_admission_control=77,
            source_prefix_controls_not_new=76,
            future_measured_entry_receipt_matches_original=True,
            original_whole_actual_trajectory_unchanged=True,
            prepared_template_constructor_warmup_calls_from_pinned_source=5,
            actual_kick_constructor_warmup_calls_from_pinned_source=5,
            constructor_warmup_counts_are_source_bound_not_runtime_intercepted=True,
            task_effectiveness_proven=False,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
