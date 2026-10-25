"""Three preregistered current-body switch previews; NOT actual skill improvement."""

import argparse
import contextlib
import io
import json
from collections import deque
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composed-complete", type=Path, required=True)
    parser.add_argument("--composed-hash", required=True)
    parser.add_argument("--reference-record", type=Path, required=True)
    parser.add_argument("--reference-hash")
    parser.add_argument("--failed-teacher-record", type=Path)
    parser.add_argument("--failed-teacher-hash")
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_teacher_physics_context import load_walk_kick_context

    clean(args.expected_head)
    if args.output.exists() or args.composed_complete.is_symlink():
        raise ValueError("unused output and ordinary closed composed proof required")
    if digest(args.composed_complete) != args.composed_hash:
        raise ValueError("exact closed composed proof required")
    proof = json.loads(args.composed_complete.read_text())
    if proof["status"] != "CURRENT_WALK_TO_KICK_NN_AND_NATIVE_FUTURE_EXACT":
        raise ValueError("qualified composed native future required")
    pins = dict(proof["input_file_hashes"])
    pins[str(args.composed_complete)] = args.composed_hash
    failure_mode = args.reference_hash is not None
    if failure_mode != (args.failed_teacher_record is not None) or failure_mode != (
        args.failed_teacher_hash is not None
    ):
        raise ValueError("public failure mode requires both exact baseline and teacher commitments")
    reference_hash = args.reference_hash if failure_mode else pins.get(str(args.reference_record))
    if args.reference_record.is_symlink() or reference_hash != digest(args.reference_record):
        raise ValueError("original pinned record required")
    reference = json.loads(args.reference_record.read_text())
    if failure_mode:
        if (
            args.failed_teacher_record.is_symlink()
            or digest(args.failed_teacher_record) != args.failed_teacher_hash
        ):
            raise ValueError("exact public failed teacher record required")
        teacher = json.loads(args.failed_teacher_record.read_text())
        if (
            reference["arm"] != "B"
            or teacher["arm"] != "C"
            or reference["outcome"]["task_pass"] is not False
            or teacher["outcome"]["task_pass"] is not False
            or reference["course"] != teacher["course"]
            or reference["parameters"] != teacher["parameters"]
            or reference["experimental_fast_approach"] != teacher["experimental_fast_approach"]
        ):
            raise ValueError("paired unchanged-course public baseline/teacher failures required")
        pins[str(args.failed_teacher_record)] = args.failed_teacher_hash
        pins[str(args.reference_record)] = reference_hash
        artifact = args.reference_record.parent / reference["artifact"]["path"]
        if (
            artifact.parent != args.reference_record.parent
            or digest(artifact) != reference["artifact"]["hash"]
        ):
            raise ValueError("original local baseline raw commitment required")
        pins[str(artifact)] = reference["artifact"]["hash"]
    elif reference["arm"] != "onnx" or reference["entry_control"] != 77:
        raise ValueError("same frozen qualification source and original entry77 required")
    entry = reference["entry_control"]
    if type(entry) is not int or not 50 <= entry <= 219:
        raise ValueError("original entry with complete bounded switch horizons required")
    origin = entry - 1
    source_pins(pins)
    check_pins(pins)
    context = load_walk_kick_context(pins)
    context["assets"]["experimental_fast"] = reference["experimental_fast_approach"]
    import numpy as np

    from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
    from rosclaw_soccer.rsi.shared_kick_history import measured_kick_sample
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.shared_walk_kick_forecast import SharedWalkKickForecaster
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    state_type, output_type, policy_type, _ = context["types"]
    with contextlib.redirect_stdout(io.StringIO()):
        cold = policy_type(state_type(29), output_type(29))
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            protocol="soccer.measured_walk_kick_switch_three_preview.v1",
            source_head=args.expected_head,
            input_file_hashes=pins,
            actual_world_ceiling=1,
            actual_native_step_ceiling=3600,
            forecast_native_step_ceiling=4050,
            from_actual_control=origin,
            public_baseline_and_teacher_failed=failure_mode,
            original_entry_control=entry,
            delays_relative_to_original_entry=[0, 6, 12],
            velocity_controls=[1, 7, 13],
            kick_controls=128,
            maximum_future_velocity_nn_calls=21,
            maximum_future_kick_nn_calls=384,
            original_constructor_warmups_source_bound_not_intercepted=10,
            no_automatic_search_expansion=True,
            actual_switch_selection_executed=False,
            task_effectiveness_proven=False,
            neural_optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    history = deque(maxlen=5)
    previous = np.zeros(3, np.float32)
    contacted, next_frame = False, 0
    forecasts, scores, forecaster = [], [], None

    def probe(*, stream, world, command, frame):
        nonlocal previous, contacted, next_frame, forecaster
        if frame != next_frame or frame != world.control_steps:
            raise ValueError("consecutive actual velocity boundaries required")
        next_frame += 1
        if frame:
            history.append(measured_kick_sample(world, "a/"))
            contacted |= bool(world.last_ball_contact_bodies["a/"])
        prior, previous = previous.copy(), command.copy()
        if frame != origin:
            return
        forecaster = SharedWalkKickForecaster(
            world,
            stream,
            cold,
            expected_model_hash=compiled_model_hash(context["model"]),
            native_step_budget=4050,
        )
        for velocity in (1, 7, 13):
            try:
                future = forecaster.forecast(
                    velocity_controls=velocity,
                    kick_controls=128,
                    previous_command=prior,
                    parameters=reference["parameters"],
                    history=tuple(history),
                    prior_contact=contacted,
                )
            except BaseException:
                if forecaster.last_attempt is not None:
                    save(forecaster.last_attempt, velocity, "failed")
                raise
            save(future, velocity, "future")  # Raw before scoring or donor continuation.
            forecasts.append(future)
            window = None
            if future["admission_control"] is not None and future["controls"] == velocity + 128:
                kick_only = dict(
                    future,
                    controls=128,
                    qpos=future["qpos"][velocity * 10 :],
                    ball_forces=future["ball_forces"][velocity * 10 :],
                )
                window = predicted_touch_window(
                    kick_only,
                    ball_address=world.layouts["a/"].bq,
                    root_address=world.layouts["a/"].rq,
                    start_control=origin + velocity,
                    diagnostic_tail=True,
                )
            scores.append(dict(delay=velocity - 1, window=window, eligible=window is not None))

    def save(future, velocity, label):
        path = args.output / f"{label}-v{velocity:02d}.npz"
        np.savez_compressed(path, **{k: v for k, v in future.items() if isinstance(v, np.ndarray)})
        pins[str(path)] = digest(path)
        path = args.output / f"{label}-v{velocity:02d}.json"
        write_once(path, {k: v for k, v in future.items() if not isinstance(v, np.ndarray)})
        pins[str(path)] = digest(path)

    row, actual = evaluate_episode(
        model=context["model"],
        assets=context["assets"],
        policy_types=context["types"],
        course=tuple(reference["course"]),
        parameters=reference["parameters"],
        hard_limits=context["hard_limits"],
        velocity_boundary_probe=probe,
    )
    path = args.output / "actual-unvalidated.npz"
    np.savez_compressed(path, **actual)
    pins[str(path)] = digest(path)
    write_once(args.output / "actual-unvalidated.json", row)
    if forecaster is None or len(forecasts) != 3 or row["native_steps"] != 3600:
        raise ValueError("declared three previews and complete actual rollout required")
    original = args.reference_record.parent / reference["artifact"]["path"]
    with np.load(original, allow_pickle=False) as old:
        if set(old.files) != set(actual) or any(
            not np.array_equal(v, old[k]) for k, v in actual.items()
        ):
            raise ValueError("previews changed the original actual rollout")
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        if not np.array_equal(forecasts[0][key], actual[key][origin * 10 : (origin + 129) * 10]):
            raise ValueError("zero-delay future differs from original actual raw: " + key)
    for key in ("joint_targets", "commands"):
        if not np.array_equal(forecasts[0][key], actual[key][origin : origin + 129]):
            raise ValueError("zero-delay future differs from original actual control: " + key)
    if not failure_mode:
        with np.load(args.composed_complete.parent / "future.npz", allow_pickle=False) as old:
            for key in old.files:
                if not np.array_equal(forecasts[0][key], old[key]):
                    raise ValueError("zero-delay baseline differs from qualified future: " + key)
    check_pins(pins)
    clean(args.expected_head)
    write_once(
        args.output / "complete.json",
        dict(
            status="THREE_MEASURED_SWITCH_PREVIEWS_RECORDED_NOT_ACTUAL_GAIN",
            source_head=args.expected_head,
            public_baseline_and_teacher_failed=failure_mode,
            input_file_hashes=pins,
            scores=scores,
            forecasts=forecaster.counters(),
            actual_worlds=1,
            actual_native_steps=row["native_steps"],
            original_actual_and_zero_delay_future_unchanged=True,
            actual_switch_selection_executed=False,
            task_effectiveness_proven=False,
            neural_optimizer_updates=0,
            goal_complete=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
