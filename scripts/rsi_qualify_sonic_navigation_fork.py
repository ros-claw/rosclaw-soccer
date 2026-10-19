"""Bounded own-neural-prefix/forecast/negative-control/actual-suffix experiment.

1800 NEW native steps:200 prefix +3x400 previews +400 actual negative suffix.
The old future is COMPARATOR_ONLY; no future target/state enters the controller.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "prior-review", "world", "model-root", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-prior-review-hash"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed source and unused output required")
    if digest(args.prior_review / "complete.json") != args.expected_prior_review_hash:
        raise ValueError("independent old physics proof differs")
    prior = json.loads((args.prior / "declaration.json").read_text())
    closed = json.loads((args.prior / "complete.json").read_text())
    review = json.loads((args.prior_review / "complete.json").read_text())
    if (
        prior["schema"] != "soccer.rsi.measured_ball_entry_teacher4.v1"
        or closed["actual_native_steps"] != 24000
        or review["additional_native_steps"] != 24000
        or any(any(x != 0 for x in row["maximum_errors"].values()) for row in review["results"])
    ):
        raise ValueError("whole old paired entry study proof required")
    for p, h in prior["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("old source/assets/evidence changed")
    if prior["input_file_hashes"].get(str(args.world)) != digest(args.world):
        raise ValueError("same compiled physical world required")
    raw_path = args.prior / "course0-ball_entry.npz"
    old = next(row for row in closed["results"] if row["job"] == "course0-ball_entry")
    if old["artifact_hash"] != digest(raw_path) or old["handoff_frame"] != 89:
        raise ValueError("fixed consumed source/context differs")
    paths = [
        p
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for root in (args.prior, args.prior_review) for p in root.iterdir() if p.is_file()]
    paths += [
        args.world,
        args.model_root / "planner_sonic.onnx",
        args.model_root / "low_latency/model_encoder.onnx",
        args.model_root / "low_latency/model_decoder.onnx",
        args.model_root / "low_latency/observation_config.yaml",
    ]
    pins = {str(p): digest(p) for p in paths}

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.sonic_navigation_forecast import (
        SonicNavigationForecaster,
        navigation_state_hash,
    )
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.sonic_navigation_fork_parity40.v1",
        source_head=head,
        input_file_hashes=pins,
        course=[0.35, -0.10, 0.0],
        prefix_frames=20,
        forecast_frames=40,
        forecast_branches=["greedy", "zero", "repeat_greedy_after_zero"],
        actual_suffix="zero",
        expected_new_native_steps=1800,
        expected_own_neural_control_ticks=180,
        borrowed_prefix_counted_as_new=False,
        old_future_is_comparator_only=True,
        optimizer_steps=0,
        official_task_exam=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    try:
        stored = json.loads((args.prior / "entry-affordance.json").read_text())
        prior_fields = {
            key: stored[key]
            for key in (
                "ball_offset_xy_m",
                "historical_frame",
                "historical_forward_speed_mps",
                "historical_artifact_hash",
            )
        }
        prior_fields["ball_offset_xy_m"] = tuple(prior_fields["ball_offset_xy_m"])
        affordance = SkillEntryAffordance(**prior_fields)
        if affordance.contract_hash != stored["contract_hash"]:
            raise ValueError("frozen historical affordance differs")
        with np.load(raw_path, allow_pickle=False) as archive:
            gold = {k: archive[k] for k in archive.files}
        nav = G1SonicNavigation(
            args.model_root,
            "tracker",
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        model = mujoco.MjModel.from_binary_path(str(args.world))
        body = G1TargetEpisode(
            model,
            world_hash=digest(args.world),
            kp=nav.backend.kp.copy(),
            kd=nav.backend.kd.copy(),
            course=(0.35, -0.10, 0.0),
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )

        def real_tick(frame, command):
            q, v = body.canonical()
            observation = TeamMotorObservation(
                agent_id="tracker",
                frame=frame,
                time_sec=float(body.data.time),
                intent="shoot",
                prospective_owner=True,
                qpos=tuple(float(x) for x in q),
                qvel=tuple(float(x) for x in v),
                target_position_m=(float(q[36]), float(q[37]), float(q[38])),
                navigation_command=command,
            )
            target = nav.propose(observation)
            action = nav.backend.action.copy()
            reference = nav.backend.reference[frame : frame + 10].copy()
            return (
                body.step(frame, np.asarray(target.target_rad, dtype=np.float64)),
                action,
                reference,
            )

        def greedy(q, v, frame):
            return ball_entry_decision(q, v, frame=frame, affordance=affordance).command

        prefix = []
        for frame in range(20):
            q, v = body.canonical()
            command = greedy(q, v, frame)
            physical, action, reference = real_tick(frame, command)
            for key, value in physical.items():
                np.testing.assert_array_equal(value[0], gold[key][frame])
            np.testing.assert_array_equal(action, gold["neural_action_isaac"][frame])
            np.testing.assert_array_equal(reference, gold["encoder_reference_poses"][frame])
            np.testing.assert_array_equal(command, gold["entry_command"][frame])
            prefix.append(physical)
        np.savez_compressed(
            args.output / "actual-prefix.npz",
            **{k: np.concatenate([row[k] for row in prefix]) for k in prefix[0]},
        )
        donor_body, donor_nav = integration_hash(model, body.data), navigation_state_hash(nav)
        forecaster = SonicNavigationForecaster(world_hash=digest(args.world), max_native_steps=1200)
        baseline = forecaster.forecast(nav, body, greedy)
        errors = {}
        for key, value in baseline.arrays.items():
            np.testing.assert_array_equal(value, gold[key][20:60])
            errors[key] = 0.0
        np.testing.assert_array_equal(baseline.neural_actions, gold["neural_action_isaac"][20:60])
        np.testing.assert_array_equal(
            baseline.reference_poses, gold["encoder_reference_poses"][20:60]
        )
        np.testing.assert_array_equal(baseline.commands, gold["entry_command"][20:60])
        zero = forecaster.forecast(nav, body, lambda q, v, frame: (0.0, 0.0, 0.0))
        repeated = forecaster.forecast(nav, body, greedy)
        for key in baseline.arrays:
            np.testing.assert_array_equal(repeated.arrays[key], baseline.arrays[key])
        np.testing.assert_array_equal(repeated.neural_actions, baseline.neural_actions)
        np.testing.assert_array_equal(repeated.reference_poses, baseline.reference_poses)
        changes = dict(
            reference_max_absolute=float(
                abs(zero.reference_poses - baseline.reference_poses).max()
            ),
            neural_action_max_absolute=float(
                abs(zero.neural_actions - baseline.neural_actions).max()
            ),
            root_xy_max_absolute=float(
                abs(
                    zero.arrays["canonical_qpos"][:, :2] - baseline.arrays["canonical_qpos"][:, :2]
                ).max()
            ),
        )
        if not all(value > 0 for value in changes.values()):
            raise ValueError(
                "negative command did not affect reference, neural action and physical root"
            )
        if (
            body.next_frame != 20
            or body.integrated_steps != 200
            or integration_hash(model, body.data) != donor_body
            or navigation_state_hash(nav) != donor_nav
        ):
            raise ValueError("forecasts mutated original actor/body")
        for name, result in (("greedy", baseline), ("zero", zero), ("repeat", repeated)):
            np.savez_compressed(
                args.output / (name + "-forecast.npz"),
                **result.arrays,
                neural_action_isaac=result.neural_actions,
                encoder_reference_poses=result.reference_poses,
                command=result.commands,
            )
        actual_suffix = []
        for frame in range(20, 60):
            physical, action, reference = real_tick(frame, (0.0, 0.0, 0.0))
            for key, value in physical.items():
                np.testing.assert_array_equal(value[0], zero.arrays[key][frame - 20])
            np.testing.assert_array_equal(action, zero.neural_actions[frame - 20])
            np.testing.assert_array_equal(reference, zero.reference_poses[frame - 20])
            actual_suffix.append(physical)
        np.savez_compressed(
            args.output / "actual-zero-suffix.npz",
            **{k: np.concatenate([row[k] for row in actual_suffix]) for k in actual_suffix[0]},
        )
        if (
            body.integrated_steps != 600
            or forecaster.integrated_native_steps != 1200
            or body.attempted_steps != 600
            or forecaster.attempted_native_steps != 1200
        ):
            raise ValueError("whole new-native accounting differs")
        for p, h in {**prior["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("source/assets/evidence changed during qualification")
        write_once(
            args.output / "complete.json",
            dict(
                maximum_old_future_physical_errors=errors,
                original_prefix_old_neural_and_physical_exact=True,
                prospective_greedy_matches_old_own_neural_future=True,
                repeat_greedy_after_negative_exact=True,
                negative_changes=changes,
                prospective_zero_matches_subsequent_original_body_neural_execution=True,
                donor_unchanged_before_actual_suffix=True,
                actual_native_steps=1800,
                actual_prefix_and_suffix_native_steps=600,
                new_forecast_native_steps=1200,
                borrowed_prefix_charged_as_new=False,
                new_forecast_planner_calls=[
                    r.new_planner_calls for r in (baseline, zero, repeated)
                ],
                neural_control_ticks=180,
                optimizer_steps=0,
                qualification_scope="one_consumed_g1_state_40frames_CPU_not_full_task_or_GPU",
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json", dict(error=repr(error), promotion_authorized=False)
        )
        raise


if __name__ == "__main__":
    main()
