"""Preregistered 4-state SIM neural-exit pilot; no teacher or learning claim."""

import argparse
import json
from pathlib import Path

INDICES = (0, 1, 2, 30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("champion-root", "champion-audit", "contact-readback", "backbone-model", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("champion-audit-hash", "contact-readback-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mode", choices=("pilot", "baseline-diagnostic"), default="pilot")
    parser.add_argument("--protect-passing-entry", action="store_true")
    parser.add_argument("--exit-audit", type=Path)
    parser.add_argument("--exit-audit-hash")
    a = parser.parse_args()
    indices = (0,) if a.mode == "baseline-diagnostic" else INDICES
    arms = ("B",) if a.mode == "baseline-diagnostic" else ("B", "C")
    world_ceiling = len(indices) * len(arms)
    native_ceiling = world_ceiling * 3600
    forecast_ceiling = world_ceiling * 37 * 1280
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_stable_plastic_sources import merge_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("new bounded pilot root required")
    pins = {}
    if a.protect_passing_entry:
        if a.mode != "pilot" or a.exit_audit is None or not a.exit_audit_hash:
            raise ValueError("protected variant requires exact completed unconditional exit audit")
        if digest(a.exit_audit) != a.exit_audit_hash:
            raise ValueError("exit audit identity differs")
        exit_audit = json.loads(a.exit_audit.read_text())
        if (
            exit_audit["status"] != "NEURAL_EXIT_RAW_HISTORY_AND_CPU_NN_VERIFIED_NOT_TEACHER"
            or exit_audit["rescues"] != [1]
            or exit_audit["regressions"] != [0]
            or exit_audit["qualified_for_expansion"] is not False
        ):
            raise ValueError(
                "same independently verified partial gain and retention defect required"
            )
        merge_pins(pins, exit_audit["input_file_hashes"])
        pins[str(a.exit_audit)] = a.exit_audit_hash
    elif a.exit_audit is not None or a.exit_audit_hash:
        raise ValueError("exit audit is only for the explicit protected variant")
    for path, expected, status in (
        (
            a.champion_audit,
            a.champion_audit_hash,
            "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED",
        ),
        (
            a.contact_readback,
            a.contact_readback_hash,
            "AUDITED_MEMORY_FAILURE_FORCE_TIMING_AND_FK_READBACK_NOT_NEW_SKILL",
        ),
    ):
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact independently bound development evidence required")
        report = json.loads(path.read_text())
        if report["status"] != status:
            raise ValueError("wrong prerequisite scope")
        if path == a.champion_audit and (
            report.get("perturbed_public_validation", False) or report["actual_worlds"] != 80
        ):
            raise ValueError("consumed nominal40 champion required")
        merge_pins(pins, report["input_file_hashes"])
        pins[str(path)] = expected
    cp = a.champion_root / "complete.json"
    if pins.get(str(cp)) != digest(cp):
        raise ValueError("same bound original champion required")
    champion = json.loads(cp.read_text())
    source_pins(pins)
    check_pins(pins)
    from scripts.rsi_teacher_physics_context import load_teacher_context

    context = load_teacher_context(pins, a.backbone_model)
    import numpy as np

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
    from rosclaw_soccer.rsi.synergy_forecast_kick_player import SynergyForecastKickPlayer
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    params = VelocityPolicyParameters.load(
        context["assets"]["velocity_config"],
        expected_hash=context["assets"]["velocity_config_hash"],
    )
    records, codec_errors = {}, {}
    # Qualify every known baseline target's representation BEFORE new physics.
    # Candidate prefix is required exact; post-exit targets use raw velocity action.
    for index in indices:
        reference = champion["results"][index]["arms"]["C"]
        path = a.champion_root / reference["record"]
        if pins.get(str(path)) != reference["hash"] or digest(path) != reference["hash"]:
            raise ValueError("exact champion course record required")
        old = json.loads(path.read_text())
        raw_path = path.parent / old["artifact"]["path"]
        if pins.get(str(raw_path)) != digest(raw_path):
            raise ValueError("bound baseline raw trajectory required")
        with np.load(raw_path, allow_pickle=False) as raw:
            errors = []
            for target in raw["joint_targets"]:
                encoded = (
                    (target - np.asarray(params.action_offset, np.float64))
                    / np.asarray(params.action_scale, np.float64)
                ).astype(np.float32)
                errors.append(float(np.max(abs(target29(encoded, params) - target))))
        if max(errors) > 5e-7:
            raise ValueError("baseline target history not representable; no physics allocation")
        records[index] = (old, raw_path)
        codec_errors[index] = max(errors)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.observed_neural_exit_pilot.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_indices=list(indices),
            mode=a.mode,
            protect_passing_entry_prediction=a.protect_passing_entry,
            old_success_controls=[0, 30],
            late_recontact_state=1,
            same_control_nonfoot_negative_control=2,
            actual_world_ceiling=world_ceiling,
            actual_native_step_ceiling=native_ceiling,
            forecast_requested_native_step_ceiling=forecast_ceiling,
            codec_preflight_errors_rad=codec_errors,
            trigger="completed_prior_control_foot_force_gt1N",
            exit_intent="zero_body_velocity_original_0.05_rate_limit",
            outcome_contract="original_H300_and_H360_body_diagnostic_unchanged",
            development_states_consumed=True,
            no_new_neural_weights=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    results, worlds, native, forecast = [], 0, 0, 0
    for index in indices:
        old, old_raw_path = records[index]
        folder = a.output / f"course-{index:02d}"
        folder.mkdir()
        pair = {}
        baseline_arrays = None
        baseline_events = None
        for arm in arms:
            instances = []

            class Controlled(SynergyForecastKickPlayer):
                basis_hg = context["basis"]
                forecast_controls = 128

                def __init__(self, *args, _instances=instances, **kwargs):
                    super().__init__(*args, **kwargs)
                    _instances.append(self)

            row, arrays = evaluate_episode(
                model=context["model"],
                assets=dict(context["assets"], experimental_fast=old["experimental_fast_approach"]),
                policy_types=context["types"],
                course=tuple(old["course"]),
                parameters=old["parameters"],
                hard_limits=context["hard_limits"],
                player_type=Controlled,
                post_contact_exit=arm == "C",
                protect_passing_entry_prediction=a.protect_passing_entry and arm == "C",
            )
            worlds += 1
            native += row["native_steps"]
            controller = instances[0] if instances else None
            counters = controller.forecaster.counters() if controller else {}
            forecast += counters.get("attempted_native_steps", 0)
            events = controller.events if controller else []
            # Persist the charged actual world BEFORE assertions. A rejected
            # comparison must not discard its diagnostic physical trajectory.
            artifact = folder / f"{arm}.npz"
            np.savez_compressed(artifact, **arrays)
            row.update(
                course_index=index,
                arm=arm,
                artifact=dict(path=artifact.name, hash=digest(artifact)),
                controller_events=events,
                forecast_counters=counters,
                old_planner_forecasts_do_not_model_new_exit=True,
                physical_teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            write_once(folder / f"{arm}-executed-unvalidated.json", row)
            if arm == "B":
                with np.load(old_raw_path, allow_pickle=False) as expected:
                    if set(expected.files) != set(arrays) or any(
                        not np.array_equal(expected[k], v) for k, v in arrays.items()
                    ):
                        raise ValueError("actual baseline differs from audited champion trajectory")
                baseline_arrays, baseline_events = arrays, events
            else:
                assert baseline_arrays is not None
                boundary = row["exit_control"]
                prefix = row["controls"] if boundary is None else boundary
                for key, previous in baseline_arrays.items():
                    n = prefix * 10 if key in ("qpos", "qvel", "torque", "ball_forces") else prefix
                    if not np.array_equal(previous[:n], arrays[key][:n]):
                        raise ValueError("candidate prefix changed before observed exit: " + key)
                if events != baseline_events:
                    raise ValueError("entry planner changed before post-contact intervention")
            if worlds > world_ceiling or native > native_ceiling or forecast > forecast_ceiling:
                raise ValueError("pilot budget exceeded")
            record = folder / f"{arm}.json"
            write_once(record, row)
            pair[arm] = dict(
                path=str(record.relative_to(a.output)),
                hash=digest(record),
                task_pass=row["outcome"]["task_pass"],
                healthy=row["healthy_full_horizon"],
                exit_control=row.get("exit_control"),
            )
        results.append(dict(course_index=index, arms=pair))
        write_once(a.output / f"completed-{index:02d}.json", results[-1])
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="OBSERVED_NEURAL_EXIT_PILOT_COMPLETE_NOT_TEACHER_QUALIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            actual_worlds=worlds,
            actual_native_steps=native,
            forecast_requested_native_steps=forecast,
            neural_optimizer_updates=0,
            independently_audited=False,
            qualified_for_expansion=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
