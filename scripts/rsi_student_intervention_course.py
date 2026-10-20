"""Bounded nine-future privileged recovery from ACTUAL student frame30.

This teacher is a research oracle, not a deployable learned actor. No trajectory
state is copied into the actual body. Same original H300/post60/entry timeout.
"""


def run_course(
    args,
    item,
    world,
    poses,
    prior,
    student,
    muscle_backend,
    expected,
    evidence,
    pins,
    *,
    source_window=(286, 767),
    option_protocol="original9",
):
    import json

    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.bilateral_entry import (
        choose_bilateral_candidate,
        contact_foot_context,
        observed_contact_foot,
    )
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
    from rosclaw_soccer.rsi.full_entry_forecast import (
        FullEntryForecaster,
        feedback_entry_laws,
        run_entry_suffix,
        validate_source_window,
    )
    from rosclaw_soccer.rsi.source_lead_protocol import choose_zero_c_six, zero_c_six_options
    from rosclaw_soccer.rsi.student_intervention_score import (
        fuse_physical_intervention30,
        score_student_intervention30,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motor_student_rollout import exact_array
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_warm_student_intervention import warm_student30

    if option_protocol not in ("original9", "zero_c_six") or (
        option_protocol == "original9" and source_window != (286, 767)
    ):
        raise ValueError("declared original9 or zero-C six source protocol required")
    validate_source_window(source_window, "legacy", (0.0, 0.0, 0.0))
    folder = args.output / f"course{item.course_index:03d}"
    folder.mkdir()
    check_pins(pins)
    body = forecaster = None
    progress, candidates, actual_rows = {}, [], []
    try:
        nav = G1SonicNavigation(
            args.model_root,
            item.agent_id,
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        body = G1TargetEpisode(
            world,
            world_hash=digest(args.world),
            kp=muscle_backend.kp.copy(),
            kd=muscle_backend.kd.copy(),
            course=item.course,
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )
        student.calls = 0
        prefix, labels = warm_student30(
            student,
            FoundationMotorCodec(muscle_backend),
            nav,
            body,
            prior,
            expected,
            progress=progress,
        )
        for name, data in (("actual-student30.npz", prefix), ("shadow-labels30.npz", labels)):
            np.savez_compressed(folder / name, **data)
            with np.load(args.warm / folder.name / name, allow_pickle=False) as z:
                if set(z.files) != set(data):
                    raise ValueError("same qualified student30/history field identities required")
                for k, v in data.items():
                    exact_array(v, z[k])
        diagnostic = diagnose_entry_prefix(world, prefix, frames=30)
        if not (
            diagnostic["prefix_physical_safe"]
            and diagnostic["post_boundary_control_can_cause_first_touch"]
        ):
            row = dict(
                status="PREINTERVENTION_HISTORY_VETO",
                course_index=item.course_index,
                prefix_diagnostics=diagnostic,
                causal_qualified=False,
                candidates=[],
                selected=None,
                forecast_native_steps=0,
                executed_native_steps=body.integrated_steps,
                new_native_steps=body.integrated_steps,
                denominator_retained=True,
                automatic_retry_authorized=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            check_pins(pins)
            write_once(folder / "complete.json", row)
            print("STUDENT30_HISTORY_VETO", item.course_index, flush=True)
            return row
        laws = feedback_entry_laws()
        options = [
            (law, offset, "right")
            for law in (0, 3, 4)
            for offset in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025))
        ]
        options += [(law, (0.0, 0.0, 0.0), "left") for law in (0, 3, 4)]
        if option_protocol == "zero_c_six":
            options = list(zero_c_six_options())
        forecast_steps = len(options) * 2700
        forecaster = FullEntryForecaster(world_hash=digest(args.world), max_forecasts=len(options))
        for index, (law, offset, foot) in enumerate(options):
            source, affordance, context = contact_foot_context(poses, prior, foot)
            future = forecaster.forecast(
                nav,
                body,
                model_root=args.model_root,
                source_poses=source,
                source_evidence_hash=evidence,
                affordance=affordance,
                law=laws[law],
                contact_reference_offset_m=offset,
                start_frame=30,
                source_window=source_window,
            )
            if (
                future.new_integrated_native_steps != 2700
                or future.new_attempted_native_steps != 2700
            ):
                raise ValueError("exact actual30..299 future required")
            path = folder / f"candidate{index}.npz"
            np.savez_compressed(path, **future.arrays)
            joined = fuse_physical_intervention30(world, prefix, future.arrays)
            row = dict(
                index=index,
                law_index=law,
                offset_root_m=list(offset),
                contact_reference_foot=foot,
                contact_foot_context=context,
                report=future.report,
                score=score_student_intervention30(world, prefix, future.arrays),
                observed_contact_foot=observed_contact_foot(world, joined, global_start_frame=0),
                artifact=path.name,
                artifact_hash=digest(path),
                new_integrated_native_steps=2700,
                new_attempted_native_steps=2700,
                donor_body_hash=future.donor_body_hash,
                donor_navigation_hash=future.donor_navigation_hash,
            )
            candidates.append(row)
            write_once(folder / f"candidate{index}.json", row)
            print(
                "STUDENT30_RECOVERY_FUTURE",
                item.course_index,
                index,
                row["score"]["qualified"],
                flush=True,
            )
        selected = (
            choose_bilateral_candidate(candidates)
            if option_protocol == "original9"
            else choose_zero_c_six(candidates)
        )
        choice = candidates[selected]
        source, affordance, _ = contact_foot_context(poses, prior, choice["contact_reference_foot"])
        actual, report = run_entry_suffix(
            nav,
            body,
            model_root=args.model_root,
            source_poses=source,
            source_evidence_hash=evidence,
            affordance=affordance,
            law=laws[choice["law_index"]],
            contact_reference_offset_m=tuple(choice["offset_root_m"]),
            start_frame=30,
            source_window=source_window,
            step_observer=actual_rows.append,
        )
        if json.loads(json.dumps(report)) != json.loads(json.dumps(choice["report"])):
            raise ValueError("actual continuous endpoint report differs from own forecast")
        with np.load(folder / choice["artifact"], allow_pickle=False) as z:
            if set(z.files) != set(actual):
                raise ValueError("all18 teacher270 field identities required")
            for k, v in actual.items():
                exact_array(v, z[k])
        if (
            body.next_frame != 300
            or body.integrated_steps != 3000
            or body.attempted_steps != 3000
            or forecaster.integrated_native_steps != forecast_steps
            or forecaster.attempted_native_steps != forecast_steps
        ):
            raise ValueError("all actual300 + declared270 futures native steps must be counted")
        np.savez_compressed(folder / "selected-actual270.npz", **actual)
        joined = fuse_physical_intervention30(world, prefix, actual)
        np.savez_compressed(folder / "selected-physical300.npz", **joined)
        score = score_student_intervention30(world, prefix, actual)
        row = dict(
            status="STUDENT30_PRIVILEGED_RECOVERY_EXECUTED",
            course_index=item.course_index,
            prefix_diagnostics=diagnostic,
            candidates=candidates,
            selected=selected,
            selected_score=score,
            causal_qualified=score["qualified"],
            observed_contact_foot=observed_contact_foot(world, joined, global_start_frame=0),
            actual_all18_and_report_byte_exact_to_own_future=True,
            forecast_native_steps=forecast_steps,
            executed_native_steps=3000,
            new_native_steps=3000 + forecast_steps,
            actual_body_continuous_no_state_adoption=True,
            original_task_horizon=300,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        if option_protocol == "zero_c_six":
            row["option_protocol"] = option_protocol
            row["source_window"] = list(source_window)
            row["selected_report"] = report
        check_pins(pins)
        write_once(folder / "complete.json", row)
        print(
            "STUDENT30_RECOVERY_ACTUAL", item.course_index, selected, score["qualified"], flush=True
        )
        return row
    except Exception as error:
        for name, data in progress.items():
            if data:
                np.savez_compressed(
                    folder / ("partial-" + name + ".npz"),
                    **{k: np.stack([r[k] for r in data]) for k in data[0]},
                )
        if actual_rows:
            np.savez_compressed(
                folder / "partial-actual270.npz",
                **{k: np.concatenate([r[k] for r in actual_rows]) for k in actual_rows[0]},
            )
        write_once(
            folder / "failure.json",
            dict(
                error=repr(error),
                completed_candidates=len(candidates),
                actual_integrated_steps=body.integrated_steps if body else 0,
                actual_attempted_steps=body.attempted_steps if body else 0,
                forecast_integrated_steps=forecaster.integrated_native_steps if forecaster else 0,
                forecast_attempted_steps=forecaster.attempted_native_steps if forecaster else 0,
                automatic_retry_authorized=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise
