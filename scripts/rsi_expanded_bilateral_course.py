"""Fresh own-state nine-option teacher and paired native endpoints, research only."""


def run_course(
    args, item, model, poses, affordance, evidence_hash, runtime_pins, *, prefix_motor_provider=None
):
    from dataclasses import asdict

    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.sonic_navigation_forecast import (
        _clone_body,
        clone_navigation,
        navigation_state_hash,
    )
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
    from rosclaw_soccer.rsi.full_entry_forecast import (
        FullEntryForecaster,
        feedback_entry_laws,
        nominal_prefix20,
        run_entry_suffix,
        score_full_entry,
    )
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    folder = args.output / f"course{item.course_index:02d}"
    folder.mkdir(exist_ok=False)
    check_pins(runtime_pins)
    body = forecaster = current = None
    current_rows, candidates = [], []
    endpoint_steps = 0
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
            model,
            world_hash=digest(args.world),
            kp=nav.backend.kp.copy(),
            kd=nav.backend.kd.copy(),
            course=item.course,
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )
        current = body
        prefix = nominal_prefix20(
            nav,
            body,
            affordance,
            step_observer=current_rows.append,
            motor_provider=prefix_motor_provider,
        )
        prefix_diagnostics = diagnose_entry_prefix(model, prefix)
        endpoint_steps = 200
        np.savez_compressed(folder / "own-prefix20.npz", **prefix)
        current, current_rows = None, []
        forecaster = FullEntryForecaster(world_hash=digest(args.world), max_forecasts=9)
        laws = feedback_entry_laws()

        def score(suffix):
            value = score_full_entry(model, suffix)
            value["original_suffix_qualified"] = value["qualified"]
            value["common_prefix_physical_safe"] = prefix_diagnostics["prefix_physical_safe"]
            if not prefix_diagnostics["prefix_physical_safe"]:
                value["qualified"], value["cost"] = False, None
                value["physical_prefix_and_suffix_safe"] = False
            return value

        options = [
            (law, offset, "right")
            for law in (0, 3, 4)
            for offset in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025))
        ]
        options += [(law, (0.0, 0.0, 0.0), "left") for law in (0, 3, 4)]
        for index, (law, offset, foot) in enumerate(options):
            source, entry, context = contact_foot_context(poses, affordance, foot)
            future = forecaster.forecast(
                nav,
                body,
                model_root=args.model_root,
                source_poses=source,
                source_evidence_hash=evidence_hash,
                affordance=entry,
                law=laws[law],
                contact_reference_offset_m=offset,
            )
            if (
                future.new_integrated_native_steps != 2800
                or future.new_attempted_native_steps != 2800
            ):
                raise ValueError("full H300 own-state candidate required")
            artifact = folder / f"candidate{index}.npz"
            np.savez_compressed(artifact, **future.arrays)
            row = dict(
                index=index,
                law_index=law,
                offset_root_m=list(offset),
                contact_reference_foot=foot,
                contact_foot_context=context,
                report=future.report,
                score=score(future.arrays),
                observed_contact_foot=observed_contact_foot(
                    model, future.arrays, global_start_frame=20
                ),
                artifact=artifact.name,
                artifact_hash=digest(artifact),
                donor_body_hash=future.donor_body_hash,
                donor_navigation_hash=future.donor_navigation_hash,
                new_integrated_native_steps=2800,
                new_attempted_native_steps=2800,
            )
            candidates.append(row)
            write_once(folder / f"candidate{index}.json", row)
            print(
                "EXPANDED_ENTRY_FUTURE",
                item.course_index,
                index,
                row["score"]["qualified"],
                flush=True,
            )
        selected = choose_bilateral_candidate(candidates)
        eligible = [i for i, c in enumerate(candidates[:6]) if c["score"]["qualified"]]
        baseline = (
            min(eligible, key=lambda i: (candidates[i]["score"]["cost"], i)) if eligible else 0
        )
        if (
            forecaster.integrated_native_steps != forecaster.attempted_native_steps
            or forecaster.integrated_native_steps != 25200
        ):
            raise ValueError("all9 full futures must be counted")

        def execute(controller, actual_body, index, name):
            nonlocal current, current_rows, endpoint_steps
            row = candidates[index]
            source, entry, context = contact_foot_context(
                poses, affordance, row["contact_reference_foot"]
            )
            if context != row["contact_foot_context"]:
                raise ValueError("actual context must be reconstructed")
            current, current_rows = actual_body, []
            suffix, report = run_entry_suffix(
                controller,
                actual_body,
                model_root=args.model_root,
                source_poses=source,
                source_evidence_hash=evidence_hash,
                affordance=entry,
                law=laws[row["law_index"]],
                contact_reference_offset_m=tuple(row["offset_root_m"]),
                step_observer=current_rows.append,
            )
            if report != row["report"]:
                raise ValueError("actual report must match own forecast")
            with np.load(folder / row["artifact"], allow_pickle=False) as expected:
                for key, value in suffix.items():
                    np.testing.assert_array_equal(value, expected[key])
            if (
                actual_body.integrated_steps != actual_body.attempted_steps
                or actual_body.integrated_steps != 3000
            ):
                raise ValueError("full native endpoint required")
            endpoint_steps += 2800
            arrays = {k: np.concatenate((prefix[k], suffix[k])) for k in prefix}
            artifact = folder / (name + ".npz")
            np.savez_compressed(artifact, **arrays)
            outcome = dict(
                selected_index=index,
                score=score(suffix),
                report=report,
                intended_reference_foot=row["contact_reference_foot"],
                observed_contact_foot=observed_contact_foot(model, arrays, global_start_frame=0),
                artifact=artifact.name,
                artifact_hash=digest(artifact),
                all18_and_report_exact_to_own_future=True,
            )
            current, current_rows = None, []
            write_once(folder / (name + ".json"), outcome)
            return outcome

        if baseline != selected:
            donor_body, donor_nav = integration_hash(model, body.data), navigation_state_hash(nav)
            old = execute(clone_navigation(nav), _clone_body(body), baseline, "baseline-actual")
            if (
                integration_hash(model, body.data) != donor_body
                or navigation_state_hash(nav) != donor_nav
            ):
                raise ValueError("paired baseline execution changed selected actor donor")
        actual = execute(nav, body, selected, "selected-actual")
        if baseline == selected:
            old = actual
        oscore, nscore = old["score"], actual["score"]
        result = dict(
            course=asdict(item),
            candidates=candidates,
            baseline=old,
            selected=actual,
            paired_endpoints_share_prefix=True,
            prefix_diagnostics=prefix_diagnostics,
            shared_endpoint=baseline == selected,
            forecast_native_steps=25200,
            endpoint_native_steps=endpoint_steps,
            old_teacher_success_lost=oscore["qualified"] and not nscore["qualified"],
            old_teacher_failure_rescued=not oscore["qualified"] and nscore["qualified"],
            causal_first_touch_rescue=(
                not oscore["qualified"]
                and nscore["qualified"]
                and prefix_diagnostics["post_boundary_control_can_cause_first_touch"]
            ),
            added_unsafe_vs_old_teacher=oscore["physical_prefix_and_suffix_safe"]
            and not nscore["physical_prefix_and_suffix_safe"],
            neural_optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        check_pins(runtime_pins)
        write_once(folder / "complete.json", result)
        print(
            "EXPANDED_ENTRY_ACTUAL",
            item.course_index,
            baseline,
            selected,
            nscore["qualified"],
            flush=True,
        )
        return result
    except Exception as error:
        if current_rows:
            np.savez_compressed(
                folder / "incomplete-observed-control.npz",
                **{k: np.concatenate([r[k] for r in current_rows]) for k in current_rows[0]},
            )
        if current is not None:
            np.savez_compressed(
                folder / "last-native-state.npz",
                qpos=current.data.qpos.copy(),
                qvel=current.data.qvel.copy(),
                time_sec=np.array([current.data.time]),
            )
        write_once(
            folder / "failure.json",
            dict(
                error=repr(error),
                complete_candidates=len(candidates),
                completed_endpoint_steps=endpoint_steps,
                forecast_attempted_steps=forecaster.attempted_native_steps if forecaster else 0,
                forecast_integrated_steps=forecaster.integrated_native_steps if forecaster else 0,
                incomplete_endpoint_steps=max(0, current.integrated_steps - 200)
                if current is not None
                else 0,
                current_body_attempted_steps=current.attempted_steps if current is not None else 0,
                current_body_integrated_steps=current.integrated_steps
                if current is not None
                else 0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise
