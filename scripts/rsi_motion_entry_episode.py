"""A predeclared source/entry intent executes from frame0, no future selection.

One complete native episode, not a warm-body forecast or learned student. The
six intent outcomes may be inspected OFFLINE, never fed into an acting policy.
"""

ARMS = {
    "original": ((286, 767), False),
    "compact_unadapted": ((466, 767), False),
    "compact_transported": ((466, 767), True),
}


def run_episode(
    a,
    course,
    world,
    poses,
    prior,
    *,
    arm,
    side,
    law_index,
    evidence,
    runtime,
    entry_calibration=None,
):
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import (
        feedback_entry_laws,
        nominal_prefix20,
        run_entry_suffix,
        score_full_entry,
    )
    from rosclaw_soccer.rsi.motion_entry_profile import build_motion_entry
    from rosclaw_soccer.rsi.recorded_entry_review import validate_recording
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        arm not in ARMS
        or side not in ("right", "left")
        or type(law_index) is not int
        or law_index not in (0, 3, 4)
    ):
        raise ValueError("predeclared three arms/three laws/two reference sides required")
    window, transport = ARMS[arm]
    check_pins(runtime)
    binding, entry, source = build_motion_entry(
        poses,
        prior,
        world_hash=digest(a.world),
        motion_artifact_hash=digest(a.motion_csv),
        reference_side=side,
        source_window=window,
        transport=transport,
    )
    if entry_calibration is not None:
        from rosclaw_soccer.rsi.native_entry_calibration import bind_native_entry

        if arm != "compact_unadapted" or side != "right":
            raise ValueError(
                "native calibration is only admitted for the declared right compact pilot"
            )
        binding, entry = bind_native_entry(binding, entry, entry_calibration)
    # This check precedes ANY physical body allocation or neural controller.
    binding.validate(
        source, entry, world_hash=digest(a.world), source_window=window, reference_side=side
    )
    folder = a.output / f"{side}-law{law_index}"
    folder.mkdir(parents=True, exist_ok=False)
    write_once(
        folder / "declaration.json",
        dict(
            arm=arm,
            reference_side=side,
            law_index=law_index,
            motion_entry_binding=binding.report(),
            motion_entry_binding_hash=binding.contract_hash,
            actual_teacher_reference_evidence_hash=evidence,
            known_before_navigation=True,
            maximum_native_steps=3000,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    body = None
    observed = []
    try:
        nav = G1SonicNavigation(
            a.model_root,
            course.agent_id,
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        body = G1TargetEpisode(
            world,
            world_hash=digest(a.world),
            kp=nav.backend.kp.copy(),
            kd=nav.backend.kd.copy(),
            course=course.course,
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )
        prefix = nominal_prefix20(nav, body, entry, step_observer=observed.append)
        np.savez_compressed(folder / "own-prefix20.npz", **prefix)
        observed.clear()
        suffix, report = run_entry_suffix(
            nav,
            body,
            model_root=a.model_root,
            source_poses=source,
            source_evidence_hash=evidence,
            affordance=entry,
            law=feedback_entry_laws()[law_index],
            contact_reference_offset_m=(0.0, 0.0, 0.0),
            source_window=window,
            step_observer=observed.append,
        )
        trace = {k: np.concatenate((prefix[k], suffix[k])) for k in prefix}
        validate_recording(trace, frames=300, nbody=world.nbody)
        if body.integrated_steps != 3000 or body.attempted_steps != 3000 or body.next_frame != 300:
            raise ValueError("real complete H300 native episode required")
        binding.validate(
            source, entry, world_hash=digest(a.world), source_window=window, reference_side=side
        )
        diagnostic = diagnose_entry_prefix(world, prefix)
        score = score_full_entry(world, suffix)
        score["original_suffix_qualified"] = score["qualified"]
        score["common_prefix_physical_safe"] = diagnostic["prefix_physical_safe"]
        if not diagnostic["prefix_physical_safe"]:
            score["qualified"], score["cost"], score["physical_prefix_and_suffix_safe"] = (
                False,
                None,
                False,
            )
        np.savez_compressed(folder / "native300.npz", **trace)
        outcome = dict(
            arm=arm,
            reference_side=side,
            law_index=law_index,
            source_window=list(window),
            motion_entry_binding_hash=binding.contract_hash,
            entry_affordance_hash=entry.contract_hash,
            actual_teacher_reference_evidence_hash=evidence,
            entry_ball_offset_xy_m=list(entry.ball_offset_xy_m),
            prefix_diagnostics=diagnostic,
            score=score,
            report=report,
            causal_qualified=bool(
                score["qualified"] and diagnostic["post_boundary_control_can_cause_first_touch"]
            ),
            observed_contact_foot=observed_contact_foot(world, trace, global_start_frame=0),
            native_hash=digest(folder / "native300.npz"),
            new_native_steps=3000,
            new_optimizer_steps=0,
            initial_state_adopted_from_other_episode=False,
            known_intent_before_frame0=True,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        check_pins(runtime)
        write_once(folder / "complete.json", outcome)
        print(
            "MOTION_ENTRY_EPISODE",
            arm,
            side,
            law_index,
            outcome["causal_qualified"],
            report["handoff_frame"],
            score["first_foot_contact_global_frame"],
            flush=True,
        )
        return outcome
    except Exception as error:
        if observed:
            np.savez_compressed(
                folder / "partial-observed-control.npz",
                **{k: np.concatenate([r[k] for r in observed]) for k in observed[0]},
            )
        write_once(
            folder / "failure.json",
            dict(
                error=repr(error),
                native_integrated_steps=body.integrated_steps if body else 0,
                native_attempted_steps=body.attempted_steps if body else 0,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise
