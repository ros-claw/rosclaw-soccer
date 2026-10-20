"""Four consumed courses, actual nominal vs full remaining-task teacher.

At most seven forecasts from actual frame20 per teacher body; no receding horizon,
state adoption or new learned policy. All failures/candidates retained.
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
    for name in (
        "prior",
        "prior-review",
        "failed-entry-study",
        "failed-entry-review",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-prior-review-hash",
        "expected-failed-entry-review-hash",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--law-family", choices=("held20", "feedback", "coupled"), default="held20")
    parser.add_argument("--preceding-full-study", type=Path)
    parser.add_argument("--preceding-full-review", type=Path)
    parser.add_argument("--expected-preceding-full-review-hash")
    parser.add_argument("--contact-prior", type=Path)
    parser.add_argument("--contact-prior-review", type=Path)
    parser.add_argument("--expected-contact-prior-review-hash")
    args = parser.parse_args()
    source = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed source and unused output required")
    if (
        digest(args.prior_review / "complete.json") != args.expected_prior_review_hash
        or digest(args.failed_entry_review / "complete.json")
        != args.expected_failed_entry_review_hash
    ):
        raise ValueError("independent preceding study proofs differ")
    prior = json.loads((args.prior / "declaration.json").read_text())
    old = json.loads((args.prior / "complete.json").read_text())
    failed = json.loads((args.failed_entry_study / "complete.json").read_text())
    failed_declaration = json.loads((args.failed_entry_study / "declaration.json").read_text())
    reviewed = json.loads((args.failed_entry_review / "complete.json").read_text())
    if (
        prior["schema"] != "soccer.rsi.measured_ball_entry_teacher4.v1"
        or old["actual_native_steps"] != 24000
        or failed["total_new_native_steps"] != 55200
        or reviewed["additional_total_native_steps"] != 55200
        or len(reviewed["results"]) != 8
        or not all(r["actual_and_all_candidate_physics_bit_exact"] for r in reviewed["results"])
        or any(
            (p / "failure.json").exists()
            for p in (
                args.prior,
                args.prior_review,
                args.failed_entry_study,
                args.failed_entry_review,
            )
        )
    ):
        raise ValueError("whole preserved preceding physics studies required")
    historical_declarations = [prior, failed_declaration]
    preceding_roots = []
    if args.law_family in ("feedback", "coupled"):
        if (
            args.preceding_full_study is None
            or args.preceding_full_review is None
            or args.expected_preceding_full_review_hash is None
        ):
            raise ValueError("feedback successor requires the closed full-task predecessor review")
        if (
            digest(args.preceding_full_review / "complete.json")
            != args.expected_preceding_full_review_hash
        ):
            raise ValueError("full-task predecessor review identity differs")
        pc = json.loads((args.preceding_full_study / "complete.json").read_text())
        pr = json.loads((args.preceding_full_review / "complete.json").read_text())
        pd = json.loads((args.preceding_full_study / "declaration.json").read_text())
        rd = json.loads((args.preceding_full_review / "declaration.json").read_text())
        preceding_native = 102400 if args.law_family == "feedback" else 91200
        preceding_schema = (
            "soccer.rsi.full_remaining_entry_teacher4.v1"
            if args.law_family == "feedback"
            else "soccer.rsi.full_remaining_entry_teacher4.v2"
        )
        if (
            pc["total_new_native_steps"] != preceding_native
            or pr["additional_total_native_steps"] != preceding_native
            or len(pr["results"]) != 8
            or pd["schema"] != preceding_schema
            or not all(
                r["actual_physics_bit_exact"] and r["all_candidate_physics_bit_exact"]
                for r in pr["results"]
            )
            or rd["input_file_hashes"].get(str(args.preceding_full_study / "complete.json"))
            != digest(args.preceding_full_study / "complete.json")
            or any(
                (p / "failure.json").exists()
                for p in (args.preceding_full_study, args.preceding_full_review)
            )
        ):
            raise ValueError("whole authentic full-task predecessor required before successor")
        historical_declarations += [pd, rd]
        preceding_roots = [args.preceding_full_study, args.preceding_full_review]
    elif any(
        x is not None
        for x in (
            args.preceding_full_study,
            args.preceding_full_review,
            args.expected_preceding_full_review_hash,
        )
    ):
        raise ValueError(
            "predecessor arguments are only for the explicitly declared feedback successor"
        )
    if args.law_family == "coupled":
        if (
            args.contact_prior is None
            or args.contact_prior_review is None
            or args.expected_contact_prior_review_hash is None
        ):
            raise ValueError("coupled study requires authenticated contact-authority evidence")
        if (
            digest(args.contact_prior_review / "complete.json")
            != args.expected_contact_prior_review_hash
        ):
            raise ValueError("contact-authority independent proof differs")
        cc = json.loads((args.contact_prior / "complete.json").read_text())
        cr = json.loads((args.contact_prior_review / "complete.json").read_text())
        cd = json.loads((args.contact_prior / "declaration.json").read_text())
        dr = json.loads((args.contact_prior_review / "declaration.json").read_text())
        if (
            cc["actual_native_steps"] != 12000
            or cr["additional_native_steps"] != 12000
            or cd["schema"] != "soccer.rsi.contact_reference_offset_pair2.v1"
            or len(cr["results"]) != 4
            or not all(
                r["all_native_arrays_bit_exact"]
                and r["handoff_and_modified_reference_reconstructed"]
                for r in cr["results"]
            )
            or dr["input_file_hashes"].get(str(args.contact_prior / "complete.json"))
            != digest(args.contact_prior / "complete.json")
            or cd["input_file_hashes"].get(str(args.preceding_full_study / "complete.json"))
            != digest(args.preceding_full_study / "complete.json")
            or any(
                (p / "failure.json").exists()
                for p in (args.contact_prior, args.contact_prior_review)
            )
        ):
            raise ValueError("whole contact-authority proof bound to feedback predecessor required")
        historical_declarations += [cd, dr]
        preceding_roots += [args.contact_prior, args.contact_prior_review]
    elif any(
        x is not None
        for x in (
            args.contact_prior,
            args.contact_prior_review,
            args.expected_contact_prior_review_hash,
        )
    ):
        raise ValueError("contact predecessor is only for the coupled mechanism")
    for declaration in historical_declarations:
        for p, h in declaration["input_file_hashes"].items():
            if digest(Path(p)) != h:
                raise ValueError("historical inputs changed")
    for path in (args.world, args.motion_csv):
        if prior["input_file_hashes"].get(str(path)) != digest(path):
            raise ValueError("compiled world/source identity changed")
    for row in old["results"]:
        if digest(args.prior / (row["job"] + ".npz")) != row["artifact_hash"]:
            raise ValueError("historical raw evidence changed")
    paths = [
        p
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [
        p
        for root in [
            args.prior,
            args.prior_review,
            args.failed_entry_study,
            args.failed_entry_review,
        ]
        + preceding_roots
        for p in root.iterdir()
        if p.is_file()
    ]
    paths += [
        args.world,
        args.motion_csv,
        args.model_root / "planner_sonic.onnx",
        args.model_root / "low_latency/model_encoder.onnx",
        args.model_root / "low_latency/model_decoder.onnx",
        args.model_root / "low_latency/observation_config.yaml",
    ]
    pins = {str(p): digest(p) for p in paths}

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
    from rosclaw_soccer.physics.sonic_navigation_forecast import navigation_state_hash
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.contact_reference_offset import offset_contact_reference
    from rosclaw_soccer.rsi.full_entry_forecast import (
        FullEntryForecaster,
        body_fields_hash,
        coupled_entry_options,
        feedback_entry_laws,
        full_entry_laws,
        nominal_prefix20,
        run_entry_suffix,
        score_full_entry,
    )
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    candidate_count = 7 if args.law_family == "held20" else 6
    planned_forecast = 4 * candidate_count * 2800
    declaration = dict(
        schema="soccer.rsi.full_remaining_entry_teacher4.v1"
        if args.law_family == "held20"
        else "soccer.rsi.full_remaining_entry_teacher4.v2"
        if args.law_family == "feedback"
        else "soccer.rsi.full_remaining_entry_teacher4.v3",
        law_family=args.law_family,
        source_head=head,
        input_file_hashes=pins,
        courses=prior["courses"],
        arms=["nominal", "full_teacher"],
        reused_courses=4,
        independent_new_courses=0,
        actual_bodies=8,
        planned_actual_native_steps=24000,
        planned_forecast_native_steps=planned_forecast,
        actual_horizon=300,
        prefix_frames=20,
        forecasts_per_teacher_body=candidate_count,
        source_interval=[286, 767],
        entry_intervals=50,
        earliest_handoff=30,
        latest_handoff=90,
        source_reference_after_handoff_not_retargeted=args.law_family != "coupled",
        reference_request_conditioned_at_handoff=args.law_family == "coupled",
        reference_tracks_ball_after_handoff=False,
        contact_reference_options_root_m=[[0.0, 0.0, 0.0], [0.0, -0.025, -0.025]]
        if args.law_family == "coupled"
        else [[0.0, 0.0, 0.0]],
        search_only_at_frame=20,
        selected_command_law_held_window=[20, 40] if args.law_family == "held20" else None,
        candidate0="causal_per_frame_nominal_feedback",
        candidate1_to6="six_fixed_commands20ticks_then_causal_nominal_feedback"
        if args.law_family == "held20"
        else "fixed_six_laws_recomputed_on_own_current_state_until_handoff_or_timeout"
        if args.law_family == "feedback"
        else "three_existing_feedback_laws_x_two_contact_reference_requests",
        all_actual_and_predicted_steps_have_own_neural_inference=True,
        actual_never_adopts_predicted_state_targets_or_reference=True,
        hard_veto=dict(nonfoot_force_over_n=1.0, minimum_pelvis_z_m=0.65, minimum_upright=0.9),
        research_eligibility=dict(
            foot_contact_by_frame=239,
            forward_to_H300_m=0.5,
            absolute_lateral_to_H300_max_m=0.3,
            maximum_forward_velocity_mps=0.5,
        ),
        research_cost="lateral2+.02*late60rootSpeed2+.02*postTouchMaxRootDrift2-.01*min(forward,3)",
        no_eligible_fallback="nominal_feedback_uncertified_NOT_success",
        consumed_development=True,
        official_football_exam=False,
        optimizer_steps=0,
        actor_updated=False,
        activation_ceiling="SIM_ONLY",
        publicity_license_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    current_body = current_forecaster = None
    completed_actual_native_steps = total_forecast = 0
    try:
        stored = json.loads((args.prior / "entry-affordance.json").read_text())
        fields = {
            k: stored[k]
            for k in (
                "ball_offset_xy_m",
                "historical_frame",
                "historical_forward_speed_mps",
                "historical_artifact_hash",
            )
        }
        fields["ball_offset_xy_m"] = tuple(fields["ball_offset_xy_m"])
        affordance = SkillEntryAffordance(**fields)
        if affordance.contract_hash != stored["contract_hash"]:
            raise ValueError("past own-body affordance identity differs")
        write_once(args.output / "entry-affordance.json", stored)
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        geometry_admissions = []
        if args.law_family == "coupled":
            # C does not change the path to handoff: independently authenticated
            # historical entry measurements admit all reference options first.
            geometry_model = mujoco.MjModel.from_binary_path(str(args.world))
            for i in range(4):
                parent = next(r for r in pc["results"] if r["job"] == f"course{i}-full_teacher")
                for law_index in (0, 3, 4):
                    old_candidate = parent["candidates"][law_index]
                    artifact = args.preceding_full_study / old_candidate["artifact"]
                    if digest(artifact) != old_candidate["artifact_hash"]:
                        raise ValueError("feedback candidate source differs")
                    handoff = old_candidate["report"]["handoff_frame"]
                    if handoff is None:
                        geometry_admissions.append(
                            dict(
                                course_index=i,
                                law_index=law_index,
                                no_handoff_reference_not_applicable=True,
                                new_native_steps=0,
                            )
                        )
                        continue
                    with np.load(artifact, allow_pickle=False) as archive:
                        entry = archive["canonical_qpos"][handoff - 20, :36].copy()
                    held, _ = quarantine_wrist_reference(poses, entry)
                    prepared = prepare_sonic_reference(
                        held[286:767],
                        entry,
                        source_frequency_hz=120,
                        entry_intervals=50,
                        source_evidence_hash=hash_json(declaration),
                    )
                    for offset in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025)):
                        modified, geometry = offset_contact_reference(
                            geometry_model, prepared.reference, offset_root_m=offset
                        )
                        modified.initialize(entry, frames=410)
                        geometry_admissions.append(
                            dict(
                                course_index=i,
                                law_index=law_index,
                                offset_root_m=list(offset),
                                geometry=geometry,
                                new_native_steps=0,
                            )
                        )
            write_once(
                args.output / "geometry-admission.json",
                dict(results=geometry_admissions, new_native_steps=0, teacher_qualified=False),
            )
        results, total_forecast = [], 0
        for i, course in enumerate(declaration["courses"]):
            with np.load(args.prior / f"course{i}-ball_entry.npz", allow_pickle=False) as archive:
                gold = {k: archive[k] for k in archive.files}
            for arm in declaration["arms"]:
                name = f"course{i}-{arm}"
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
                dimensions = inspect_native_ball_dimensions(
                    model, geom_id=int(model.geom("ball_geom").id)
                )
                if dimensions.radius_m != 0.11 or dimensions.body_mass_kg != 0.43:
                    raise ValueError("historical sphere dimensions changed")
                body = G1TargetEpisode(
                    model,
                    world_hash=digest(args.world),
                    kp=nav.backend.kp.copy(),
                    kd=nav.backend.kd.copy(),
                    course=tuple(course),
                    defaults=G1SonicRunupController.default_angles.copy(),
                    record_all_ball_contacts=True,
                )
                current_body = body
                prefix = nominal_prefix20(nav, body, affordance)
                if set(prefix) != set(gold):
                    raise ValueError("full numeric evidence coverage differs")
                for key in prefix:
                    np.testing.assert_array_equal(prefix[key], gold[key][:20])
                q, v = body.canonical()
                laws = (
                    full_entry_laws(q, v, affordance=affordance)
                    if args.law_family == "held20"
                    else feedback_entry_laws()
                )
                offsets = [(0.0, 0.0, 0.0)] * len(laws)
                names = [law.name for law in laws]
                if args.law_family == "coupled":
                    options = coupled_entry_options()
                    laws = [o.law for o in options]
                    offsets = [o.offset_root_m for o in options]
                    names = [o.name for o in options]
                if len(laws) != candidate_count:
                    raise ValueError("declared command-law bank coverage differs")
                candidates, selected, selected_future, forecast_steps = [], 0, None, 0
                forecaster = FullEntryForecaster(world_hash=digest(args.world))
                current_forecaster = forecaster
                donor = (
                    integration_hash(model, body.data),
                    navigation_state_hash(nav),
                    body_fields_hash(body),
                )
                if arm == "full_teacher":
                    for index, law in enumerate(laws):
                        forecast = forecaster.forecast(
                            nav,
                            body,
                            model_root=args.model_root,
                            source_poses=poses,
                            source_evidence_hash=hash_json(declaration),
                            affordance=affordance,
                            law=law,
                            contact_reference_offset_m=offsets[index],
                        )
                        if index == 0:
                            for key in forecast.arrays:
                                np.testing.assert_array_equal(forecast.arrays[key], gold[key][20:])
                        if args.law_family == "coupled" and not any(offsets[index]):
                            parent = next(
                                r for r in pc["results"] if r["job"] == f"course{i}-full_teacher"
                            )
                            ancestor = parent["candidates"][law.feedback_index]
                            with np.load(
                                args.preceding_full_study / ancestor["artifact"], allow_pickle=False
                            ) as archive:
                                for key in forecast.arrays:
                                    np.testing.assert_array_equal(
                                        forecast.arrays[key], archive[key]
                                    )
                        if args.law_family == "coupled":
                            parent = next(
                                r for r in pc["results"] if r["job"] == f"course{i}-full_teacher"
                            )
                            historical_handoff = parent["candidates"][law.feedback_index]["report"][
                                "handoff_frame"
                            ]
                            if forecast.report["handoff_frame"] != historical_handoff:
                                raise ValueError("contact request changed the original entry path")
                            if historical_handoff is not None:
                                expected_geometry = next(
                                    g["geometry"]
                                    for g in geometry_admissions
                                    if g["course_index"] == i
                                    and g["law_index"] == law.feedback_index
                                    and g.get("offset_root_m") == list(offsets[index])
                                )
                                if (
                                    forecast.report["contact_reference_adaptation"]
                                    != expected_geometry
                                ):
                                    raise ValueError(
                                        "own forecast entry geometry differs from preflight"
                                    )
                        artifact = args.output / f"{name}-candidate{index}.npz"
                        np.savez_compressed(artifact, **forecast.arrays)
                        row = dict(
                            index=index,
                            name=names[index],
                            contact_reference_offset_m=list(offsets[index]),
                            law=forecast.report["law"],
                            report=forecast.report,
                            score=score_full_entry(model, forecast.arrays),
                            artifact=artifact.name,
                            artifact_hash=digest(artifact),
                            donor_body_hash=forecast.donor_body_hash,
                            donor_navigation_hash=forecast.donor_navigation_hash,
                            new_attempted_native_steps=forecast.new_attempted_native_steps,
                            new_integrated_native_steps=forecast.new_integrated_native_steps,
                            nominal_future_bit_exact_to_historical=index == 0,
                        )
                        candidates.append(row)
                        write_once(args.output / f"{name}-candidate{index}.json", row)
                        print(
                            "FULL_ENTRY_CANDIDATE",
                            name,
                            index,
                            names[index],
                            row["score"],
                            flush=True,
                        )
                    eligible = [r["index"] for r in candidates if r["score"]["qualified"]]
                    selected = (
                        min(eligible, key=lambda j: candidates[j]["score"]["cost"])
                        if eligible
                        else 0
                    )
                    with np.load(
                        args.output / candidates[selected]["artifact"], allow_pickle=False
                    ) as archive:
                        selected_future = {k: archive[k] for k in archive.files}
                    forecast_steps = forecaster.integrated_native_steps
                    if (
                        forecast_steps != candidate_count * 2800
                        or forecaster.attempted_native_steps != candidate_count * 2800
                        or sum(r["new_integrated_native_steps"] for r in candidates)
                        != candidate_count * 2800
                    ):
                        raise ValueError("full remaining-task forecast accounting differs")
                if donor != (
                    integration_hash(model, body.data),
                    navigation_state_hash(nav),
                    body_fields_hash(body),
                ):
                    raise ValueError("search changed actual neural or physical state")
                suffix, report = run_entry_suffix(
                    nav,
                    body,
                    model_root=args.model_root,
                    source_poses=poses,
                    source_evidence_hash=hash_json(declaration),
                    affordance=affordance,
                    law=laws[selected],
                    contact_reference_offset_m=offsets[selected],
                )
                if selected_future is not None:
                    for key in suffix:
                        np.testing.assert_array_equal(suffix[key], selected_future[key])
                    if report != candidates[selected]["report"]:
                        raise ValueError("whole prospective/actual handoff reports differ")
                arrays = {key: np.concatenate((prefix[key], suffix[key])) for key in prefix}
                if arm == "nominal":
                    for key in arrays:
                        np.testing.assert_array_equal(arrays[key], gold[key])
                artifact = args.output / (name + ".npz")
                np.savez_compressed(artifact, **arrays)
                report.update(
                    job=name,
                    course=list(course),
                    arm=arm,
                    artifact_hash=digest(artifact),
                    candidates=candidates,
                    selected_index=selected,
                    selected_name=names[selected],
                    selected_contact_reference_offset_m=list(offsets[selected]),
                    actual_score=score_full_entry(model, suffix),
                    no_eligible_candidate=bool(
                        candidates and not any(r["score"]["qualified"] for r in candidates)
                    ),
                    actual_native_steps=3000,
                    forecast_native_steps=forecast_steps,
                    historical_prefix_all_arrays_bit_exact=True,
                    historical_nominal_all_arrays_bit_exact=arm == "nominal",
                    selected_whole_remaining_future_all_arrays_bit_exact=selected_future
                    is not None,
                    strike=summarize_reference_ball_strike(model, arrays),
                )
                results.append(report)
                total_forecast += forecast_steps
                write_once(args.output / (name + "-complete.json"), report)
                print(
                    "FULL_ENTRY_ACTUAL_CLOSED",
                    name,
                    report["selected_name"],
                    report["actual_score"],
                    flush=True,
                )
                completed_actual_native_steps += 3000
                current_body = current_forecaster = None
        final_pins = dict(pins)
        for historical in historical_declarations:
            final_pins.update(historical["input_file_hashes"])
        for p, h in final_pins.items():
            if digest(Path(p)) != h:
                raise ValueError("sealed source/assets/evidence changed during experiment")
        if total_forecast != planned_forecast:
            raise ValueError("full paired experiment forecast accounting differs")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                actual_native_steps=24000,
                forecast_native_steps=total_forecast,
                total_new_native_steps=24000 + total_forecast,
                reused_courses=4,
                independent_new_courses=0,
                optimizer_steps=0,
                official_football_exam_pass=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json",
            dict(
                error=repr(error),
                completed_actual_native_steps=completed_actual_native_steps,
                completed_forecast_native_steps=total_forecast,
                incomplete_actual_attempted_native_steps=current_body.attempted_steps
                if current_body is not None
                else 0,
                incomplete_actual_integrated_native_steps=current_body.integrated_steps
                if current_body is not None
                else 0,
                incomplete_forecast_attempted_native_steps=current_forecaster.attempted_native_steps
                if current_forecaster is not None
                else 0,
                incomplete_forecast_integrated_native_steps=current_forecaster.integrated_native_steps
                if current_forecaster is not None
                else 0,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
