"""Frozen balanced16 B/C control-authority pilot; actual native physics only."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("qualification", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--qualification-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--shard-root", type=Path, nargs=4, required=True)
    p.add_argument("--synergy-model", type=Path)
    p.add_argument("--synergy-model-hash")
    for name in ("plastic-model", "plastic-qualification", "retention-proof", "champion-root"):
        p.add_argument("--" + name, type=Path)
    for name in ("plastic-model-hash", "plastic-qualification-hash", "retention-proof-hash"):
        p.add_argument("--" + name)
    p.add_argument("--window-qualification", type=Path)
    p.add_argument("--window-qualification-hash")
    p.add_argument("--expand-40", action="store_true")
    p.add_argument("--expansion-gate", type=Path)
    p.add_argument("--expansion-gate-hash")
    p.add_argument("--perturb-40", action="store_true")
    p.add_argument("--independent-local-40", action="store_true")
    p.add_argument("--development-gate", type=Path, nargs=2)
    p.add_argument("--development-gate-hash", nargs=2)
    p.add_argument(
        "--interface",
        choices=(
            "measured-pose",
            "foundation-increment",
            "precontact-sparse",
            "nn-outcome",
            "nn-outcome-full-leg",
            "nn-outcome-response",
            "motion-timing",
            "motion-synergy",
            "motion-synergy-response",
            "motion-synergy-full-window",
            "motion-synergy-stable-plastic",
        ),
        default="measured-pose",
    )
    a = p.parse_args()
    synergy_response = a.interface == "motion-synergy-response"
    plastic = a.interface == "motion-synergy-stable-plastic"
    full_window = plastic or a.interface == "motion-synergy-full-window"
    synergy = a.interface in (
        "motion-synergy",
        "motion-synergy-response",
        "motion-synergy-full-window",
        "motion-synergy-stable-plastic",
    )
    forecast_controls = 128 if full_window else 96
    if a.expand_40 and not full_window:
        raise ValueError("expanded validation requires the full-window teacher protocol")
    if a.perturb_40 and not a.expand_40:
        raise ValueError("local robustness requires explicit expanded40 protocol and its gate")
    if plastic and not a.expand_40:
        raise ValueError("stable/plastic development requires all40 original context clusters")
    if a.independent_local_40 and not (plastic and a.perturb_40):
        raise ValueError("independent local exam requires the qualified stable/plastic teacher")
    course_count = 40 if a.expand_40 else 16
    world_cap, native_cap = 2 * course_count, 2 * course_count * 3600
    motion_timing = synergy or a.interface == "motion-timing"
    nn_outcome = motion_timing or a.interface in (
        "nn-outcome",
        "nn-outcome-full-leg",
        "nn-outcome-response",
    )
    response = a.interface == "nn-outcome-response"
    axes = (
        tuple(range(12))
        if a.interface in ("nn-outcome-full-leg", "nn-outcome-response")
        else (0, 3, 4, 6, 9, 10)
    )
    forecast_cap = (1 + 2 * len(axes) + int(response)) * 640 * 16 if nn_outcome else 172800
    if motion_timing:
        forecast_cap = (
            (73 if plastic else 38 if synergy_response else 37 if synergy else 3)
            * forecast_controls
            * 10
            * course_count
        )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or digest(a.qualification) != a.qualification_hash:
        raise ValueError("unused pilot output and exact measured-G1 qualification required")
    qualification = json.loads(a.qualification.read_text())
    required = (
        "G1_SYNERGY_REFERENCE_NN_FUTURES_EXACT"
        if synergy
        else "G1_COPIED_NN_CLOSED_LOOP_96_CONTROL_FORECASTS_EXACT"
        if motion_timing
        else "G1_COPIED_NN_CLOSED_LOOP_64_CONTROL_FORECASTS_EXACT"
        if nn_outcome
        else "ACTUAL_G1_MATERIAL_POINT_HG29_DERIVATIVES_VERIFIED"
    )
    if qualification["status"] != required:
        raise ValueError("actual body derivative prerequisite missing")
    if (
        motion_timing
        and not synergy
        and (
            not qualification["before_proposal"]
            or qualification["reference_conditioning"] != "OFF_original_arrays"
        )
    ):
        raise ValueError("explicit pre-proposal OFF reference qualification required")
    pins = dict(qualification["input_file_hashes"])
    pins[str(a.qualification)] = a.qualification_hash
    if a.expand_40:
        if a.expansion_gate is None or digest(a.expansion_gate) != a.expansion_gate_hash:
            raise ValueError("exact independently verified expansion gate required")
        gate = json.loads(a.expansion_gate.read_text())
        if (
            gate["status"] != "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED"
            or not gate["qualified_for_expansion"]
            or gate["regressions"]
            or len(set(gate["rescues"])) < 2
            or gate["actual_worlds"] != (80 if a.perturb_40 or plastic else 32)
            or gate["new_physical_steps"] != 0
            or (a.perturb_40 and not gate.get("qualified_on_expanded_public", False))
            or gate.get("perturbed_public_validation", False)
        ):
            raise ValueError("two distinct rescues and zero regression independent gate required")
        for key, value in gate["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting expansion gate provenance")
            pins[key] = value
        pins[str(a.expansion_gate)] = a.expansion_gate_hash
    if plastic:
        from scripts.rsi_stable_plastic_sources import admit_stable_plastic_sources

        admit_stable_plastic_sources(a, pins)
    development_gates = None
    if a.independent_local_40:
        from scripts.rsi_stable_plastic_sources import admit_development_gates

        development_gates = admit_development_gates(
            a.development_gate, a.development_gate_hash, a.plastic_model_hash, pins
        )
    if full_window:
        if (
            a.window_qualification is None
            or digest(a.window_qualification) != a.window_qualification_hash
        ):
            raise ValueError("separate exact128 observation qualification required")
        longer = json.loads(a.window_qualification.read_text())
        if (
            longer["status"] != "G1_COPIED_NN_CLOSED_LOOP_128_CONTROL_FORECASTS_EXACT"
            or not longer["before_proposal"]
            or longer["reference_conditioning"] != "OFF_original_arrays"
        ):
            raise ValueError("original128 pre-proposal OFF qualification required")
        for key, value in longer["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting128 qualification source/asset binding")
            pins[key] = value
        pins[str(a.window_qualification)] = a.window_qualification_hash
    if synergy and (
        a.synergy_model is None
        or pins.get(str(a.synergy_model)) != a.synergy_model_hash
        or digest(a.synergy_model) != a.synergy_model_hash
    ):
        raise ValueError("explicit qualified data-derived motion model required")
    originals, original_paths = {}, {}
    indices = (
        list(range(40)) if a.expand_40 else [b * 10 + j for b in range(4) for j in (0, 1, 2, 3)]
    )
    for index in indices:
        path = a.shard_root[index % 4] / f"course-{index:02d}" / "onnx.json"
        if pins.get(str(path)) != digest(path):
            raise ValueError("original closed baseline reference not bound")
        original_paths[index] = path
        originals[index] = json.loads(path.read_text())
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique source asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.motion_timing_kick_player import MotionTimingKickPlayer
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.outcome_forecast_kick_player import OutcomeForecastKickPlayer
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
    from rosclaw_soccer.rsi.synergy_forecast_kick_player import SynergyForecastKickPlayer
    from rosclaw_soccer.rsi.task_space_kick_player import TaskSpaceKickPlayer
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import search_course_bank
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    configs = []
    for name in pins:
        path = Path(name)
        if (
            path.suffix == ".json"
            and path.stat().st_size < 2048
            and json.loads(path.read_text()).get("schema")
            == "soccer.native_single_finisher_experimental_approach.v1"
        ):
            configs.append(path)
    if len(configs) != 1:
        raise ValueError("one qualified experimental velocity codec required")
    fast, _ = admit_heading_evidence(configs[0], pins[str(configs[0])])
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
        **fast,
    )
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    courses = search_course_bank(expanded_public=True)
    preflight_receiving_courses(courses)
    perturbation_declaration = None
    if a.perturb_40:
        from rosclaw_soccer.rsi.synergy_teacher_perturbations import declare_synergy_perturbations

        if a.independent_local_40:
            from rosclaw_soccer.rsi.synergy_teacher_perturbations import declare_feedback_validation

            courses, perturbation_declaration = declare_feedback_validation(courses)
        else:
            courses, perturbation_declaration = declare_synergy_perturbations(
                courses, seed=2026102519
            )
    instances = []

    parent_type = (
        SynergyForecastKickPlayer
        if synergy
        else MotionTimingKickPlayer
        if motion_timing
        else OutcomeForecastKickPlayer
        if nn_outcome
        else TaskSpaceKickPlayer
    )
    fitted_basis = plastic_basis = None
    if synergy:
        fitted = json.loads(a.synergy_model.read_text())
        artifact = a.synergy_model.parent / fitted["artifact"]["path"]
        if pins.get(str(artifact)) != fitted["artifact"]["hash"]:
            raise ValueError("qualified synergy artifact binding differs")
        with np.load(artifact, allow_pickle=False) as raw:
            fitted_basis = raw["basis_hg"].copy()
        if plastic:
            fitted_plastic = json.loads(a.plastic_model.read_text())
            artifact = a.plastic_model.parent / fitted_plastic["artifact"]["path"]
            with np.load(artifact, allow_pickle=False) as raw:
                plastic_basis = raw["basis_hg"].copy()

    class Controlled(parent_type):
        basis_hg = fitted_basis
        plastic_basis_hg = plastic_basis
        synergy_response_enabled = synergy_response
        forecast_controls = 128 if full_window else 96
        candidate_axes = axes
        response_enabled = response
        foundation_relative = a.interface != "measured-pose"
        precontact_sparse = a.interface == "precontact-sparse"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.g2.bounded_contact_authority_pilot.v3",
            interface=a.interface,
            expanded_public_validation=a.expand_40,
            perturbed_public_validation=a.perturb_40,
            stable_plastic_development=plastic and not a.independent_local_40,
            consumed_perturbation_development=plastic
            and a.perturb_40
            and not a.independent_local_40,
            independent_local_validation=a.independent_local_40,
            development_gates=development_gates,
            development_gate_paths=[str(p) for p in a.development_gate]
            if a.independent_local_40
            else None,
            champion_root=str(a.champion_root) if plastic else None,
            plastic_model_hash=a.plastic_model_hash if plastic else None,
            plastic_search_requires_backbone_abstention=plastic,
            perturbation_declaration=perturbation_declaration,
            expansion_gate_hash=a.expansion_gate_hash if a.expand_40 else None,
            freeze_inadmissible_parent_joints=a.interface == "precontact-sparse",
            end_alignment_after_actual_first_foot=a.interface == "precontact-sparse",
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_indices=indices,
            arms=["B", "C"],
            actual_world_ceiling=world_cap,
            actual_native_step_ceiling=native_cap,
            forecast_native_step_ceiling=forecast_cap,
            forecast_controls_per_candidate=forecast_controls
            if motion_timing
            else 64
            if nn_outcome
            else 1,
            timing_intervals=24 if motion_timing and not synergy else None,
            timing_peak_shift_frames=[-3.0, 3.0] if motion_timing and not synergy else None,
            timing_trigger="first_measured_skill_entry_pre_proposal" if motion_timing else None,
            reference_arrays_only=motion_timing,
            synergy_cem_population=12 if synergy else None,
            synergy_cem_generations=3 if synergy else None,
            synergy_cem_elites=3 if synergy else None,
            synergy_cem_seed=7001 if synergy else None,
            synergy_cem_initial_std_rad=0.025 if synergy else None,
            synergy_cem_minimum_std_rad=0.006 if synergy else None,
            synergy_cem_update_smoothing=0.5 if synergy else None,
            synergy_reference_trust_rad=0.06 if synergy else None,
            synergy_reference_intervals=48 if synergy else None,
            synergy_temporal_lobes=2 if synergy else None,
            synergy_response_enabled=synergy_response,
            synergy_response_ridge=0.1 if synergy_response else None,
            synergy_response_forward_minimum_m=1.05 if synergy_response else None,
            synergy_response_lateral_ratio=0.27 if synergy_response else None,
            reference_installations_not_changed_pd_controls=motion_timing,
            candidate_set=(
                ["parent", "3generations_x12_data_synergy_plans"]
                + (["3generations_x12_feedback_complement_on_abstention"] if plastic else [])
                if synergy
                else ["parent", "timing--3", "timing-+3"]
                if motion_timing
                else ["parent", "one_admissible_declared_joint_plusminus_0.03"]
                + (["native-response-coordinated"] if response else [])
            )
            if nn_outcome
            else [
                "original",
                "foot_nominal_plus_shin_clearance",
                "foot_contact_alignment_plus_shin_clearance",
            ],
            active_joints=None if motion_timing else list(axes) if nn_outcome else list(range(12)),
            max_nominal_delta_rad=None if motion_timing else 0.03 if nn_outcome else 0.12,
            max_current_delta_rad=0.2 if a.interface == "measured-pose" else None,
            max_command_step_rad=0.1 if a.interface == "measured-pose" else None,
            candidate_increment_step_cap_rad=0.03
            if a.interface != "measured-pose" and not motion_timing
            else None,
            fallback_to_parent_resets_increment=True,
            foot_shin_cost=None if nn_outcome else 100.0,
            posture_cost=None if nn_outcome else 0.1,
            shin_margin_m=None if nn_outcome else 0.015,
            maximum_retreat_m=None if nn_outcome else 0.04,
            contact_alignment_correction_m=None if nn_outcome else [0.02, 0.03, 0.015],
            activation_foot_gap_m=None if motion_timing else 0.06 if nn_outcome else 0.18,
            maximum_actual_interventions_per_world=1 if nn_outcome else None,
            candidate_selection_requires_observed_window_rescue=nn_outcome,
            local_native_response_model=response,
            response_target_forward_margin_m=1.2 if response else None,
            response_target_lateral_ratio=0.25 if response else None,
            response_ridge=0.1 if response else None,
            no_model_or_gain_changes=True,
            privileged_per_course_approach=True,
            original_exam_unchanged=True,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            success_criterion=(
                "at least2 distinct rescues, no historical task/safety/body regressions; "
                "then expand40+perturbations"
            ),
            falsifier=(
                "no repeatable physical gain or only regressions; "
                "no unbounded parameter/gain sweeps"
            ),
        ),
    )
    results, worlds, native, forecasts, forecast_attempted = [], 0, 0, 0, 0
    for index in indices:
        folder = a.output / f"course-{index:02d}"
        folder.mkdir()
        pair = {}
        for arm in ("B", "C"):
            write_once(
                folder / f"{arm}-attempt.json",
                dict(
                    course_index=index,
                    arm=arm,
                    source_head=a.expected_head,
                    original_record_hash=pins[str(original_paths[index])],
                ),
            )
            instances.clear()
            original = originals[index]
            row, arrays = evaluate_episode(
                model=model,
                assets=dict(assets, experimental_fast=original["experimental_fast_approach"]),
                policy_types=types,
                course=courses[index].course,
                parameters=original["parameters"],
                hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
                player_type=SharedRoboNaldoPlayer if arm == "B" else Controlled,
            )
            worlds += 1
            native += row["native_steps"]
            if hash_json(
                relabel_arrays(
                    arrays, controls=row["controls"], ball_address=layout.bq, root_address=layout.rq
                )
            ) != hash_json(row["outcome"]):
                raise ValueError("independent actual native label differs")
            if (
                abs(arrays["torque"][:, layout.ai]) > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
            ).any():
                raise ValueError("hard torque bound changed")
            if arm == "B":
                if not a.perturb_40:
                    with np.load(
                        original_paths[index].parent / original["artifact"]["path"]
                    ) as old:
                        if set(old.files) != set(arrays) or any(
                            not np.array_equal(old[k], v) for k, v in arrays.items()
                        ):
                            raise ValueError("baseline differs from closed original raw physics")
                row["exact_original_baseline"] = not a.perturb_40
                row["new_perturbed_baseline_execution"] = a.perturb_40
            elif instances:
                if len(instances) != 1:
                    raise ValueError("one private local controller required")
                controller = instances[0]
                for frame, prediction in controller.selected_records:
                    predicted_controls = prediction.get("controls", 1)
                    for key in ("qpos", "qvel", "torque", "ball_forces"):
                        if not np.array_equal(
                            prediction[key],
                            arrays[key][frame * 10 : (frame + predicted_controls) * 10],
                        ):
                            raise ValueError("chosen prediction differs at native substeps: " + key)
                    if nn_outcome and not np.array_equal(
                        prediction["joint_targets"],
                        arrays["joint_targets"][frame : frame + predicted_controls],
                    ):
                        raise ValueError("selected future NN targets differ from actual branch")
                row["controller_events"] = controller.events
                if a.interface == "precontact-sparse":
                    first_foot = row["outcome"]["first_foot_contact_frame"]
                    if first_foot is not None and any(
                        e["changed"] and e["frame"] > first_foot for e in controller.events
                    ):
                        raise ValueError("contact alignment continued after actual first foot")
                row["changed_controls"] = controller.changed_controls
                row["forecasts"] = controller.forecaster.counters()
                forecasts += controller.forecaster.integrated_steps
                forecast_attempted += controller.forecaster.attempted_steps
                if nn_outcome:
                    if controller.changed_controls > 1:
                        raise ValueError("one-intervention protocol exceeded")
                    saved = []
                    for i, (name, prediction) in enumerate(controller.candidate_predictions):
                        forecast_file = folder / f"C-forecast-{i:02d}.npz"
                        np.savez_compressed(
                            forecast_file,
                            **{
                                k: prediction[k]
                                for k in (
                                    "qpos",
                                    "qvel",
                                    "torque",
                                    "ball_forces",
                                    "joint_targets",
                                    "policy_ticks",
                                )
                            },
                        )
                        saved.append(
                            dict(
                                name=name,
                                artifact=dict(path=forecast_file.name, hash=digest(forecast_file)),
                                donor_control_frame=prediction["donor_control_frame"],
                                controls=prediction["controls"],
                                requested_controls=prediction["requested_controls"],
                            )
                        )
                    row["candidate_forecasts"] = saved
                if controller.changed_controls == 0:
                    zero_reference = (
                        folder / "B.npz"
                        if a.perturb_40
                        else original_paths[index].parent / original["artifact"]["path"]
                    )
                    with np.load(zero_reference) as old:
                        if any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                            raise ValueError("zero/off branch altered the legacy trajectory")
            if worlds > world_cap or native > native_cap or forecast_attempted > forecast_cap:
                raise ValueError("frozen physical budget exceeded")
            artifact = folder / f"{arm}.npz"
            np.savez_compressed(artifact, **arrays)
            row.update(
                arm=arm,
                course_index=index,
                artifact=dict(path=artifact.name, hash=digest(artifact)),
            )
            write_once(folder / f"{arm}.json", row)
            pair[arm] = dict(
                record=str((folder / f"{arm}.json").relative_to(a.output)),
                hash=digest(folder / f"{arm}.json"),
                task_pass=row["outcome"]["task_pass"],
                safe=row["outcome"]["safe"],
                healthy=row["healthy_full_horizon"],
                first_foot=row["outcome"]["first_foot_contact_frame"],
                changed_controls=row.get("changed_controls", 0),
            )
            write_once(
                folder / f"{arm}-finished.json",
                dict(
                    actual_worlds=worlds,
                    actual_native_steps=native,
                    forecast_native_steps=forecasts,
                ),
            )
            print(index, arm, pair[arm], flush=True)
        results.append(dict(course_index=index, arms=pair))
    champion_regressions, champion_rescues = [], []
    if plastic and not a.independent_local_40:
        from scripts.rsi_stable_plastic_sources import compare_champion

        champion_regressions, champion_rescues = compare_champion(a.output, a.champion_root, pins)
    rescues = [
        r["course_index"]
        for r in results
        if not r["arms"]["B"]["task_pass"]
        and r["arms"]["C"]["task_pass"]
        and r["arms"]["C"]["healthy"]
    ]
    regressions = [
        r["course_index"]
        for r in results
        if any(
            r["arms"]["B"][k] and not r["arms"]["C"][k] for k in ("task_pass", "safe", "healthy")
        )
    ]
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="BOUNDED_SURFACE_CONTACT_PILOT_COMPLETE",
            interface=a.interface,
            declaration_hash=digest(a.output / "declaration.json"),
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            actual_worlds=worlds,
            actual_native_steps=native,
            forecast_native_steps=forecasts,
            forecast_attempted_native_steps=forecast_attempted,
            rescues=rescues,
            regressions=regressions,
            qualified_for_expansion=len(rescues) >= 2
            and not regressions
            and not champion_regressions,
            expanded_public_validation=a.expand_40,
            perturbed_public_validation=a.perturb_40,
            stable_plastic_development=plastic and not a.independent_local_40,
            consumed_perturbation_development=plastic
            and a.perturb_40
            and not a.independent_local_40,
            independent_local_validation=a.independent_local_40,
            champion_regressions=champion_regressions,
            champion_rescues=champion_rescues,
            qualified_on_expanded_public=(
                a.expand_40
                and not a.perturb_40
                and len(rescues) >= 2
                and not regressions
                and not champion_regressions
                and sum(r["arms"]["C"]["task_pass"] for r in results) >= 32
            ),
            qualified_on_local_perturbations=(
                a.perturb_40
                and (not plastic or a.independent_local_40)
                and not regressions
                and sum(r["arms"]["C"]["task_pass"] for r in results) >= 32
                and all(r["arms"]["C"]["healthy"] for r in results)
                and all(results[i]["arms"]["C"]["task_pass"] for i in (0, 30))
            ),
            optimizer_updates=0,
            physical_teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
