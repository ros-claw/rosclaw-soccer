"""Fixed six-state actual old/new teacher pilot, full NN physics for every probe."""

import argparse
import json
from pathlib import Path

INDICES = (0, 2, 8, 22, 27, 30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("champion-audit", "champion-root", "qualification", "backbone-model", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("champion-audit-hash", "qualification-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--sequential-restoration", action="store_true")
    p.add_argument("--previous-pilot-audit", type=Path)
    p.add_argument("--previous-pilot-audit-hash")
    p.add_argument("--aim-command-search", action="store_true")
    p.add_argument("--aim-qualification", type=Path)
    p.add_argument("--aim-qualification-hash")
    p.add_argument("--executed-target-search", action="store_true")
    p.add_argument("--target-qualification", type=Path)
    p.add_argument("--target-qualification-hash")
    p.add_argument("--target-feasible-axes", action="store_true")
    p.add_argument("--actuated-task-response", action="store_true")
    p.add_argument("--task-qualification-audit", type=Path)
    p.add_argument("--task-qualification-audit-hash")
    p.add_argument("--motor-memory", type=Path)
    p.add_argument("--motor-memory-hash")
    p.add_argument("--memory-nominal-16", action="store_true")
    p.add_argument("--source-retrieval-audit", type=Path)
    p.add_argument("--source-retrieval-audit-hash")
    a = p.parse_args()
    memory = a.motor_memory is not None
    if a.memory_nominal_16 and not memory:
        raise ValueError("nominal transfer is an explicit fixed-memory study")
    if memory != (a.motor_memory_hash is not None):
        raise ValueError("explicit memory artifact and exact hash required together")
    if a.target_feasible_axes and not (a.executed_target_search or a.actuated_task_response):
        raise ValueError("explicit executed-target study required for per-axis feasibility")
    if a.actuated_task_response and not a.target_feasible_axes:
        raise ValueError("actual-task response requires the qualified per-axis interface")
    if a.actuated_task_response and any(
        value is None
        for value in (
            a.target_qualification,
            a.target_qualification_hash,
            a.task_qualification_audit,
            a.task_qualification_audit_hash,
        )
    ):
        raise ValueError("actual-task direction qualification and independent readback required")
    if not a.actuated_task_response and (
        a.task_qualification_audit is not None or a.task_qualification_audit_hash is not None
    ):
        raise ValueError("task-axis audit belongs only to the explicit actual-task protocol")
    if (
        sum(
            (
                a.aim_command_search,
                a.sequential_restoration,
                a.executed_target_search,
                memory,
                a.actuated_task_response,
            )
        )
        > 1
    ):
        raise ValueError("one explicit teacher mechanism per pilot required")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_stable_plastic_sources import merge_pins

    clean(a.expected_head)
    from rosclaw_soccer.rsi.partial_motor_memory import NOMINAL_MEMORY_TRANSFER_INDICES
    from scripts.rsi_actuated_task_pilot_common import (
        ACTUATED_INDICES,
        admit_actuated_qualification,
    )

    indices = (
        ACTUATED_INDICES
        if a.actuated_task_response
        else NOMINAL_MEMORY_TRANSFER_INDICES
        if a.memory_nominal_16
        else INDICES
    )
    world_ceiling = 2 * len(indices)
    native_ceiling = 3600 * world_ceiling
    if a.output.exists():
        raise ValueError("unused declared six-state pilot output required")
    pins = {}
    for path, h, status in (
        (
            a.champion_audit,
            a.champion_audit_hash,
            "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED",
        ),
        (a.qualification, a.qualification_hash, "G1_WHOLE_JOINT_REFERENCE_NN_FUTURES_EXACT"),
    ):
        if digest(path) != h or path.is_symlink():
            raise ValueError("exact independent inputs required")
        report = json.loads(path.read_text())
        if report["status"] != status:
            raise ValueError("closed old teacher and new full-reference qualification required")
        merge_pins(pins, report["input_file_hashes"])
        pins[str(path)] = h
    if a.sequential_restoration:
        path = a.previous_pilot_audit
        if path is None or path.is_symlink() or digest(path) != a.previous_pilot_audit_hash:
            raise ValueError(
                "explicit closed one-shot counterexample audit required for sequential research"
            )
        prior = json.loads(path.read_text())
        if (
            prior["status"] != "FULL_NN_JOINT_TASK_RESPONSE_RAW_AND_PHYSICAL_EXAM_VERIFIED"
            or prior["actual_worlds"] != 12
            or prior["new_physical_steps"] != 0
            or prior["qualified_for_expansion"]
        ):
            raise ValueError(
                "independent bounded one-shot failure, not unbounded solver sweeping, required"
            )
        merge_pins(pins, prior["input_file_hashes"])
        pins[str(path)] = a.previous_pilot_audit_hash
    if a.aim_command_search:
        path = a.aim_qualification
        if path is None or path.is_symlink() or digest(path) != a.aim_qualification_hash:
            raise ValueError("explicit nonzero aim-command NN physical qualification required")
        qualified = json.loads(path.read_text())
        if (
            qualified["status"] != "G1_AIM_GOAL_COMMAND_NN_FUTURES_EXACT"
            or not qualified["entry_command_forecast_conditioned"]
            or qualified["reference_conditioning"] != "OFF_original_arrays"
            or qualified["task_scoring_goal_world_point"] != [8.0, 0.0, 0.11]
        ):
            raise ValueError("same unchanged task and verified entry goal-command branch required")
        merge_pins(pins, qualified["input_file_hashes"])
        pins[str(path)] = a.aim_qualification_hash
    if a.executed_target_search:
        path = a.target_qualification
        if path is None or path.is_symlink() or digest(path) != a.target_qualification_hash:
            raise ValueError("explicit nonzero executed-target NN physics qualification required")
        qualified = json.loads(path.read_text())
        if (
            qualified["status"] != "G1_EXECUTED_TARGET_SYNERGY_NN_FUTURES_EXACT"
            or not qualified["all_future_nn_targets_and_native_arrays_exact"]
            or not qualified["reference_arrays_unchanged_by_target_plan"]
            or qualified.get("executed_target_projection_mode", "whole_tick")
            != ("per_axis" if a.target_feasible_axes else "whole_tick")
            or qualified["reference_conditioning"] != "OFF_original_arrays"
            or qualified["task_scoring_goal_world_point"] != [8.0, 0.0, 0.11]
        ):
            raise ValueError("unchanged reference and exact actual composite futures required")
        merge_pins(pins, qualified["input_file_hashes"])
        pins[str(path)] = a.target_qualification_hash
    if a.actuated_task_response:
        admit_actuated_qualification(
            a.target_qualification,
            a.target_qualification_hash,
            a.task_qualification_audit,
            a.task_qualification_audit_hash,
            pins,
        )
    audit = json.loads(a.champion_audit.read_text())
    if (
        audit.get("perturbed_public_validation", False)
        != (not (a.memory_nominal_16 or a.actuated_task_response))
        or audit["actual_worlds"] != 80
    ):
        raise ValueError("exact declared nominal/perturbed consumed40 champion source required")
    cp = a.champion_root / "complete.json"
    if pins.get(str(cp)) != digest(cp):
        raise ValueError("bound same-state actual champion required")
    champion = json.loads(cp.read_text())
    source_pins(pins)
    check_pins(pins)
    from scripts.rsi_teacher_physics_context import load_teacher_context

    context = load_teacher_context(pins, a.backbone_model)
    memory_options = ()
    memory_report = None
    if memory:
        from scripts.rsi_build_partial_motor_memory import admit_memory

        memory_options, memory_report = admit_memory(
            a.motor_memory, a.motor_memory_hash, pins, context, a.backbone_model
        )
        if a.memory_nominal_16:
            from scripts.rsi_build_partial_motor_memory import admit_source_retrieval

            admit_source_retrieval(
                a.source_retrieval_audit,
                a.source_retrieval_audit_hash,
                pins,
                a.motor_memory,
                a.motor_memory_hash,
                memory_report,
            )
        check_pins(pins)
    import numpy as np

    from rosclaw_soccer.rsi.actuated_response_kick_player import ActuatedResponseKickPlayer
    from rosclaw_soccer.rsi.aim_command_kick_player import (
        AimCommandKickPlayer,
        declared_aim_commands,
    )
    from rosclaw_soccer.rsi.executed_synergy_kick_player import ExecutedSynergyKickPlayer
    from rosclaw_soccer.rsi.partial_memory_kick_player import PartialMemoryKickPlayer
    from rosclaw_soccer.rsi.synergy_forecast_kick_player import SynergyForecastKickPlayer
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    a.output.mkdir(parents=True)
    candidate_cap = (
        39
        if memory
        else 73
        if a.executed_target_search
        else 57
        if a.aim_command_search
        else 273
        if a.sequential_restoration
        else 96
    )
    forecast_cap = len(indices) * (37 + candidate_cap) * 1280
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.full_nn_actuated_task_response_nominal6.v1"
            if a.actuated_task_response
            else "soccer.partial_motor_memory_nominal16_transfer.v1"
            if a.memory_nominal_16
            else "soccer.partial_motor_memory_source_retrieval.v1"
            if memory
            else "soccer.full_nn_joint_task_response_six_state_pilot.v1",
            partial_motor_memory=memory,
            memory_nominal_transfer=a.memory_nominal_16,
            source_retrieval_audit_hash=a.source_retrieval_audit_hash,
            motor_memory_hash=a.motor_memory_hash,
            memory_runtime_options=memory_report["runtime_options"] if memory else None,
            memory_source_states_are_consumed=memory,
            replays_memory_source_states=memory and not a.memory_nominal_16,
            development_bank_is_consumed=True,
            aim_command_search=a.aim_command_search,
            executed_target_search=a.executed_target_search,
            actuated_task_response=a.actuated_task_response,
            task_qualification_audit_hash=a.task_qualification_audit_hash,
            executed_target_projection_mode="per_axis"
            if a.target_feasible_axes or memory
            else "whole_tick",
            target_qualification_hash=a.target_qualification_hash,
            aim_command_world_points=[list(c.world_point) for c in declared_aim_commands()]
            if a.aim_command_search
            else None,
            task_scoring_goal_world_point=[8.0, 0.0, 0.11],
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_indices=indices,
            actual_world_ceiling=world_ceiling,
            actual_native_step_ceiling=native_ceiling,
            requested_forecast_native_step_ceiling=forecast_cap,
            sequential_restoration=a.sequential_restoration,
            maximum_response_rounds=4 if a.sequential_restoration else 1,
            initial_force_normalization_fixed=a.sequential_restoration,
            require_observed_ghost_progress_for_new_center=a.sequential_restoration,
            forecast_controls=128,
            controls_retention_indices=[0, 30],
            failure_indices=[2, 6, 8, 21] if a.actuated_task_response else [2, 8, 22, 27],
            old_candidate_ceiling=37,
            new_candidate_ceiling=candidate_cap,
            new_probe_axes=None if memory else 6 if a.executed_target_search else 29,
            new_probe_signs=None if a.executed_target_search or memory else [-1, 1],
            probe_reference_rad=None
            if a.executed_target_search or memory or a.actuated_task_response
            else 0.03,
            probe_actual_target_rad=0.03 if a.actuated_task_response else None,
            proposal_l1_trust_rad=None if a.executed_target_search or memory else 0.06,
            proposal_axis_trust_rad=0.06 if a.executed_target_search or memory else 0.03,
            temporal_lobe="typed_frozen_memory_options"
            if memory
            else "two24interval_lobes"
            if a.executed_target_search
            else "first24intervals_only",
            second_lobe_coefficients_zero=not (a.executed_target_search or memory),
            old_success_branch_must_not_search_new_actions=True,
            new_research_direction="task_derived_HG29_actual_target_response_not_PCA_or_reference"
            if a.actuated_task_response
            else "frozen_partial_options_current_body_retrieval"
            if memory
            else "actual_HG29_target_authority_not_reference_input"
            if a.executed_target_search
            else "task_conditioned_full_NN_physics_response_not_success_PCA",
            parent_nn_weights_and_PD_unchanged=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    results, worlds, native, charged, integrated = [], 0, 0, 0, 0
    for i in indices:
        ref = champion["results"][i]["arms"]["C"]
        rp = a.champion_root / ref["record"]
        if pins.get(str(rp)) != ref["hash"] or digest(rp) != ref["hash"]:
            raise ValueError("exact old actual teacher record required")
        old = json.loads(rp.read_text())
        folder = a.output / f"course-{i:02d}"
        folder.mkdir()
        pair = {}
        for arm in ("B", "C"):
            instances = []

            class Controlled(
                PartialMemoryKickPlayer
                if memory and arm == "C"
                else ActuatedResponseKickPlayer
                if a.actuated_task_response and arm == "C"
                else ExecutedSynergyKickPlayer
                if a.executed_target_search and arm == "C"
                else AimCommandKickPlayer
                if a.aim_command_search and arm == "C"
                else SynergyForecastKickPlayer
            ):
                basis_hg = context["basis"]
                forecast_controls = 128
                joint_task_response_enabled = (
                    arm == "C"
                    and not a.aim_command_search
                    and not a.executed_target_search
                    and not memory
                    and not a.actuated_task_response
                )
                joint_response_iterations = 4 if a.sequential_restoration and arm == "C" else 1
                target_projection_mode = (
                    "per_axis" if a.target_feasible_axes or memory else "whole_tick"
                )
                observed_instances = instances

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.observed_instances.append(self)

            Controlled.memory_options = memory_options

            row, arrays = evaluate_episode(
                model=context["model"],
                assets=dict(context["assets"], experimental_fast=old["experimental_fast_approach"]),
                policy_types=context["types"],
                course=tuple(old["course"]),
                parameters=old["parameters"],
                hard_limits=context["hard_limits"],
                player_type=Controlled,
            )
            if a.actuated_task_response:
                # Preserve charged physical output even if a later assertion fails.
                np.savez_compressed(folder / f"{arm}-unvalidated.npz", **arrays)
            if len(instances) != 1:
                raise ValueError("one privately owned actual teacher required")
            ctrl = instances[0]
            if a.actuated_task_response:
                for j, (_, forecast) in enumerate(ctrl.candidate_predictions):
                    np.savez_compressed(
                        folder / f"{arm}-forecast-{j:02d}-unvalidated.npz",
                        **{
                            k: forecast[k]
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
                write_once(
                    folder / f"{arm}-unvalidated.json",
                    dict(
                        actual_record=row,
                        controller_events=ctrl.events,
                        actual_target_edits=getattr(ctrl, "target_edits", []),
                        forecasts=ctrl.forecaster.counters(),
                        validation_complete=False,
                    ),
                )
            if arm == "B":
                ap = rp.parent / old["artifact"]["path"]
                if pins.get(str(ap)) != digest(ap):
                    raise ValueError("exact old full native baseline required")
                with np.load(ap, allow_pickle=False) as before:
                    if set(before.files) != set(arrays) or any(
                        not np.array_equal(before[k], v) for k, v in arrays.items()
                    ):
                        raise ValueError("actual B differs from original champion whole trajectory")
            for frame, forecast in ctrl.selected_records:
                n = forecast["controls"]
                for key in ("qpos", "qvel", "torque", "ball_forces"):
                    if not np.array_equal(
                        forecast[key], arrays[key][frame * 10 : (frame + n) * 10]
                    ):
                        raise ValueError("selected full NN physical prediction differs from actual")
                if not np.array_equal(
                    forecast["joint_targets"], arrays["joint_targets"][frame : frame + n]
                ):
                    raise ValueError("selected future NN target differs from actual")
            if arm == "C" and i in (0, 30):
                with np.load(folder / "B.npz", allow_pickle=False) as before:
                    if any(not np.array_equal(before[k], v) for k, v in arrays.items()):
                        raise ValueError("old retained skill altered by new search interface")
            saved = []
            for j, (name, forecast) in enumerate(ctrl.candidate_predictions):
                fp = folder / f"{arm}-forecast-{j:02d}.npz"
                np.savez_compressed(
                    fp,
                    **{
                        k: forecast[k]
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
                        artifact=dict(path=fp.name, hash=digest(fp)),
                        donor_control_frame=forecast["donor_control_frame"],
                        controls=forecast["controls"],
                        requested_controls=forecast["requested_controls"],
                        **(
                            dict(target_edits=forecast["executed_target_edits"])
                            if a.executed_target_search or memory or a.actuated_task_response
                            else {}
                        ),
                    )
                )
            ap = folder / f"{arm}.npz"
            np.savez_compressed(ap, **arrays)
            row.update(
                controller_events=ctrl.events,
                candidate_forecasts=saved,
                forecasts=ctrl.forecaster.counters(),
                changed_controls=ctrl.changed_controls,
                artifact=dict(path=ap.name, hash=digest(ap)),
                course_index=i,
                arm=arm,
                task_scoring_goal_world_point=[8.0, 0.0, 0.11],
                desired_command_world_point=ctrl.goal.tolist(),
                **(
                    dict(actual_target_edits=getattr(ctrl, "target_edits", []))
                    if a.executed_target_search or memory or a.actuated_task_response
                    else {}
                ),
            )
            write_once(folder / f"{arm}.json", row)
            pair[arm] = dict(
                record=f"course-{i:02d}/{arm}.json",
                hash=digest(folder / f"{arm}.json"),
                task_pass=row["outcome"]["task_pass"],
                safe=row["outcome"]["safe"],
                healthy=row["healthy_full_horizon"],
            )
            worlds += 1
            native += row["native_steps"]
            charged += ctrl.forecaster.attempted_steps
            integrated += ctrl.forecaster.integrated_steps
            if worlds > world_ceiling or native > native_ceiling or charged > forecast_cap:
                raise ValueError("frozen full-NN pilot budget exceeded")
            print(i, arm, pair[arm], flush=True)
        results.append(dict(course_index=i, arms=pair))
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
            status="FULL_NN_ACTUATED_TASK_RESPONSE_PILOT_COMPLETE"
            if a.actuated_task_response
            else "FULL_NN_PARTIAL_MEMORY_RETRIEVAL_COMPLETE"
            if memory
            else "FULL_NN_EXECUTED_TARGET_PILOT_COMPLETE"
            if a.executed_target_search
            else "FULL_NN_AIM_COMMAND_PILOT_COMPLETE"
            if a.aim_command_search
            else "FULL_NN_JOINT_TASK_RESPONSE_PILOT_COMPLETE",
            source_head=a.expected_head,
            input_file_hashes=pins,
            declaration_hash=digest(a.output / "declaration.json"),
            results=results,
            rescues=rescues,
            regressions=regressions,
            actual_worlds=worlds,
            actual_native_steps=native,
            forecast_requested_native_steps=charged,
            forecast_integrated_native_steps=integrated,
            independently_audited=False,
            neural_optimizer_updates=0,
            physical_teacher_qualified=False,
            memory_nominal_transfer=a.memory_nominal_16,
            actuated_task_response=a.actuated_task_response,
            memory_retrieval_is_not_independent_generalization=memory,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
