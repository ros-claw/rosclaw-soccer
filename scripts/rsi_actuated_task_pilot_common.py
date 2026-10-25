"""Admission and raw-data replay for one finite actual-target pilot, no stepping."""

import json

ACTUATED_INDICES = (0, 2, 6, 8, 21, 30)


def admit_actuated_qualification(path, expected, audit_path, audit_hash, pins):
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_stable_plastic_sources import merge_pins

    for p, h in ((path, expected), (audit_path, audit_hash)):
        if p is None or h is None or p.is_symlink() or digest(p) != h:
            raise ValueError("exact actual-task qualification and independent readback required")
    report = json.loads(path.read_text())
    audit = json.loads(audit_path.read_text())
    if (
        report["status"] != "G1_ACTUATED_TASK_DIRECTION_NN_FUTURES_EXACT"
        or report["actuated_task_axis"] != 3
        or report["actuated_task_probe_rad"] != 0.03
        or report["executed_target_projection_mode"] != "per_axis"
        or report["reference_conditioning"] != "OFF_original_arrays"
        or report["fitted_pca_basis_used"] is not False
        or report["all_future_nn_targets_and_native_arrays_exact"] is not True
        or report["reference_arrays_unchanged_by_target_plan"] is not True
        or report["actual_whole_trajectory_unchanged"] is not False
        or audit["status"] != "G1_TASK_AXIS_TARGET_RAW_CODEC_AND_FUTURES_VERIFIED"
        or audit["qualification_hash"] != expected
        or audit["actual_native_steps_readback"] != 3600
        or audit["future_native_steps_readback"] != 6400
        or audit["nonzero_applied_controls"] != 23
        or audit["new_physical_steps"] != 0
        or audit["new_neural_calls"] != 0
        or audit["neural_optimizer_updates"] != 0
        or any(audit[k] is not False for k in ("goal_complete", "hardware_authorized"))
        or audit["input_file_hashes"].get(str(path)) != expected
    ):
        raise ValueError("new rank-one actual-target interface cannot inherit old PCA certificate")
    for p, h, r in ((path, expected, report), (audit_path, audit_hash, audit)):
        merge_pins(pins, r["input_file_hashes"])
        pins[str(p)] = h


def descriptor_plan(descriptor, window, entry):
    import numpy as np

    from rosclaw_soccer.rsi.actuated_task_response import actuated_task_plan
    from rosclaw_soccer.sim.contracts import hash_json

    delta = np.array(descriptor["first_lobe_offset_hg_rad"], np.float64)
    plan = actuated_task_plan(entry, delta)
    expected = dict(
        name=window["name"],
        first_lobe_offset_hg_rad=delta.tolist(),
        basis_hg=plan.basis_hg,
        coefficients=plan.coefficients,
        start_control=entry,
        projection_mode="per_axis",
    )
    if hash_json(descriptor) != hash_json(expected) or hash_json(
        window["coefficients"]
    ) != hash_json(plan.coefficients):
        raise ValueError("actual-target rank-one direction/clock/coefficients differ")
    return plan


def verify_actuated_response(windows, forecasts, event, entry):
    import numpy as np

    from rosclaw_soccer.rsi.joint_task_response import (
        propose_joint_task_response,
        response_features,
    )
    from rosclaw_soccer.sim.contracts import hash_json

    additional = event["additional_candidate_search"]
    plans = additional["candidate_plans"]
    if (
        len(windows) not in (95, 96)
        or len(plans) != len(windows) - 37
        or len(forecasts) != len(windows)
        or event["joint_task_response"] is not None
    ):
        raise ValueError("exact 58 signed actual-target futures and at most one proposal required")
    probes = []
    for axis in range(29):
        for k, sign in enumerate((-1, 1)):
            local = axis * 2 + k
            j = 37 + local
            delta = np.zeros(29, np.float64)
            delta[axis] = sign * 0.03
            if windows[j]["name"] != f"actual-axis-{axis:02d}-{sign:+d}" or not np.array_equal(
                delta, np.array(plans[local]["first_lobe_offset_hg_rad"])
            ):
                raise ValueError("complete ordered signed actual-target directions required")
            descriptor_plan(plans[local], windows[j], entry)
            probes.append((axis, sign, response_features(forecasts[j], windows[j]["outcome"])))
    proposal, fit = propose_joint_task_response(
        response_features(forecasts[0], windows[0]["outcome"]), probes
    )
    if proposal is None:
        if len(windows) != 95:
            raise ValueError("abstained actual-target response cannot fabricate a proposal")
    elif (
        len(windows) != 96
        or np.any(proposal[1])
        or windows[95]["name"] != "actual-task-response"
        or not np.array_equal(proposal[0], np.array(plans[58]["first_lobe_offset_hg_rad"]))
    ):
        raise ValueError("one exact bounded first-lobe actual-target LP proposal required")
    else:
        descriptor_plan(plans[58], windows[95], entry)
    expected = dict(
        schema="soccer.full_nn_actuated_task_response.v1",
        candidate_plans=plans,
        solver_report=fit,
        solver_math_reused_from_reference_study=True,
        fitted_pca_basis_used_for_new_actions=False,
        policy_reference_modified=False,
        actual_target_codec="existing_per_axis_qualified_history",
        new_action_or_torque_limits=False,
        neural_optimizer_updates=0,
        physical_teacher_qualified=False,
        hardware_authorized=False,
    )
    if hash_json(additional) != hash_json(expected):
        raise ValueError("actual-target solver/authority report differs from raw full-NN futures")
