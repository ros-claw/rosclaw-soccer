"""Queue admission tests only, no physical experiment or neural policy."""

import copy

import pytest

from scripts.rsi_stage_release_contact_replay import JOBS, expected_jobs, require_four_endpoints


def fixture():
    cfg = dict(
        mode="SIM_ONLY",
        hardware_authorized=False,
        promotion_authorized=False,
        neural_training_authorized=False,
        max_attempts=4,
        new_physical_budget=4,
        suppress_postcontact_swing_release=True,
    )
    decl = dict(report_hash="declaration", jobs=[dict(job=j) for j in JOBS])
    results = [
        dict(job=j, receipt_hash=str(i), outcome=dict(task_pass=i != 2, safe=i != 2))
        for i, j in enumerate(JOBS)
    ]
    complete = dict(
        schema="soccer.rsi.swing_release_ablation_pilot_complete.v1",
        report_hash="complete",
        declaration_hash="declaration",
        results=results,
        native_substeps=12000,
        independent_cpu_substeps=12000,
        new_optimizer_updates=0,
    )
    readback = dict(
        schema="soccer.rsi.swing_release_ablation_terminal_readback.v1",
        complete_hash="complete",
        source_native_bodies=4,
        rows=copy.deepcopy(results),
        whole_input_and_witness_files_verified=True,
        shared_payload_logical_seals_rebuilt=True,
        prior_independent_replays_authenticated=True,
        all_original_first_contact_prefixes_bit_equal=True,
        actual_postcontact_taskspace_increment_exactly_zero=True,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return cfg, decl, complete, readback


def test_all_four_failures_are_legitimate_diagnostic_inputs():
    args = fixture()
    assert args[3]["rows"][2]["outcome"]["safe"] is False
    require_four_endpoints(*args)


def mpc_fixture():
    args = fixture()
    cfg, declaration, complete, readback = args
    cfg.update(use_contact_exit_torque_mpc=True, suppress_postcontact_swing_release=False)
    jobs = expected_jobs(cfg)
    for document, key in ((declaration, "jobs"), (complete, "results"), (readback, "rows")):
        for row, job in zip(document[key], jobs, strict=True):
            row["job"] = job
    complete["schema"] = "soccer.rsi.contact_exit_torque_mpc_pilot_complete.v1"
    readback["schema"] = "soccer.rsi.contact_exit_torque_mpc_terminal_readback.v1"
    del readback["actual_postcontact_taskspace_increment_exactly_zero"]
    return args


def test_all_four_mpc_endpoints_use_original_release_not_zero_release_claim():
    args = mpc_fixture()
    assert args[3]["rows"][2]["outcome"]["safe"] is False
    require_four_endpoints(*args)


@pytest.mark.parametrize("tamper", ["jobs", "schema", "mixed", "flag_type", "partial", "outcome"])
def test_mpc_diagnostic_refuses_retagged_partial_or_mixed_studies(tamper):
    args = mpc_fixture()
    if tamper == "jobs":
        args[2]["results"][0]["job"] = JOBS[0]
    elif tamper == "schema":
        args[3]["schema"] = "soccer.rsi.swing_release_ablation_terminal_readback.v1"
    elif tamper == "mixed":
        args[0]["suppress_postcontact_swing_release"] = True
    elif tamper == "flag_type":
        args[0]["use_contact_exit_torque_mpc"] = 1
    elif tamper == "partial":
        args[3]["rows"].pop()
    else:
        args[3]["rows"][2]["outcome"]["safe"] = True
    with pytest.raises(ValueError):
        require_four_endpoints(*args)


@pytest.mark.parametrize(
    "key",
    [
        "max_attempts",
        "new_physical_budget",
        "mode",
        "suppress_postcontact_swing_release",
        "hardware_authorized",
        "promotion_authorized",
        "neural_training_authorized",
    ],
)
def test_wrong_scientific_scope_is_not_reopened(key):
    args = fixture()
    args[0][key] = None
    with pytest.raises(ValueError):
        require_four_endpoints(*args)


@pytest.mark.parametrize(
    "key",
    [
        "whole_input_and_witness_files_verified",
        "shared_payload_logical_seals_rebuilt",
        "prior_independent_replays_authenticated",
        "all_original_first_contact_prefixes_bit_equal",
        "actual_postcontact_taskspace_increment_exactly_zero",
    ],
)
def test_partial_audit_does_not_launch_physics(key):
    args = fixture()
    args[3][key] = False
    with pytest.raises(ValueError):
        require_four_endpoints(*args)


@pytest.mark.parametrize(
    "tamper",
    ["missing", "reordered", "outcome", "receipt_hash", "complete_hash", "optimizer", "substeps"],
)
def test_all_ordered_whole_endpoints_required(tamper):
    args = fixture()
    if tamper == "missing":
        args[2]["results"].pop()
    elif tamper == "reordered":
        args[3]["rows"].reverse()
    elif tamper == "outcome":
        args[3]["rows"][2]["outcome"]["safe"] = True
    elif tamper == "receipt_hash":
        args[3]["rows"][0]["receipt_hash"] = "other"
    elif tamper == "complete_hash":
        args[3]["complete_hash"] = "other"
    elif tamper == "optimizer":
        args[2]["new_optimizer_updates"] = 1
    else:
        args[2]["native_substeps"] = 11999
    with pytest.raises(ValueError):
        require_four_endpoints(*args)
