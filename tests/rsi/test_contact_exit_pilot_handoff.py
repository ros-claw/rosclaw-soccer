"""Synthetic classification, not physical qualification or promotion evidence."""

import json
import sys

import pytest

from rosclaw_soccer.rsi.contact_exit_pilot_handoff import contact_exit_pilot_handoff
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_contact_exit_pilot_handoff as cli


def fixture(surface=False):
    count = 6 if surface else 4
    baselines = [dict(pair_id=str(i), receipt_hash=f"old{i}") for i in range(count)]
    outcomes = [dict(task_pass=i != 2, safe=i != 2) for i in range(count)]
    results = [
        dict(pair_id=str(i), receipt_hash=f"new{i}", outcome=o) for i, o in enumerate(outcomes)
    ]
    complete = dict(
        schema="soccer.rsi.mesh_surface_swing_pilot_complete.v1"
        if surface
        else "soccer.rsi.directional_contact_compliance_pilot_complete.v1",
        report_hash="complete",
        paired_development={"rescued_pairs": 0},
        results=[dict(receipt_hash="off-native")] + results if surface else results,
        new_optimizer_updates=0,
    )
    summary = dict(
        schema="soccer.rsi.mesh_surface_measured_summary.v1"
        if surface
        else "soccer.rsi.directional_compliance_measured_summary.v1",
        report_hash="summary",
        complete_hash="complete",
        paired_development=complete["paired_development"],
        rows=[
            dict(pair_id=str(i), baseline=dict(outcome=o), evaluated=dict(outcome=o))
            for i, o in enumerate(outcomes)
        ],
        teacher_qualified=False,
        new_optimizer_updates=0,
        decision="REJECT_LOCAL_TEACHER",
    )
    rows = []
    # Four current, three paired historical, one unrelated historical endpoint.
    # Historical C1 has no dense replay; missing must stay missing.
    for identity, outcome in [
        *((f"new{i}", o) for i, o in enumerate(outcomes)),
        *((f"old{i}", outcomes[i]) for i in range(3)),
        ("other-history", outcomes[2]),
    ]:
        rows.append(
            dict(
                source_receipt_hash=identity,
                original_outcome=outcome.copy(),
                metrics=dict(
                    first_nonfoot_robot_contact_substep=1000 if identity.endswith("2") else None,
                    total_robot_downward_impulse_n_s=30.0,
                    maximum_linear_newton_residual_n=1e-13,
                    summed_active_foot_duration_s=0.4,
                    first_forward_to_backward_crossing_after_initial_20ms_substep=850,
                ),
            )
        )
    dense = dict(
        schema="soccer.rsi.contact_exit_metrics_readback.v1",
        report_hash="dense",
        rows=rows,
        new_optimizer_updates=0,
    )
    return summary, complete, dense, baselines


def test_six_current_endpoints_are_all_reviewed_without_inventing_missing_historical_pairs():
    result = contact_exit_pilot_handoff(*fixture(True))
    assert result["schema"] == "soccer.rsi.mesh_surface_research_handoff.v1"
    assert result["dense_pair_count"] == 3
    assert result["dense_single_endpoint_count"] == 4
    assert result["dense_current_endpoint_count"] == 6
    assert len(result["rows"]) == 6
    assert result["all_six_current_endpoints_have_dense_contact_review"] is True
    assert "all_four_current_endpoints_have_dense_contact_review" not in result
    assert result["teacher_qualified"] is False


def test_all_six_dense_pairs_can_be_completed_with_only_three_missing_historical_replays():
    args = fixture(True)
    rows = args[2]["rows"]
    for i in (3, 4, 5):
        old = dict(rows[i], source_receipt_hash=f"old{i}")
        rows.append(old)
    result = contact_exit_pilot_handoff(*args)
    assert result["dense_pair_count"] == 6
    assert result["dense_single_endpoint_count"] == 1
    assert result["teacher_qualified"] is False
    rows[-1]["source_receipt_hash"] = "some-unpaired-extra"
    with pytest.raises(ValueError, match="all six"):
        contact_exit_pilot_handoff(*args)


def test_failure_preserved_three_pairs_not_four_no_retry_or_promotion():
    result = contact_exit_pilot_handoff(*fixture())
    assert result["dense_pair_count"] == 3
    assert result["dense_single_endpoint_count"] == 2
    assert result["rows"][3]["baseline"] is None
    assert result["current_nonfoot_ball_contact_cases"] == ["2"]
    assert result["original_task_failure_cases"] == ["2"]
    assert (
        result["next_agent_action"] == "REJECT_LOCAL_TEACHER_ANALYZE_CONTACT_AND_SUPPORT_REDESIGN"
    )
    assert result["automatic_same_setting_retry_authorized"] is False
    assert result["teacher_qualified"] is False
    assert result["agent_reasoning_still_required"] is True


@pytest.mark.parametrize("nonfoot", [True, False])
def test_local_success_requires_expansion_or_body_review(nonfoot):
    args = fixture()
    args[0]["decision"] = "LOCAL_PILOT_PASS_EXPANSION_REQUIRED"
    if not nonfoot:
        args[2]["rows"][2]["metrics"]["first_nonfoot_robot_contact_substep"] = None
    result = contact_exit_pilot_handoff(*args)
    assert result["next_agent_action"] == (
        "REVIEW_ALL_BODY_CONTACT_BEFORE_EXPANSION"
        if nonfoot
        else "DECLARE_EXPANDED_RETENTION_AND_PERTURBATION_STUDY"
    )
    assert result["promotion_authorized"] is False


@pytest.mark.parametrize(
    "tamper",
    [
        "missing",
        "duplicate",
        "identity",
        "outcome",
        "nan",
        "negative",
        "clock",
        "residual",
        "study",
        "pair",
    ],
)
def test_no_handoff_from_incomplete_or_inconsistent_evidence(tamper):
    args = fixture()
    rows = args[2]["rows"]
    if tamper == "missing":
        rows.pop()
    elif tamper == "duplicate":
        rows[3]["source_receipt_hash"] = rows[0]["source_receipt_hash"]
    elif tamper == "identity":
        rows[0]["source_receipt_hash"] = "some-other-receipt"
    elif tamper == "outcome":
        rows[0]["original_outcome"]["task_pass"] = False
    elif tamper == "nan":
        rows[0]["metrics"]["total_robot_downward_impulse_n_s"] = float("nan")
    elif tamper == "negative":
        rows[0]["metrics"]["summed_active_foot_duration_s"] = -1
    elif tamper == "clock":
        rows[0]["metrics"]["first_nonfoot_robot_contact_substep"] = 3000
    elif tamper == "residual":
        rows[0]["metrics"]["maximum_linear_newton_residual_n"] = 1e-3
    elif tamper == "study":
        args[0]["complete_hash"] = "other-study"
    else:
        args[3][0]["pair_id"] = args[3][1]["pair_id"]
    with pytest.raises(ValueError):
        contact_exit_pilot_handoff(*args)


@pytest.mark.parametrize("surface", [False, True])
@pytest.mark.parametrize("tamper", [None, "file", "chain", "existing"])
def test_cli_binds_whole_inputs_and_writes_once(tmp_path, monkeypatch, tamper, surface):
    summary, complete, dense, baselines = fixture(surface)

    def write_sealed(name, value):
        value["report_hash"] = hash_json({k: v for k, v in value.items() if k != "report_hash"})
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    upstream = tmp_path / "witness"
    upstream.write_bytes(b"original witness")
    write_sealed("complete.json", complete)
    summary.update(
        complete_hash=complete["report_hash"],
        terminal_readback_hash="audit",
        input_file_hashes={str(upstream): cli.digest(upstream)},
    )
    sp = write_sealed("summary.json", summary)
    dp = write_sealed("dense.json", dense)
    diagnostic = dict(
        schema="soccer.rsi.mesh_surface_contact_diagnostic_complete.v1"
        if surface
        else "soccer.rsi.directional_compliance_contact_diagnostic_complete.v1",
        metrics_hash=dense["report_hash"],
        upstream_readback_hash="audit",
        new_fixed_control_replay_substeps=18000 if surface else 12000,
        new_closed_loop_policy_trials=0,
        new_optimizer_updates=0,
    )
    if tamper == "chain":
        diagnostic["upstream_readback_hash"] = "different audit"
    diag = write_sealed("diagnostic.json", diagnostic)
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            dict(
                output_root=str(tmp_path),
                baseline_receipts=baselines,
                use_mesh_surface_swing=surface,
            )
        )
    )
    output = tmp_path / "handoff.json"
    if tamper == "existing":
        output.write_bytes(b"existing")
    if tamper == "file":
        upstream.write_bytes(b"changed witness")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "handoff",
            "--config",
            str(cfg),
            "--summary",
            str(sp),
            "--dense",
            str(dp),
            "--diagnostic",
            str(diag),
            "--output",
            str(output),
        ],
    )
    if tamper is not None:
        with pytest.raises(ValueError):
            cli.main()
        assert output.read_bytes() == b"existing" if tamper == "existing" else not output.exists()
    else:
        cli.main()
        result = json.loads(output.read_text())
        assert result["report_hash"] == hash_json(
            {k: v for k, v in result.items() if k != "report_hash"}
        )
        assert result["dense_pair_count"] == 3
        assert result["new_physical_steps"] == 0
        assert result["automatic_scientific_successor_launched"] is False
