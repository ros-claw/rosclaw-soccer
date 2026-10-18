from copy import deepcopy
from pathlib import Path

import pytest

from scripts import rsi_read_mesh_contact_native_pilot as audit


def fixture(monkeypatch):
    monkeypatch.setattr(audit, "digest", lambda _: "pinned")
    contract = dict(
        source_hashes=dict(teacher="pinned", forecast="pinned", kinematics="pinned"),
        forecast_budget_steps=6000,
        maximum_offset_nm=12.0,
        maximum_hard_limit_fraction=0.2,
        maximum_offset_change_per_control_frame_nm=2.0,
    )
    counters = dict(integrated_forecast_steps=50)
    report = dict(
        contact_exit_mpc_counters=counters,
        contact_exit_mpc_plans=[dict(mode="PRESSURE_EXIT")]
        + [dict(mode="INACTIVE") for _ in range(299)],
    )
    receipt = dict(contact_exit_mpc_counters=counters, forecast_physical_substeps=50)
    reconstruction = dict(
        schema="soccer.rsi.contact_exit_torque_mpc_reconstruction.v1",
        contract=contract,
        planning_counters=counters,
        frames_reconstructed=300,
        optimization_reconstructed_with_same_pinned_implementation=True,
        independent_optimization_implementation=False,
        locally_bitexact_active_intervals=1,
        chosen_control_forecast_attempts=1,
        chosen_control_attempted_forecast_steps=10,
        chosen_control_integrated_forecast_steps=10,
        source_numeric_replay_tolerance=1e-8,
        hardware_authorized=False,
        promotion_authorized=False,
        source_end_state_hash_matches=0,
    )
    return (
        report,
        dict(contact_exit_torque_mpc=contract),
        receipt,
        dict(contact_exit_torque_mpc_review=reconstruction),
        Path("synthetic-source"),
    )


def test_no_cross_replay_bitexact_claim_required_when_numeric_replay_authenticated(monkeypatch):
    audit.verify_contact_exit_reconstruction(*fixture(monkeypatch))


@pytest.mark.parametrize(
    "field,value",
    [
        ("frames_reconstructed", 299),
        ("locally_bitexact_active_intervals", 0),
        ("chosen_control_forecast_attempts", 0),
        ("chosen_control_attempted_forecast_steps", 0),
        ("chosen_control_integrated_forecast_steps", 0),
        ("source_end_state_hash_matches", 2),
        ("source_numeric_replay_tolerance", 1e-3),
        ("independent_optimization_implementation", True),
        ("hardware_authorized", True),
    ],
)
def test_resealed_counter_or_authority_claim_rejected(monkeypatch, field, value):
    args = list(fixture(monkeypatch))
    args[3] = deepcopy(args[3])
    args[3]["contact_exit_torque_mpc_review"][field] = value
    with pytest.raises(ValueError):
        audit.verify_contact_exit_reconstruction(*args)


def test_contact_exit_jobs_and_prefix_scope_are_not_release_zero_intervention():
    spec = audit.bounded_profile("contact-exit-four")
    assert spec["count"] == 4
    assert [j["pair_id"] for j in spec["jobs"]] == ["raw-8", "raw-56", "context-0", "context-1"]
    assert all(j["role"] == "teacher" and j["option_index"] == 4 for j in spec["jobs"])
    assert "NOT_QUALIFIED_TEACHER" in spec["phase"]
