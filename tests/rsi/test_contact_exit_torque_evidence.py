import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.contact_exit_torque_evidence import (
    compare_plans,
    require_contact_exit_capture,
)
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.sim.contracts import hash_json

WORLD = "sha256:" + "b" * 64


def capture():
    contract = dict(
        schema="soccer.rsi.contact_exit_torque_mpc.v1",
        world_hash=WORLD,
        trained_policy=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        forecast_budget_steps=6000,
    )
    contract["contract_hash"] = hash_json(contract)
    commitment = dict(
        contact_exit_torque_mpc=contract,
        compiled_model_hash=WORLD,
        bounded_control_profile={},
        foundation_observation_capture={},
        mesh_task_contact_protocol={},
        execution_profile="taskspace_plus_motor",
    )
    plans = []
    for frame in range(300):
        plan = dict(
            schema="soccer.rsi.contact_exit_torque_plan.v1",
            frame=frame,
            mode="INACTIVE",
            side=None,
            choices=[],
            selected_index=None,
            forecast_substeps=0,
            expected_end_state_hash=None,
            no_safe_choice_fallback=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        plan["report_hash"] = hash_json(plan)
        plans.append(plan)
    report = dict(
        contact_exit_torque_mpc=deepcopy(contract),
        contact_exit_mpc_plans=plans,
        contact_exit_mpc_counters=dict(
            forecast_attempts=0,
            attempted_forecast_steps=0,
            integrated_forecast_steps=0,
            maximum_forecast_steps=6000,
        ),
    )
    trace = dict(
        contact_exit_mpc_requested_offset_nm=np.zeros((300, 1, 29)),
        contact_exit_mpc_applied_offset_nm=np.zeros((300, 1, 10, 29)),
        contact_exit_mpc_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
        pd_requested_torque_nm=np.zeros((300, 1, 10, 29)),
        torque_nm=np.zeros((300, 1, 10, 29)),
    )
    return report, commitment, trace


def test_inactive_full_episode_evidence():
    require_contact_exit_capture(*capture())


@pytest.mark.parametrize(
    "bad",
    ["offset", "applied", "nan", "dtype", "steps", "counter", "seal", "mixed", "missing", "extra"],
)
def test_tampered_capture_rejected(bad):
    report, commitment, trace = capture()
    if bad == "offset":
        trace["contact_exit_mpc_requested_offset_nm"][1, 0, 0] = 3
    elif bad == "applied":
        trace["contact_exit_mpc_applied_offset_nm"][1, 0, 0, 0] = 0.1
    elif bad == "nan":
        trace["contact_exit_mpc_requested_offset_nm"][0, 0, 0] = np.nan
    elif bad == "dtype":
        trace["contact_exit_mpc_forecast_steps"] = trace["contact_exit_mpc_forecast_steps"].astype(
            float
        )
    elif bad == "steps":
        trace["contact_exit_mpc_forecast_steps"][0, 0, 0] = 10
    elif bad == "counter":
        report["contact_exit_mpc_counters"]["integrated_forecast_steps"] = 10
    elif bad == "seal":
        report["contact_exit_mpc_plans"][0]["mode"] = "PRESSURE_EXIT"
    elif bad == "mixed":
        commitment["swing_release_ablation"] = {}
    elif bad == "missing":
        del trace["contact_exit_mpc_applied_offset_nm"]
    else:
        trace["contact_exit_mpc_undeclared"] = np.zeros(1)
    with pytest.raises(ValueError):
        require_contact_exit_capture(report, commitment, trace)


def test_optimization_comparison_ignores_only_hash_roundoff_not_decisions():
    original = dict(offset_nm=[1.0], forecast_cost=2.0, selected_index=0, donor_state_hash="a")
    replay = dict(original, forecast_cost=2.0 + 1e-10, donor_state_hash="b")
    compare_plans(replay, original)
    for bad in (dict(replay, selected_index=1), dict(replay, offset_nm=[1.01])):
        with pytest.raises(ValueError):
            compare_plans(bad, original)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_contact_exit_torque_mpc", 1),
        ("partition", "REAL"),
        ("record_foundation_observation", False),
        ("foundation_only", True),
        ("proposal_decoder", "compiled"),
        ("step_model", None),
        ("control_profile", None),
        ("mesh_world_review", None),
        ("mesh_task_contact_world_hash", None),
        ("approach_mode", "legacy"),
        ("swing_mode", "legacy"),
        ("postcontact_profile", object()),
        ("cerebellar_model", {}),
        ("generated_seed_episode_binding", True),
        ("suppress_postcontact_swing_release", True),
        ("sampling_factory", object()),
        ("neural_model", object()),
        ("motor_policy", object()),
    ],
)
def test_mixed_teacher_request_rejected_before_allocating_or_opening_models(field, value):
    args = dict(
        record_foundation_observation=True,
        foundation_only=False,
        proposal_decoder="reference",
        step_model=Path("never-opened"),
        body_response_bundle=None,
        cached_proposal_envelope=False,
        neural_model=None,
        motor_policy=None,
    )
    options = dict(
        partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
        control_profile=BoundedFirstTouchControl(),
        mesh_task_contact_world_hash=WORLD,
        mesh_world_review=object.__new__(ReviewedContactWorld),
        recurrent_sampling_factory=object.__new__(RecurrentSamplingEpisodeFactory),
        approach_mode="bounded_approach_diagnostic",
        swing_mode="bounded_contact_diagnostic",
        use_contact_exit_torque_mpc=True,
    )
    (args if field in args else options)[field] = value
    with pytest.raises(ValueError):
        run_native_first_touch_episode(
            argparse.Namespace(**args),
            course=(2.3, 0.0, -0.4),
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            **options,
        )
