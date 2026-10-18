import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.directional_compliance_evidence import require_directional_capture
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

WORLD = "sha256:" + "b" * 64


def capture():
    kp = np.arange(11, 40, dtype=np.float64)
    contract = dict(
        schema="soccer.rsi.directional_contact_compliance.v1",
        world_hash=WORLD,
        kp=kp.tolist(),
        kd=np.ones(29).tolist(),
        entry_frame=0,
        forecast_budget_steps=700,
        contraction_fraction=0.5,
        maximum_relative_matrix_change_per_frame=0.1,
        relative_stiffness_eigenvalue_bounds=[0.5, 1.0],
        activation_ceiling="SIM_ONLY",
    )
    contract.update(
        {
            k: False
            for k in (
                "trained_policy",
                "teacher_qualified",
                "passivity_proven",
                "floating_base_stability_proven",
                "damping_changed",
                "physical_hard_torque_limits_changed",
                "hardware_authorized",
                "promotion_authorized",
            )
        }
    )
    contract["contract_hash"] = hash_json(contract)
    commitment = dict(
        compiled_model_hash=WORLD,
        directional_contact_compliance=contract,
        physics=dict(canonical_pd_kp=contract["kp"], canonical_pd_kd=contract["kd"]),
        execution_profile="taskspace_plus_motor",
        bounded_control_profile={},
        foundation_observation_capture={},
        mesh_task_contact_protocol={},
    )
    plans = []
    for frame in range(300):
        p = dict(
            schema="soccer.rsi.directional_compliance_plan.v1",
            frame=frame,
            mode="INACTIVE",
            side=None,
            identity=True,
            forecast_substeps=0,
            stiffness_matrix_hash=hash_bytes(np.diag(kp).tobytes()),
            expected_end_state_hash=None,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        p["report_hash"] = hash_json(p)
        plans.append(p)
    report = dict(
        directional_contact_compliance=deepcopy(contract),
        directional_compliance_plans=plans,
        directional_compliance_counters=dict(
            frames_planned=300,
            forecast_attempts=0,
            reserved_forecast_steps=0,
            attempted_forecast_steps=0,
            integrated_forecast_steps=0,
            maximum_forecast_steps=700,
        ),
    )
    trace = dict(
        directional_compliance_stiffness_nm_per_rad=np.tile(np.diag(kp), (300, 1, 1, 1)),
        directional_compliance_requested_delta_nm=np.zeros((300, 1, 10, 29)),
        directional_compliance_applied_delta_nm=np.zeros((300, 1, 10, 29)),
        directional_compliance_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
        pd_requested_torque_nm=np.zeros((300, 1, 10, 29)),
        torque_nm=np.zeros((300, 1, 10, 29)),
        joint_target_rad=np.zeros((300, 1, 29)),
        joint_position_rad=np.zeros((300, 1, 29)),
        joint_velocity_rad_s=np.zeros((300, 1, 29)),
    )
    return report, commitment, trace


def test_exact_inactive_whole_episode_capture():
    require_directional_capture(*capture())


@pytest.mark.parametrize(
    "bad", ["mixed", "slew", "torso", "nan", "asymmetry", "delta", "count", "seal", "extra"]
)
def test_unsealed_or_unbounded_gain_evidence_is_not_accepted(bad):
    r, c, t = capture()
    if bad == "mixed":
        c["contact_exit_torque_mpc"] = {}
    elif bad == "slew":
        t["directional_compliance_stiffness_nm_per_rad"][0, 0, 0, 0] *= 0.5
    elif bad == "torso":
        t["directional_compliance_stiffness_nm_per_rad"][0, 0, 12, 12] *= 0.99
    elif bad == "nan":
        t["directional_compliance_stiffness_nm_per_rad"][0, 0, 0, 0] = np.nan
    elif bad == "asymmetry":
        t["directional_compliance_stiffness_nm_per_rad"][0, 0, 0, 1] = 1
    elif bad == "delta":
        t["directional_compliance_requested_delta_nm"][0, 0, 0, 0] = 0.01
    elif bad == "count":
        r["directional_compliance_counters"]["integrated_forecast_steps"] = 10
    elif bad == "seal":
        r["directional_compliance_plans"][0]["identity"] = False
    else:
        t["directional_compliance_extra"] = np.zeros(1)
    with pytest.raises(ValueError):
        require_directional_capture(r, c, t)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_directional_contact_compliance", 1),
        ("partition", "REAL"),
        ("record_foundation_observation", False),
        ("record_default_planner_events", False),
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
        ("use_contact_exit_torque_mpc", True),
        ("sampling_factory", object()),
        ("neural_model", object()),
    ],
)
def test_bad_directional_request_rejected_before_allocation(field, value):
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
    opts = dict(
        partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
        control_profile=BoundedFirstTouchControl(),
        mesh_task_contact_world_hash=WORLD,
        mesh_world_review=object.__new__(ReviewedContactWorld),
        recurrent_sampling_factory=object.__new__(RecurrentSamplingEpisodeFactory),
        approach_mode="bounded_approach_diagnostic",
        swing_mode="bounded_contact_diagnostic",
        record_default_planner_events=True,
        use_directional_contact_compliance=True,
    )
    (args if field in args else opts)[field] = value
    with pytest.raises(ValueError):
        run_native_first_touch_episode(
            argparse.Namespace(**args),
            course=(2.3, 0.0, -0.4),
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            **opts,
        )
