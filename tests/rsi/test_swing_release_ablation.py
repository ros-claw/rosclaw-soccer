import argparse
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.rsi.swing_release_ablation import (
    release_ablation_contract,
    require_zero_postcontact_swing,
)

WORLD = "sha256:" + "b" * 64


def capture():
    commitment = dict(
        compiled_model_hash=WORLD,
        bounded_control_profile={},
        foundation_observation_capture={},
        execution_profile="taskspace_plus_motor",
        swing_mode="bounded_contact_diagnostic",
        swing_release_ablation=release_ablation_contract(WORLD),
    )
    trace = dict(
        force_n=np.zeros((300, 1, 6)), taskspace_joint_increment_rad=np.zeros((300, 1, 29))
    )
    trace["force_n"][10, 0, 0] = 4
    trace["taskspace_joint_increment_rad"][:11] = 0.1
    return commitment, trace


def test_zero_after_contact_only_and_precontact_motion_allowed():
    commitment, trace = capture()
    require_zero_postcontact_swing(commitment, trace)
    assert commitment["swing_release_ablation"]["optimized_recovery_claimed"] is False


def test_no_contact_is_valid_data_not_infrastructure_failure():
    commitment, trace = capture()
    trace["force_n"][:] = 0
    trace["taskspace_joint_increment_rad"][:] = 0.1
    require_zero_postcontact_swing(commitment, trace)


@pytest.mark.parametrize(
    "bad", ["increment", "nonfinite", "mixed", "wrong_contract", "capture", "profile"]
)
def test_ablation_trace_and_contract_tampering_rejected(bad):
    commitment, trace = capture()
    if bad == "increment":
        trace["taskspace_joint_increment_rad"][11, 0, 0] = 1e-15
    elif bad == "nonfinite":
        trace["force_n"][0, 0, 0] = np.nan
    elif bad == "mixed":
        commitment["cerebellar_torque_contract"] = {}
    elif bad == "wrong_contract":
        commitment["swing_release_ablation"]["original_release_frames"] = 1
    elif bad == "capture":
        del commitment["foundation_observation_capture"]
    else:
        commitment["execution_profile"] = "foundation_only"
    with pytest.raises(ValueError):
        require_zero_postcontact_swing(commitment, trace)


@pytest.mark.parametrize(
    "field,value",
    [
        ("suppress_postcontact_swing_release", 1),
        ("partition", "REAL"),
        ("record_foundation_observation", False),
        ("foundation_only", True),
        ("proposal_decoder", "compiled"),
        ("step_model", None),
        ("control_profile", None),
        ("mesh_world_review", None),
        ("mesh_task_contact_world_hash", None),
        ("swing_mode", "legacy"),
        ("postcontact_profile", object()),
        ("cerebellar_model", {}),
        ("generated_seed_episode_binding", True),
        ("sampling_factory", object()),
    ],
)
def test_bad_request_rejected_before_factory_or_model_allocation(field, value):
    args = dict(
        record_foundation_observation=True,
        foundation_only=False,
        proposal_decoder="reference",
        step_model=Path("never-opened"),
        body_response_bundle=None,
        cached_proposal_envelope=False,
    )
    options = dict(
        partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
        control_profile=BoundedFirstTouchControl(),
        mesh_task_contact_world_hash=WORLD,
        mesh_world_review=object.__new__(ReviewedContactWorld),
        recurrent_sampling_factory=object.__new__(RecurrentSamplingEpisodeFactory),
        swing_mode="bounded_contact_diagnostic",
        suppress_postcontact_swing_release=True,
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
