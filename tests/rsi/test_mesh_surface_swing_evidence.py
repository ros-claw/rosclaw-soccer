"""Synthetic admission/capture tests; not a G1 physical success certificate."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import validate_cpu_foundation_capture
from rosclaw_soccer.rsi.mesh_surface_swing_evidence import require_surface_swing_capture
from rosclaw_soccer.rsi.mesh_surface_swing_probe import inactive_surface_probe_record
from rosclaw_soccer.sim.contracts import hash_json


def capture():
    contract = dict(
        schema="soccer.rsi.mesh_surface_swing.v1",
        world_hash="world",
        strike_direction_world=[1.0, 0.0, 0.0],
        maximum_joint_increment_rad=0.35,
        release_frames=20,
        parent_motor_recomputed_after_new_pre_motor_target=True,
        activation_ceiling="SIM_ONLY",
        trained_policy=False,
        teacher_qualified=False,
        independent_ik_implementation=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    contract["contract_hash"] = hash_json(contract)
    commitment = dict(
        mesh_surface_swing=contract,
        compiled_model_hash="world",
        bounded_control_profile={},
        foundation_observation_capture={},
        default_locomotion_reference_capture={},
        swing_mode="bounded_contact_diagnostic",
        execution_profile="taskspace_plus_motor",
    )
    report = deepcopy(commitment)
    trace = dict(
        mesh_surface_swing_record=np.stack([inactive_surface_probe_record(f) for f in range(300)])[
            :, None
        ],
        force_n=np.zeros((300, 1, 6)),
    )
    return report, commitment, trace


def test_inactive_is_not_fabricated_geometry_and_active_capture_is_bounded():
    args = capture()
    require_surface_swing_capture(*args)
    args[2]["mesh_surface_swing_record"][60, 0, 1:3] = [0, 1]
    require_surface_swing_capture(*args)


@pytest.mark.parametrize(
    "tamper",
    [
        "nan",
        "dtype",
        "shape",
        "clock",
        "side",
        "geometry",
        "increment",
        "early",
        "after_nonfoot",
        "mixed",
        "authority",
        "extra",
    ],
)
def test_corrupt_or_mixed_records_are_rejected(tamper):
    r, c, t = capture()
    a = t["mesh_surface_swing_record"]
    if tamper == "nan":
        a[60, 0, 3] = np.nan
    elif tamper == "dtype":
        t["mesh_surface_swing_record"] = a.astype(np.float32)
    elif tamper == "shape":
        t["mesh_surface_swing_record"] = a[:299]
    elif tamper == "clock":
        a[60, 0, 0] = 61
    elif tamper == "side":
        a[60, 0, 1] = 0
    elif tamper == "geometry":
        a[60, 0, 4] = 0.1
    elif tamper == "increment":
        a[60, 0, 1:3] = [1, 1]
        a[60, 0, 33] = 0.36
    elif tamper == "early":
        a[29, 0, 1:3] = [0, 1]
    elif tamper == "after_nonfoot":
        t["force_n"][59, 0, 4] = 2
        a[60, 0, 1:3] = [0, 1]
    elif tamper == "mixed":
        c["contact_exit_torque_mpc"] = {}
    elif tamper == "authority":
        c["mesh_surface_swing"]["hardware_authorized"] = True
    else:
        t["mesh_surface_unbound"] = np.zeros(1)
    with pytest.raises(ValueError):
        require_surface_swing_capture(r, c, t)


def test_orphan_surface_trace_is_not_accepted_as_ordinary_cpu_capture():
    _, _, trace = capture()
    with pytest.raises(ValueError, match="presence"):
        validate_cpu_foundation_capture({}, {}, trace)


@pytest.mark.parametrize(
    "key,value",
    [
        ("use_mesh_surface_swing", 1),
        ("partition", "REAL"),
        ("record_default_planner_events", False),
        ("record_foundation_observation", False),
        ("foundation_only", True),
        ("proposal_decoder", "unreviewed"),
        ("swing_mode", "legacy"),
        ("approach_mode", "legacy"),
        ("mesh_task_contact_world_hash", None),
        ("mesh_world_review", None),
        ("use_contact_exit_torque_mpc", True),
        ("use_directional_contact_compliance", True),
        ("suppress_postcontact_swing_release", True),
        ("locomotion_mode_probe", 3),
        ("reference_height_probe_m", 0.75),
        ("foundation_approach_probe", True),
        ("generated_seed_episode_binding", True),
        ("neural_model", "other model"),
        ("motor_policy", "other policy"),
        ("cerebellar_model", {}),
    ],
)
def test_bad_surface_request_fails_before_allocating(key, value, monkeypatch):
    from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
    from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
    from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
        RecurrentSamplingEpisodeFactory,
    )

    factory = object.__new__(RecurrentSamplingEpisodeFactory)
    monkeypatch.setattr(
        factory, "_stable", lambda: pytest.fail("must reject before factory access")
    )
    args = SimpleNamespace(
        record_foundation_observation=True,
        foundation_only=False,
        proposal_decoder="reference",
        step_model="pinned",
        body_response_bundle=None,
        cached_proposal_envelope=False,
        neural_model=None,
        motor_policy=None,
    )
    kwargs = dict(
        course=(0, 0, 0, 0),
        partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
        entry_source_path=None,
        consumed_bank_hash=None,
        recurrent_sampling_factory=factory,
        control_profile=BoundedFirstTouchControl(),
        approach_mode="bounded_approach_diagnostic",
        swing_mode="bounded_contact_diagnostic",
        mesh_task_contact_world_hash="world",
        mesh_world_review=object(),
        record_default_planner_events=True,
        use_mesh_surface_swing=True,
    )
    if hasattr(args, key):
        setattr(args, key, value)
    else:
        kwargs[key] = value
    with pytest.raises(ValueError):
        run_native_first_touch_episode(args, **kwargs)
