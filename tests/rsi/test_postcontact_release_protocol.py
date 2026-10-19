"""Numeric protocol tests only, no real G1 execution or teacher evidence."""

import copy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from scripts.rsi_postcontact_release_inputs import require_scope
from scripts.rsi_postcontact_release_trace import check_release_trace


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="postcontact-release-native14-diagnostic-v1",
        source_cases=6,
        local_context_index=1,
        original_binding_index=0,
        diagnostic_physical_contexts=1,
        new_rollouts=14,
        native_step_budget=42000,
        forecast_step_budget=0,
        executed_control_foundation_pairs=4200,
        official_horizon_frames=300,
        post_contact_frames=60,
        maximum_release_duration_frames=3,
        source_campaign_completion_not_required=True,
        new_optimizer_updates=0,
        private_fresh_accessed=False,
        training_authorized=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
        action_names=[*ACTION_NAMES, "baseline"],
    )


def test_fixed_whole_bank():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_missing_or_changed_scope_rejected(key):
    c = scope()
    del c[key]
    with pytest.raises(ValueError):
        require_scope(c)
    c = scope()
    v = c[key]
    c[key] = int(v) if type(v) is bool else float(v) if type(v) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(c)


def arrays(name="lift_plus"):
    action = np.zeros((300, 1, 29))
    if name != "baseline":
        action[41:44, 0, 3] = 0.1
    control = np.tile(action[:, :, None], (1, 1, 10, 1))
    force = np.zeros((300, 1, 6))
    force[40, 0, 0] = 3
    root = np.zeros((300, 1, 7))
    root[:, :, 2] = 0.793
    trace = dict(
        postcontact_release_requested_offset_nm=action,
        postcontact_release_applied_offset_nm=control.copy(),
        postcontact_release_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
        pd_requested_torque_nm=np.zeros((300, 1, 10, 29)),
        torque_nm=control.copy(),
        actual_actuator_force_nm=control.copy(),
        force_n=force,
        root_pose_xyzw_m=root,
    )
    r = dict(
        action_name=name,
        maximum_action_duration_frames=3,
        selection_from_completed_contact_only=True,
        future_source_trace_consumed=False,
        trigger_frame=41,
        measured_contact_side=0,
        cancelled=False,
        action_available=True,
    )
    return trace, dict(postcontact_release=r)


def test_causal_trigger_and_measured_force_attribution():
    t, r = arrays()
    original = copy.deepcopy(t)
    review = check_release_trace(t, r, action_name="lift_plus")
    assert (
        review["observed_trigger_frame"]
        == review["first_requested_correction_frame"]
        == review["first_physically_applied_correction_frame"]
        == 41
    )
    assert review["nonzero_requested_frames"] == 3
    assert all(t[k].tobytes() == a.tobytes() for k, a in original.items())


def test_saturated_requested_action_is_not_falsely_counted_as_actual_authority():
    from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS

    t, r = arrays()
    t["pd_requested_torque_nm"][:] = 1000
    t["torque_nm"][:] = G1_HARD_TORQUE_LIMITS
    t["actual_actuator_force_nm"][:] = G1_HARD_TORQUE_LIMITS
    t["postcontact_release_applied_offset_nm"][:] = 0
    d = check_release_trace(t, r, action_name="lift_plus")
    assert d["first_requested_correction_frame"] == 41
    assert d["first_physically_applied_correction_frame"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        "trigger",
        "side",
        "cancel",
        "lease",
        "cap",
        "support",
        "applied",
        "actual",
        "nan",
        "dtype",
        "forecast",
        "old_profile",
    ],
)
def test_causal_or_physical_mismatch_rejected(mutate):
    t, r = arrays()
    if mutate in ("trigger", "side", "cancel"):
        key, value = {
            "trigger": ("trigger_frame", 42),
            "side": ("measured_contact_side", 1),
            "cancel": ("cancelled", True),
        }[mutate]
        r["postcontact_release"][key] = value
    elif mutate == "lease":
        t["postcontact_release_requested_offset_nm"][43] = 0
    elif mutate == "cap":
        t["postcontact_release_requested_offset_nm"][41, 0, 3] = 100
    elif mutate == "support":
        t["postcontact_release_requested_offset_nm"][41, 0, 9] = 0.1
    elif mutate == "applied":
        t["postcontact_release_applied_offset_nm"][41, 0, 0, 3] += 0.1
    elif mutate == "actual":
        t["actual_actuator_force_nm"][42, 0, 3, 3] += 0.1
    elif mutate == "nan":
        t["force_n"][1] = np.nan
    elif mutate == "dtype":
        t["force_n"] = t["force_n"].astype(np.float32)
    elif mutate == "forecast":
        t["postcontact_release_forecast_steps"][20] = 40
    else:
        t["early_impact_feedback_requested_offset_nm"] = np.zeros((300, 1, 29))
    with pytest.raises(ValueError):
        check_release_trace(t, r, action_name="lift_plus")


def test_observed_cancellation_shortens_lease_without_fake_completed_three_frames():
    t, r = arrays()
    t["force_n"][41, 0, 4] = 2
    r["postcontact_release"]["cancelled"] = True
    for k in (
        "postcontact_release_requested_offset_nm",
        "postcontact_release_applied_offset_nm",
        "torque_nm",
        "actual_actuator_force_nm",
    ):
        t[k][42:44] = 0
    d = check_release_trace(t, r, action_name="lift_plus")
    assert d["nonzero_requested_frames"] == 1 and d["cancelled"]


def test_preflight_no_model_or_physics_allocation(monkeypatch, capsys):
    from scripts import rsi_postcontact_release_authority as mod

    p = SimpleNamespace(
        source=Path(mod.__file__).resolve().parent.parent,
        cfg=dict(deadline_unix=time.time() + 3600),
    )
    monkeypatch.setattr(mod, "prepare", lambda path: p)
    monkeypatch.setattr(sys, "argv", ["release", "--config", "fixture.json", "--preflight"])
    mod.main()
    assert "NO_FULL_MODELS_OR_PHYSICS" in capsys.readouterr().out
