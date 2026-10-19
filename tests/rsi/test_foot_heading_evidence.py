"""Synthetic causal composition/release, not qualified G1 dynamics."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
from rosclaw_soccer.rsi.foot_heading_evidence import (
    FIELDS,
    PREFIX,
    TAG,
    FootHeadingCpuReconstruction,
    compose_heading_step,
    require_heading_capture,
)
from rosclaw_soccer.rsi.original_ik_planner_evidence import OriginalIkCpuReconstruction
from tests.rsi.test_initial_contact_heading import kinematic_fixture
from tests.rsi.test_original_ik_planner_evidence import captured as navigation_capture


def fixture(enabled=True, contact=45):
    import mujoco

    model, _, _ = kinematic_fixture()
    model.body_pos[model.body("left_ankle_roll_link").id, 2] += 0.05
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.cos(np.pi / 12), 0, 0, np.sin(np.pi / 12)]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    c, trace = navigation_capture(0.08, contact)
    probe = FootHeadingContactProbe(enabled)
    c[TAG] = probe.contract()
    return model, data, c, trace, probe


def reconstruct(enabled=True, contact=45):
    model, data, c, t, probe = fixture(enabled, contact)
    replay = FootHeadingCpuReconstruction(model, probe)
    records = {PREFIX + k: [] for k in FIELDS}
    for frame in range(300):
        delta = replay.before_step(data, frame, data.xpos, np.zeros(29))
        assert replay.last_record is not None
        for k, value in replay.last_record.items():
            records[k].append(value.copy()[None])
        if enabled and 30 <= frame <= (299 if contact is None else contact):
            assert delta[5] == pytest.approx(-0.1)
        elif enabled and contact is not None and frame > contact and frame >= 30:
            assert delta[5] == pytest.approx(-0.1 * max(0, 1 - (frame - contact) / 20))
        else:
            assert not delta.any()
        replay.after_step(frame, t["force_n"][frame, 0])
    t.update({k: np.asarray(values) for k, values in records.items()})
    return c, t, replay


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("contact", [None, 45, 299])
def test_causal_actual_model_ik_heading_and_combined_release_reconstruct_all300(enabled, contact):
    c, t, replay = reconstruct(enabled, contact)
    r = require_heading_capture(c, c, t)
    assert r["frames_reconstructed"] == 300
    assert not r["actual_cpu_geometry_reconstructed_here"]
    cpu = replay.review()
    assert cpu["heading_and_combined_release_from_actual_cpu_state_reconstructed"]
    assert cpu["original_ik_from_actual_cpu_state_reconstructed"]
    assert not cpu["independent_heading_implementation"]
    assert not cpu["independent_ik_implementation"]
    assert cpu["new_physical_steps"] == cpu["optimizer_updates"] == 0
    if contact == 45 and enabled:
        assert t[PREFIX + "combined_increment_rad"][46, 0, 5] == pytest.approx(-0.095)
        assert not t[PREFIX + "additional_applied_rad"][46:].any()


def test_off_retains_original_ik_and_release_bits_without_dummy_reference_model():
    model, data, c, t, probe = fixture(False)
    original, heading = (
        OriginalIkCpuReconstruction(model),
        FootHeadingCpuReconstruction(model, probe),
    )
    for frame in range(300):
        a = original.before_step(data, frame, data.xpos, np.zeros(29))
        b = heading.before_step(data, frame, data.xpos, np.zeros(29))
        assert a.tobytes() == b.tobytes()
        for controller in (original, heading):
            controller.after_step(frame, t["force_n"][frame, 0])
    assert "reference_student_model" not in c


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "legacy_model",
        "unknown_source",
        "partial",
        "nan",
        "dtype",
        "phase",
        "enabled",
        "force",
        "post_heading",
        "frame",
        "combined",
        "projection",
        "rank",
    ],
)
def test_missing_mixed_noncausal_or_unbounded_captures_are_not_accepted(mutation):
    c, t, _ = reconstruct()
    report = copy.deepcopy(c)
    if mutation == "missing":
        t.pop(PREFIX + "state")
    elif mutation == "extra":
        t[PREFIX + "unknown"] = np.zeros((300, 1, 1))
    elif mutation == "legacy_model":
        c["reference_student_model"] = None
    elif mutation == "unknown_source":
        c[TAG]["source_hash"] = "wrong"
    elif mutation == "partial":
        t[PREFIX + "state"] = t[PREFIX + "state"][:-1]
    elif mutation == "nan":
        t[PREFIX + "state"][40, 0, 5] = np.nan
    elif mutation == "dtype":
        t[PREFIX + "state"] = t[PREFIX + "state"].astype(np.float32)
    elif mutation == "phase":
        t[PREFIX + "state"][40, 0, 2] = 2
    elif mutation == "enabled":
        t[PREFIX + "state"][40, 0, 3] = 0
    elif mutation == "force":
        t["force_n"][31, 0, 0] = 2
    elif mutation == "post_heading":
        t[PREFIX + "additional_applied_rad"][50, 0, 5] = 0.01
    elif mutation == "frame":
        t[PREFIX + "state"][40, 0, 0] = 41
    elif mutation == "combined":
        t[PREFIX + "combined_increment_rad"][40, 0, 0] += 0.01
    elif mutation == "projection":
        t[PREFIX + "linear_predicted_displacement_m"][40, 0, 0] = 0.001
    else:
        t[PREFIX + "state"][40, 0, 9] = 4
    with pytest.raises(ValueError):
        require_heading_capture(report, c, t)


def test_current_frame_cannot_observe_its_own_future_contact_or_finish_a_partial_review():
    model, data, _, _, probe = fixture()
    with pytest.raises(ValueError):
        compose_heading_step(
            probe,
            model,
            data,
            frame=40,
            side=0,
            contact_frame=40,
            navigation_target=np.zeros(29),
            increment=np.zeros(6),
        )
    replay = FootHeadingCpuReconstruction(model, probe)
    with pytest.raises(ValueError):
        replay.review()
    replay.before_step(data, 0, data.xpos, np.zeros(29))
    with pytest.raises(ValueError):
        replay.before_step(data, 0, data.xpos, np.zeros(29))
