"""MuJoCo state/phase tests with synthetic Jacobians, not physical skill proof."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi.impact_torque_screen import CAPS
from rosclaw_soccer.rsi.postcontact_release_control import (
    ACTION_NAMES,
    PostContactReleaseControl,
)
from rosclaw_soccer.sim.contracts import hash_bytes
from tests.physics.test_g1_feedback_forecast import world
from tests.rsi.test_impact_torque_screen import sample


@pytest.fixture(autouse=True)
def synthetic_sampler(monkeypatch):
    from rosclaw_soccer.rsi import postcontact_release_control as mod

    class Sampler:
        def __init__(self, model, **kwargs):
            self.model = model

        def sample(self, data):
            assert data.model is self.model
            return sample()

    monkeypatch.setattr(mod, "G1JointTaskKinematicSampler", Sampler)


def setup(name="lift_plus"):
    m, _, _ = world()
    binary = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, binary)
    c = PostContactReleaseControl(m, world_hash=hash_bytes(binary.tobytes()), action_name=name)
    d = mujoco.MjData(m)
    d.qpos[:7] = (0, 0, 0.793, 1, 0, 0, 0)
    d.qpos[7:36] = np.linspace(-0.2, 0.2, 29)
    d.qpos[36:] = (2, 0, 0.13, 1, 0, 0, 0)
    mujoco.mj_forward(m, d)
    return c, d


def call(c, d, frame, forces=None, height=0.793):
    d.time = frame * 0.02
    return c.plan(
        d,
        frame=frame,
        previous_contact_forces=np.zeros(6) if forces is None else forces,
        pelvis_z_m=height,
    )


def test_release_uses_completed_contact_then_three_frames_and_never_retriggers():
    c, d = setup()
    for frame in range(31):
        assert not np.any(call(c, d, frame).offset_nm)
    original = {k: getattr(d, k).copy() for k in ("qpos", "qvel", "xpos", "xquat", "cvel")}
    force = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    out = call(c, d, 31, force)
    assert out.report["trigger_frame"] == 31 and out.report["measured_contact_side"] == 1
    assert out.report["action_available"] is True
    assert (
        np.any(out.offset_nm[6:12])
        and not np.any(out.offset_nm[:6])
        and not np.any(out.offset_nm[12:])
    )
    assert np.all(abs(out.offset_nm) <= CAPS)
    assert all(getattr(d, k).tobytes() == a.tobytes() for k, a in original.items())
    held = out.offset_nm.copy()
    out.offset_nm[:] = 0  # Returned command cannot change private lease state.
    np.testing.assert_array_equal(call(c, d, 32).offset_nm, held)
    np.testing.assert_array_equal(call(c, d, 33).offset_nm, held)
    for frame in range(34, 300):
        assert not np.any(call(c, d, frame, force).offset_nm)
    review = c.review()
    assert review["selection_from_completed_contact_only"] is True
    assert review["teacher_qualified"] is review["hardware_authorized"] is False


def test_baseline_is_real_zero_control_but_keeps_observed_trigger_metadata():
    c, d = setup("baseline")
    for frame in range(31):
        call(c, d, frame)
    out = call(c, d, 31, np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert out.report["trigger_frame"] == 31 and out.report["action_available"]
    assert not np.any(out.offset_nm)


@pytest.mark.parametrize("bad", ["nonfoot", "height"])
def test_observed_safety_event_cancels_lease_without_later_resumption(bad):
    c, d = setup()
    for frame in range(31):
        call(c, d, frame)
    assert np.any(call(c, d, 31, np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])).offset_nm)
    force = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0]) if bad == "nonfoot" else np.zeros(6)
    assert not np.any(call(c, d, 32, force, height=0.6 if bad == "height" else 0.793).offset_nm)
    assert not np.any(call(c, d, 33, np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])).offset_nm)
    assert c.cancelled


@pytest.mark.parametrize(
    "bad", ["clock", "repeat", "nan", "negative", "dtype", "initial_contact", "state"]
)
def test_invalid_inputs_are_terminal_not_a_silent_zero_or_retry(bad):
    c, d = setup()
    force = np.zeros(6)
    frame = 0
    if bad == "clock":
        frame = 1
    elif bad == "repeat":
        call(c, d, 0)
    elif bad == "nan":
        force[0] = np.nan
    elif bad == "negative":
        force[0] = -1
    elif bad == "dtype":
        force = force.astype(np.float32)
    elif bad == "initial_contact":
        force[0] = 2
    else:
        d.qpos[7] = np.nan
    with pytest.raises(ValueError):
        call(c, d, frame, force)
    assert c.failed
    with pytest.raises(ValueError, match="cannot resume"):
        call(c, d, c.next_frame)


@pytest.mark.parametrize("value", [None, True, 1, "unknown"])
def test_action_name_is_fixed_before_any_model_allocation(value):
    with pytest.raises(ValueError, match="declared"):
        PostContactReleaseControl(None, world_hash="unused", action_name=value)
    assert len(ACTION_NAMES) == len(set(ACTION_NAMES)) == 13
