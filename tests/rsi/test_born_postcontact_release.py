"""Toy measured-Jacobian and synthetic clocks/forces, not physical outcomes."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi.born_postcontact_release import (
    BornPostContactReleaseCommand,
    BornPostContactReleaseControl,
)
from rosclaw_soccer.sim.contracts import hash_bytes
from tests.physics.test_g1_feedback_forecast import world
from tests.rsi.test_impact_torque_screen import sample


@pytest.fixture(autouse=True)
def synthetic_sampler(monkeypatch):
    from rosclaw_soccer.rsi import born_postcontact_release as mod

    class Sampler:
        def __init__(self, model, **kwargs):
            self.model = model

        def sample(self, data):
            assert data.model is self.model
            return sample()

    monkeypatch.setattr(mod, "G1JointTaskKinematicSampler", Sampler)


def lease(action="forward_minus"):
    model, data, _ = world()
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    control = BornPostContactReleaseControl(
        model, world_hash=hash_bytes(binary.tobytes()), action_name=action, start_frame=31
    )
    return data, control


def test_origin_has_no_fabricated_prefix_and_exact_three_command_lease():
    data, control = lease()
    assert control.next_frame == 31 and control.observed_command_frames == 0
    with pytest.raises(ValueError):
        control.review()
    values = []
    for frame in range(31, 300):
        data.time = frame * 0.02  # Explicit synthetic clock; zero mj_step calls.
        forces = np.array([2.0, 0, 0, 0, 0, 0]) if frame == 31 else np.zeros(6)
        command = control.plan(data, frame=frame, previous_contact_forces=forces, pelvis_z_m=0.79)
        assert type(command) is BornPostContactReleaseCommand
        assert command.report["commands_before_origin_observed"] == 0
        values.append(command.offset_nm)
    assert np.array_equal(values[0], values[1]) and np.array_equal(values[1], values[2])
    assert all(not np.any(a) for a in values[3:])
    report = control.review()
    assert report["observed_command_frames"] == 269 and report["new_physical_steps"] == 0
    assert report["origin_frame"] == 31 and report["trigger_frame"] == 31


def test_unobserved_foot_is_terminal_not_retryable():
    data, control = lease()
    data.time = 0.62
    with pytest.raises(ValueError):
        control.plan(data, frame=31, previous_contact_forces=np.zeros(6), pelvis_z_m=0.79)
    with pytest.raises(ValueError):
        control.plan(
            data, frame=31, previous_contact_forces=np.array([2.0, 0, 0, 0, 0, 0]), pelvis_z_m=0.79
        )


@pytest.mark.parametrize("frame", [0, 30, 300, True])
def test_invalid_birth_rejected_before_model_allocation(frame):
    with pytest.raises(ValueError):
        BornPostContactReleaseControl(
            None, world_hash="unused", action_name="baseline", start_frame=frame
        )
