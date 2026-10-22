from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.velocity_torque_transfer import (
    decode_pd_torque,
    require_torque_encoding,
    transfer_cases,
    world_pulse,
)


def encoding():
    gain, gear = np.zeros((29, 10)), np.zeros((29, 6))
    gain[:, 0] = gear[:, 0] = 1
    model = SimpleNamespace(
        actuator_gainprm=gain,
        actuator_biasprm=np.zeros((29, 10)),
        actuator_gaintype=np.zeros(29),
        actuator_biastype=np.zeros(29),
        actuator_gear=gear,
        jnt_actfrclimited=np.ones(29, dtype=bool),
        jnt_actfrcrange=np.tile([-25.0, 25.0], (29, 1)),
    )
    return model, SimpleNamespace(ai=list(range(29)), joints=list(range(29)))


def test_effort_limits_are_read_only_and_not_overridden():
    model, layout = encoding()
    caps = require_torque_encoding(model, layout)
    caps[0] = 1
    assert model.jnt_actfrcrange[0, 1] == 25


@pytest.mark.parametrize("bad", ["gain", "bias", "gear", "limit"])
def test_wrong_actuator_semantics_fail_closed(bad):
    model, layout = encoding()
    if bad == "gain":
        model.actuator_gainprm[0, 0] = 40
    elif bad == "bias":
        model.actuator_biasprm[0, 1] = -40
    elif bad == "gear":
        model.actuator_gear[0, 0] = 2
    else:
        model.jnt_actfrclimited[0] = False
    with pytest.raises(ValueError):
        require_torque_encoding(model, layout)


def test_pd_math_and_saturation_without_modifying_measured_state():
    q, v = np.zeros(29), np.full(29, 0.5)
    torque = decode_pd_torque(
        np.ones(29), q, v, np.full(29, 40.0), np.full(29, 2.0), np.full(29, 25.0)
    )
    assert np.array_equal(torque, np.full(29, 25.0))
    assert np.array_equal(q, np.zeros(29)) and np.array_equal(v, np.full(29, 0.5))
    with pytest.raises(ValueError):
        decode_pd_torque(np.ones(29, np.float32), q, v, np.ones(29), np.ones(29), np.ones(29))


def test_fixed_complete_transfer_bank_has_idle_and_dynamic_cases():
    cases = transfer_cases()
    assert len(cases) == len(set(cases)) == 8
    assert {i for i, _ in cases} == {"velocity_home", "recorded_paid"}
    assert {s for _, s in cases} == {0.0, 0.3, 0.6, 1.0}
    assert len(cases) * 2 * 300 * 10 == 48000


@pytest.mark.parametrize("yaw", [0.0, 1.2, -3.0])
def test_every_active_command_passes_exact_codec_before_physics(yaw):
    from rosclaw_soccer.physics.g1_velocity_observation import world_command_to_body

    q = np.zeros(43)
    q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    for _, speed in transfer_cases():
        world = world_pulse(speed, yaw)
        assert all(type(v) is float for v in world)
        assert np.allclose(world_command_to_body(q, world), [speed, 0, 0], atol=1e-7)
