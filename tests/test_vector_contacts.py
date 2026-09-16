from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.vector_contacts import (
    CONTACT_CHANNELS,
    G1ContactVectorMotorBatch,
    robot_contact_geometry_labels,
)


def model():
    names = {"pelvis": 1, "left_ankle_roll_link": 3, "right_ankle_roll_link": 5}
    return SimpleNamespace(
        body_parentid=np.array([0, 0, 1, 2, 3, 1, 5, 0, 0]),
        geom_bodyid=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        body=lambda name: SimpleNamespace(id=names[name]),
        geom=lambda name: SimpleNamespace(id=7 if name == "ball_geom" else 8),
    )


def test_foot_descendants_keep_shin_environment_and_ball_separate():
    value = model()
    before = value.geom_bodyid.copy(), value.body_parentid.copy()
    labels = robot_contact_geometry_labels(value)
    assert CONTACT_CHANNELS == ("left_foot", "right_foot", "other_robot")
    np.testing.assert_array_equal(labels, [0, 3, 3, 1, 1, 2, 2, 0, 0])
    assert labels.dtype == np.int32 and not labels.flags.writeable
    with pytest.raises(ValueError):
        labels[2] = 1
    np.testing.assert_array_equal(value.geom_bodyid, before[0])
    np.testing.assert_array_equal(value.body_parentid, before[1])


@pytest.mark.parametrize(
    "failure", ["cycle", "outside", "negative", "float", "nested", "ball_in_robot", "missing_foot"]
)
def test_malformed_or_wrong_body_tree_rejected(failure):
    value = model()
    if failure == "cycle":
        value.body_parentid[1] = 2
    elif failure == "outside":
        value.geom_bodyid[0] = 99
    elif failure == "negative":
        value.body_parentid[2] = -1
    elif failure == "float":
        value.body_parentid = value.body_parentid.astype(float)
    elif failure == "nested":
        value.body_parentid[5] = 3
    elif failure == "ball_in_robot":
        value.body_parentid[7] = 1
    else:
        value.geom_bodyid[5:7] = 0
    with pytest.raises(ValueError):
        robot_contact_geometry_labels(value)


@pytest.mark.parametrize(
    "fault", [None, "nonfinite_force", "bad_world", "bad_geometry", "overflow", "negative_count"]
)
def test_sensor_kernel_counts_only_solved_positive_robot_force_and_rejects_bad_data(fault):
    wp = pytest.importorskip("warp")
    from rosclaw_soccer.providers.g1._vector_contact_kernel import accumulate_ball_contacts

    wp.init()
    geoms = np.array([[0, 1], [0, 2], [0, 3], [0, 3], [0, 4], [0, 2]], dtype=np.int32)
    world = np.array([0, 1, 0, 0, 0, 0], dtype=np.int32)
    forces = np.zeros((6, 6), dtype=np.float32)
    forces[:, 0] = [3, 7, -2, 0, 100, 1000]
    if fault == "nonfinite_force":
        forces[0, 0] = np.nan
    elif fault == "bad_world":
        world[0] = 2
    elif fault == "bad_geometry":
        geoms[0, 1] = 99
    inputs = [
        wp.array(
            np.array([7 if fault == "overflow" else -1 if fault == "negative_count" else 5]),
            dtype=int,
            device="cpu",
        ),
        wp.array(world, dtype=int, device="cpu"),
        wp.array(geoms, dtype=wp.vec2i, device="cpu"),
        wp.array(forces, dtype=wp.spatial_vector, device="cpu"),
        wp.array(np.array([0, 1, 2, 3, 0], dtype=np.int32), dtype=int, device="cpu"),
        0,
        2,
        5,
    ]
    peak = wp.zeros((2, 3), dtype=float, device="cpu")
    samples = wp.zeros((2, 3), dtype=int, device="cpu")
    invalid = wp.zeros(1, dtype=int, device="cpu")
    before = [value.numpy().copy() for value in inputs[:5]]
    wp.launch(
        accumulate_ball_contacts, dim=6, inputs=[*inputs, peak, samples, invalid], device="cpu"
    )
    assert invalid.numpy()[0] == int(fault is not None)
    expected = np.array([[3 if fault is None else 0, 0, 0], [0, 7, 0]])
    if fault in ("overflow", "negative_count"):
        expected[:] = 0
    np.testing.assert_array_equal(peak.numpy(), expected)
    np.testing.assert_array_equal(samples.numpy(), expected > 0)
    for old, value in zip(before, inputs[:5], strict=True):
        np.testing.assert_array_equal(old, value.numpy())


def test_sensor_requires_explicit_graph_preparation():
    motor = object.__new__(G1ContactVectorMotorBatch)
    motor._physics_graph = None
    with pytest.raises(RuntimeError, match="prepared physics graph"):
        motor._step(None, None)


@pytest.mark.parametrize("fault", ["invalid", "nonfinite"])
def test_sensor_fault_disarms_and_zeros_control_without_reset(monkeypatch, fault):
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from rosclaw_soccer.providers.g1.vector_motor import G1VectorMotorBatch

    wp.init()
    motor = object.__new__(G1ContactVectorMotorBatch)
    motor._physics_graph = object()
    motor._torch = torch
    motor._ready = True
    motor._ctrl = torch.ones((1, 29))
    motor._contact_peak = wp.zeros((1, 3), dtype=float, device="cpu")
    motor._contact_samples = wp.zeros((1, 3), dtype=int, device="cpu")
    motor._contact_invalid = wp.zeros(1, dtype=int, device="cpu")
    calls = []

    def step(self, torque, pd):
        calls.append((torque, pd))
        if fault == "invalid":
            wp.to_torch(self._contact_invalid).fill_(1)
        else:
            wp.to_torch(self._contact_peak).fill_(float("nan"))
        return {}

    monkeypatch.setattr(G1VectorMotorBatch, "_step", step)
    with pytest.raises(FloatingPointError, match="failed state retained"):
        motor._step(None, None)
    assert calls == [(None, None)]
    assert not motor._ready and torch.count_nonzero(motor._ctrl) == 0
