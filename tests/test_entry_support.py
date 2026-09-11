import numpy as np
import pytest

from rosclaw_soccer.training.entry_support import EntrySupportGate, calibrate_entry_support


def test_training_entries_supported_and_far_query_rejected():
    examples = np.random.default_rng(601).uniform(-0.02, 0.02, (130, 12))
    gate = calibrate_entry_support(examples)
    assert all(gate.assess(tuple(row)).supported for row in examples.tolist())
    assert not gate.assess((5.0,) * 12).supported
    assert gate.assess((5.0,) * 12).squared_distance > gate.threshold
    assert gate.threshold > 0


def test_gate_copies_inputs_and_zero_variance_is_bounded():
    examples = np.zeros((10, 3))
    gate = calibrate_entry_support(examples)
    examples[:] = 9
    assert gate.centers[0] == (0.0, 0.0, 0.0)
    assert gate.scales == (0.01,) * 3
    assert gate.assess((0.0,) * 3).supported
    assert not gate.assess((1.0,) * 3).supported


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 11.0])
def test_bad_calibration_and_queries_rejected(bad):
    values = np.zeros((3, 2))
    values[0, 0] = bad
    with pytest.raises(ValueError):
        calibrate_entry_support(values)
    gate = calibrate_entry_support(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        gate.assess((bad, 0.0))


@pytest.mark.parametrize(
    "values", [np.zeros((1, 2)), np.zeros(3), np.zeros((3, 2), dtype=bool), np.zeros((512, 512))]
)
def test_invalid_or_unbounded_matrix_rejected(values):
    with pytest.raises(ValueError):
        calibrate_entry_support(values)


@pytest.mark.parametrize(
    "kwargs", [{"quantile": 0.5}, {"expansion": 0}, {"scale_floor": 0}, {"expansion": True}]
)
def test_invalid_calibration_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        calibrate_entry_support(np.zeros((3, 2)), **kwargs)


def test_direct_construction_rejects_mutable_or_invalid_state():
    with pytest.raises(ValueError):
        EntrySupportGate(((0.0,), (0.0,)), (0.0,), 1.0)
    with pytest.raises(ValueError):
        EntrySupportGate([[0.0], [0.0]], (0.01,), 1.0)


def test_bad_query_shape_and_type_rejected():
    gate = calibrate_entry_support(np.zeros((3, 2)))
    for query in [(0.0,), (True, 0.0), [0.0, 0.0]]:
        with pytest.raises(ValueError):
            gate.assess(query)
