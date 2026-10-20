import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_measured_context import measured_joint_context


def inputs():
    context = np.zeros((4, 36), dtype=np.float64)
    context[:, 3] = 1
    context[:, 7:] = np.arange(4)[:, None] * 0.01
    q = np.zeros(43, dtype=np.float64)
    q[3] = 1
    reference = q[:36].copy()
    q[7:] = np.linspace(-0.3, 0.3, 36)
    limits = np.tile([-1.0, 1.0], (29, 1))
    return context, q, reference, limits


def test_only_joint_context_changes_preserving_direction_and_ownership():
    values = inputs()
    saved = [a.copy() for a in values]
    context, q, reference, _ = values
    result, delta = measured_joint_context(*values)
    raw_delta = q[7:36] - reference[7:36]
    np.testing.assert_allclose(delta, raw_delta * (0.12 / max(abs(raw_delta))), rtol=0, atol=1e-16)
    assert max(abs(delta)) <= 0.120000000000001
    np.testing.assert_array_equal(result[:, :7], context[:, :7])
    np.testing.assert_allclose(
        np.diff(result[:, 7:], axis=0), np.diff(context[:, 7:], axis=0), atol=1e-16
    )
    assert not result.flags.writeable and not delta.flags.writeable
    for before, after in zip(saved, values, strict=True):
        np.testing.assert_array_equal(before, after)
        assert not np.shares_memory(after, result) and not np.shares_memory(after, delta)


def test_zero_measurement_gap_is_bit_exact():
    context, q, reference, limits = inputs()
    q[7:36] = reference[7:36]
    result, delta = measured_joint_context(context, q, reference, limits)
    np.testing.assert_array_equal(result, context)
    np.testing.assert_array_equal(delta, np.zeros(29))


@pytest.mark.parametrize(
    "kind", ["shape", "dtype", "nan", "quaternion", "limits", "original", "proposed"]
)
def test_bad_or_illegal_context_rejected_without_repair(kind):
    context, q, reference, limits = inputs()
    if kind == "shape":
        q = q[:36]
    elif kind == "dtype":
        context = context.astype(np.float32)
    elif kind == "nan":
        q[1] = np.nan
    elif kind == "quaternion":
        context[0, 3] = 0
    elif kind == "limits":
        limits[0] = [1, -1]
    elif kind == "original":
        context[0, 7] = 1.01
    else:
        context[:, 35] = 0.99
        q[35] = 0.3
    saved = context.copy()
    with pytest.raises(ValueError):
        measured_joint_context(context, q, reference, limits)
    np.testing.assert_array_equal(context, saved)
