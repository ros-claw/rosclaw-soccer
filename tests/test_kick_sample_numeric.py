import numpy as np
import pytest

from rosclaw_soccer.providers.g1.kick_measured_history import KickMeasuredSample


def sample():
    values = tuple(map(float, np.linspace(-0.2, 0.2, 29)))
    return KickMeasuredSample(
        5,
        0.1,
        4,
        values,
        values,
        (0.1, 0.2, 0.3),
        (1.0, 2.0, 0.8),
        (1.0, 0.0, 0.0, 0.0),
        (2.0, 0.1, 0.11),
        values,
    )


def test_float64_trace_roundtrip_is_lossless():
    original = sample()
    restored = KickMeasuredSample.from_numeric_record(original.numeric_record())
    assert restored == original
    np.testing.assert_array_equal(restored.numeric_record(), original.numeric_record())


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "fraction", "large", "quaternion"])
def test_never_round_bad_clocks_or_accept_nonfinite_history(bad):
    row = sample().numeric_record()
    if bad == "shape":
        row = row[:-1]
    elif bad == "dtype":
        row = row.astype(np.float32)
    elif bad == "nan":
        row[1] = np.nan
    elif bad == "fraction":
        row[0] = 5.5
    elif bad == "large":
        row[0] = 2**54
    elif bad == "quaternion":
        row[67:71] = 0
    with pytest.raises(ValueError):
        KickMeasuredSample.from_numeric_record(row)
