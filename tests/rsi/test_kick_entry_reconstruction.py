from dataclasses import asdict

import numpy as np
import pytest
from test_kick_measured_history import sample

from rosclaw_soccer.rsi.kick_entry_reconstruction import sample_from_numeric


def test_persisted_actual_history_roundtrip_is_lossless():
    s = sample(5)
    assert asdict(sample_from_numeric(s.numeric_record())) == asdict(s)


@pytest.mark.parametrize("defect", ["short", "fractional", "nan", "wrongclock"])
def test_invalid_history_is_not_silently_repaired(defect):
    x = sample(5).numeric_record()
    if defect == "short":
        x = x[:-1]
    elif defect == "fractional":
        x[0] += 0.1
    elif defect == "nan":
        x[5] = np.nan
    else:
        x[2] = x[0]
    with pytest.raises(ValueError):
        sample_from_numeric(x)
