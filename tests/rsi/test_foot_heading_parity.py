"""Synthetic strict ABI/parity tests, not physical outcome evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX
from rosclaw_soccer.rsi.foot_heading_parity import SHAPES, require_heading_parity


def baseline():
    return {
        k: np.zeros(
            (300, 1, *s),
            dtype=np.bool_
            if k == "recurrent_sampling_performed"
            else np.float32
            if k.startswith("foundation_neural_")
            else np.float64,
        )
        for k, s in SHAPES.items()
    }


@pytest.mark.parametrize("frames", [30, 300])
@pytest.mark.parametrize("capture", [False, True])
def test_exact_original36_bits_preserved(frames, capture):
    b = baseline()
    t = copy.deepcopy(b)
    if capture:
        t.update({PREFIX + k: np.zeros((300, 1, w)) for k, w in FIELDS.items()})
    before = {k: a.tobytes() for k, a in t.items()}
    r = require_heading_parity(t, b, frames=frames)
    assert r["physical_array_count"] == 36
    assert r["source_receipt_authenticated_here"] is False
    assert {k: a.tobytes() for k, a in t.items()} == before
    t["joint_target_rad"][frames - 1, 0, 0] = -0.0
    with pytest.raises(ValueError, match="bits"):
        require_heading_parity(t, b, frames=frames)


def test_on_can_differ_only_after_declared_pre_action_prefix():
    b, t = baseline(), baseline()
    t["joint_target_rad"][30, 0, 0] = 0.1
    require_heading_parity(t, b, frames=30)
    with pytest.raises(ValueError, match="bits"):
        require_heading_parity(t, b, frames=300)


@pytest.mark.parametrize("frames", [True, 0, 29, 31, 100, 301, 30.0, "300"])
def test_no_weaker_prefix(frames):
    with pytest.raises(ValueError):
        require_heading_parity(baseline(), baseline(), frames=frames)


@pytest.mark.parametrize(
    "mutation",
    [
        "teacher33",
        "reference",
        "oldpulse",
        "unknown",
        "replace",
        "short",
        "dtype",
        "nan_after_prefix",
        "partialheading",
        "extraheading",
        "headingdtype",
        "headingshort",
    ],
)
@pytest.mark.parametrize("side", ["trace", "baseline"])
def test_no_hidden_or_malformed_tensors(mutation, side):
    b, t = baseline(), baseline()
    x = t if side == "trace" else b
    if mutation == "teacher33":
        for k in list(x):
            if k.startswith("ik_planner_pulse_"):
                del x[k]
    elif mutation in ("reference", "oldpulse", "unknown"):
        key = {
            "reference": "reference_student_input",
            "oldpulse": "planner_pulse_x",
            "unknown": "other",
        }[mutation]
        x[key] = np.zeros((300, 1, 3))
    elif mutation == "replace":
        x["unknown"] = x.pop("torque_nm")
    elif mutation == "short":
        x["force_n"] = x["force_n"][:-1]
    elif mutation == "dtype":
        x["force_n"] = x["force_n"].astype(np.float32)
    elif mutation == "nan_after_prefix":
        x["force_n"][299, 0, 0] = np.nan
    else:
        x.update({PREFIX + k: np.zeros((300, 1, w)) for k, w in FIELDS.items()})
        key = PREFIX + "state"
        if mutation == "partialheading":
            del x[key]
        elif mutation == "extraheading":
            x[PREFIX + "unknown"] = np.zeros((300, 1, 1))
        elif mutation == "headingdtype":
            x[key] = x[key].astype(np.float32)
        else:
            x[key] = x[key][:-1]
    with pytest.raises(ValueError):
        require_heading_parity(t, b, frames=30)
