from dataclasses import replace

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.motor_option_state import MotorOptionState
from rosclaw_soccer.rsi.private_motor_history import clone_history, history_trace, recurrent_history


def state():
    return MotorOptionState((torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)), "sha256:" + "a" * 64)


def test_option_clone_and_trace_preserve_every_private_field_without_aliases():
    a = state()
    b = replace(a, option=2, controls=1, recurrent=tuple(t + 1 for t in a.recurrent))
    c = clone_history(b)
    assert c.option == 2 and c.controls == 1 and c.generation_hash == a.generation_hash
    assert all(x.data_ptr() != y.data_ptr() for x, y in zip(b.recurrent, c.recurrent, strict=True))
    trace = history_trace(a, b)
    assert trace["NN_option_before"] == -1 and trace["NN_option_after"] == 2
    assert trace["NN_option_controls_after"] == 1
    assert trace["NN_option_generation"].shape == (32,)
    assert trace["NN_option_generation"].dtype == np.uint8
    assert all(np.isfinite(t).all() for t in trace.values())
    trace["NN_h_after"].fill(999)
    assert b.recurrent[0].max() == 1


@pytest.mark.parametrize(
    "changes", [dict(option=3), dict(controls=3), dict(generation_hash="sha256:" + "b" * 64)]
)
def test_trace_rejects_reselection_skipped_controls_and_cross_generation(changes):
    a = replace(state(), option=2, controls=1)
    b = replace(a, controls=2)
    with pytest.raises(ValueError, match="preserve"):
        history_trace(a, replace(b, **changes))


@pytest.mark.parametrize("width", [128, 256])
def test_legacy_contract_has_no_option_fields_and_is_deep_cloned(width):
    a = (torch.zeros(2, 1, width), torch.ones(2, 1, width))
    b = clone_history(a, hidden_width=width)
    assert type(b) is tuple and b[0].data_ptr() != a[0].data_ptr()
    assert set(history_trace(a, b, hidden_width=width)) == {
        "NN_h_before",
        "NN_c_before",
        "NN_h_after",
        "NN_c_after",
    }


def test_history_codec_rejects_alias_width_and_implicit_option_erasure():
    a = state()
    with pytest.raises(ValueError):
        recurrent_history((a.recurrent[0], a.recurrent[0]))
    with pytest.raises(ValueError):
        recurrent_history(a, hidden_width=128)
    with pytest.raises(ValueError, match="representation"):
        history_trace(a, a.recurrent)
