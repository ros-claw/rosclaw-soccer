"""Opt-in ONNX pool sizing leaves the frozen navigation default unchanged."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_runup import _sonic_session_kwargs


@pytest.mark.parametrize("value", [0, 9, True, 1.5, "1"])
def test_inference_thread_budget_is_explicit_and_bounded(value: object) -> None:
    with pytest.raises(ValueError, match="inference threads"):
        SonicNavigationConfig(inference_threads=value)


def test_session_options_are_opt_in_and_private() -> None:
    fake_ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace())
    assert _sonic_session_kwargs(fake_ort, None) == {}
    first = _sonic_session_kwargs(fake_ort, 1)["sess_options"]
    second = _sonic_session_kwargs(fake_ort, 1)["sess_options"]
    assert first is not second
    assert first.intra_op_num_threads == first.inter_op_num_threads == 1
