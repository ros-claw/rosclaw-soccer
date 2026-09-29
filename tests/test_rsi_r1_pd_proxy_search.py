"""A frozen-extra PD proxy cannot accept unsealed data or unbounded actions."""

from pathlib import Path

import numpy as np
import pytest
from rsi_r1_pd_proxy_search_v140 import load_tape, replay


def test_pd_proxy_rejects_unsealed_tape(tmp_path: Path):
    tape = tmp_path / "course.npz"
    tape.write_bytes(b"not a qualified eight-player tape")
    with pytest.raises(ValueError, match="sealed eight-player motor tape"):
        load_tape(tape, "sha256:" + "0" * 64)


@pytest.mark.parametrize("weights", [np.ones(5), np.full(6, float("nan")), np.full(6, 1.01)])
def test_pd_proxy_rejects_invalid_action_before_any_simulator_access(weights):
    with pytest.raises(ValueError, match="six bounded finite"):
        replay(None, {}, weights)
