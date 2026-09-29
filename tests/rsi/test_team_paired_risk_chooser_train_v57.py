"""Paired risk training never treats a fallback or incomplete rollout as safe."""

import numpy as np
import pytest

from scripts.rsi_team_paired_risk_chooser_train_v57 import _choose, _tally


def test_fallback_uses_real_baseline_outcome() -> None:
    labels = np.zeros((3, 3, 3), dtype=np.bool_)
    labels[0, 1] = (True, True, False)
    labels[0, 2] = (False, False, False)
    labels[1, 1] = (False, False, False)  # incomplete is unsafe
    labels[1, 2] = (True, True, True)
    labels[2, 1] = (False, False, False)
    labels[2, 2] = (False, False, False)
    scores = np.asarray([[0.8, 0.9], [0.01, 0.9], [0.8, 0.0]])
    selected = _choose(scores, 0.04, 0.10)
    assert selected.tolist() == [1, 2, 1]
    assert _tally(selected, labels) == {
        "safe": 2,
        "safe_contact": 2,
        "useful": 1,
        "unsafe": 1,
        "gate_count": 1,
    }


def test_selection_rejects_abstain_and_nonfinite_scores() -> None:
    labels = np.zeros((1, 3, 3), dtype=np.bool_)
    with pytest.raises(ValueError):
        _tally(np.asarray([-1]), labels)
    with pytest.raises(ValueError):
        _choose(np.asarray([[np.nan, 0.0]]), 0.1, 0.0)
