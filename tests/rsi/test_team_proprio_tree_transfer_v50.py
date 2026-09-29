"""Fail-closed counterfactual scoring of abstained navigation choices."""

import numpy as np
import pytest

from scripts.rsi_team_proprio_tree_transfer_v50 import outcome_counts


def test_abstention_inherits_parent_failure() -> None:
    labels = np.zeros((2, 6, 3), dtype=np.bool_)
    labels[1, 2] = (True, True, True)
    parent = np.asarray([[False, False, False], [True, False, False]])
    score = outcome_counts(np.asarray([-1, 2]), labels, parent)
    assert score == {
        "scenes": 2,
        "abstained_to_parent": 1,
        "safe": 1,
        "safe_contact": 1,
        "useful": 1,
        "unsafe": 1,
    }


def test_invalid_choice_rejected() -> None:
    with pytest.raises(ValueError, match="counterfactual"):
        outcome_counts(np.asarray([6]), np.zeros((1, 6, 3)), np.zeros((1, 3)))
