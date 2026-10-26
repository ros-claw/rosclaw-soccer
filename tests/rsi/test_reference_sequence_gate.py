import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_sequence_gate import assess_reference_sequence


def _case():
    roots = np.tile(np.array([0, 0, 0.8, 1, 0, 0, 0], np.float64), (3, 1))
    joints = np.zeros((3, 29), np.float64)
    ankles = np.zeros((3, 2, 3), np.float64)
    masks = np.tile(np.array([True, False]), (3, 1))
    valid = np.ones(3, dtype=bool)
    trust = np.zeros(3, dtype=bool)
    nonfoot = np.zeros(3, np.float64)
    return roots, joints, ankles, masks, valid, trust, nonfoot


def test_valid_static_reference_gate_is_not_physical_success():
    result = assess_reference_sequence(*_case(), declared_fps=30.0)
    assert result["reference_sequence_gate_passed"] is True
    assert result["physical_execution_qualified"] is False
    assert result["same_support_ankle_pairs"] == 2


def test_abstention_is_not_dropped_or_filled():
    roots, joints, ankles, masks, valid, trust, nonfoot = _case()
    valid[1] = False
    for array in (roots, joints, ankles, nonfoot):
        array[1] = np.nan
    result = assess_reference_sequence(
        roots, joints, ankles, masks, valid, trust, nonfoot, declared_fps=30.0
    )
    assert result["missing_or_rejected_frames"] == 1
    assert result["reference_sequence_gate_passed"] is False
    roots[1] = 0
    with pytest.raises(ValueError, match="must not be padded"):
        assess_reference_sequence(
            roots, joints, ankles, masks, valid, trust, nonfoot, declared_fps=30.0
        )


def test_jump_and_boundary_are_rejected():
    roots, joints, ankles, masks, valid, trust, nonfoot = _case()
    roots[1, 2] += 0.02
    result = assess_reference_sequence(
        roots, joints, ankles, masks, valid, trust, nonfoot, declared_fps=30.0
    )
    assert result["reference_sequence_gate_passed"] is False
    roots[1, 2] -= 0.02
    trust[0] = True
    assert not assess_reference_sequence(
        roots, joints, ankles, masks, valid, trust, nonfoot, declared_fps=30.0
    )["reference_sequence_gate_passed"]
