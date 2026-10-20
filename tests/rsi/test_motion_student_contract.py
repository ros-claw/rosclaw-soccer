import numpy as np
import pytest

from scripts.rsi_motion_student_rollout import ForbiddenTeacherNetwork, validate_motion_recording


def test_teacher_networks_are_fail_closed_not_fallbacks():
    blocked = ForbiddenTeacherNetwork()
    with pytest.raises(RuntimeError, match="forbidden"):
        blocked.run(None, {})
    with pytest.raises(RuntimeError, match="forbidden"):
        blocked.get_inputs()


@pytest.mark.parametrize(
    "raw", [{}, {"student_motion_token": np.zeros((300, 29), dtype=np.float32)}]
)
def test_29_actions_or_incomplete_history_cannot_claim64_motion_proof(raw):
    with pytest.raises(ValueError, match="distinct intention/muscle/native"):
        validate_motion_recording(raw, nbody=46)
