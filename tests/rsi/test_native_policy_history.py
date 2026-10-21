from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.velocity_policy import phase_delta
from rosclaw_soccer.rsi.native_policy_history import restore_native_policy_history


def session():
    return SimpleNamespace(
        next_frame=0,
        neural_calls=0,
        attempted_neural_calls=0,
        faulted=False,
        reference_mode="declared79",
    )


def phase_at(frame):
    p = phase_delta()
    for _ in range(frame):
        p = np.float32(np.fmod(np.float32(p + phase_delta()), np.float32(1)))
    return float(p)


def test_restore_measured_history_owns_copy_and_preserves_fresh_counters():
    obj = session()
    a = np.ones(29, dtype=np.float32)
    restore_native_policy_history(obj, frame=125, previous_action=a, phase_before=phase_at(125))
    assert obj.next_frame == 125 and obj.neural_calls == 0
    assert float(obj.phase) == phase_at(125)
    a[:] = 7
    assert np.all(obj.last_action == 1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("faulted", True),
        ("neural_calls", 1),
        ("next_frame", 4),
        ("attempted_neural_calls", 1),
        ("reference_mode", "zero79"),
    ],
)
def test_active_or_faulted_policy_cannot_be_rewound(key, value):
    obj = session()
    setattr(obj, key, value)
    with pytest.raises(ValueError, match="fresh"):
        restore_native_policy_history(
            obj, frame=20, previous_action=np.zeros(29, dtype=np.float32), phase_before=phase_at(20)
        )


def test_wrong_phase_or_unrecorded_first_action_fails_without_mutation():
    obj = session()
    with pytest.raises(ValueError, match="gait phase"):
        restore_native_policy_history(
            obj, frame=20, previous_action=np.zeros(29, dtype=np.float32), phase_before=phase_at(19)
        )
    assert obj.next_frame == 0
    with pytest.raises(ValueError, match="frame-zero"):
        restore_native_policy_history(
            obj, frame=0, previous_action=np.ones(29, dtype=np.float32), phase_before=phase_at(0)
        )
