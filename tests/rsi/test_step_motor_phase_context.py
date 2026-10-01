import numpy as np
import pytest

from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory, phase_sequence


def test_contact_phase_reads_only_completed_physics_and_retains_event_memory():
    forces = np.zeros((300, 6))
    forces[69, 0] = 10
    phases = phase_sequence(forces)
    assert np.all(phases[:70] == 0)
    assert np.all(phases[70:90] == 1)
    assert np.all(phases[90:] == 2)
    forces[100:] = 1000
    assert np.array_equal(phases[:101], phase_sequence(forces)[:101])


def test_skipping_frames_or_nonfinite_forces_fails_closed():
    memory = ContactPhaseMemory()
    assert memory.advance(0, np.zeros(6)) == 0
    with pytest.raises(ValueError):
        memory.advance(2, np.zeros(6))
    with pytest.raises(ValueError):
        memory.advance(1, np.full(6, np.nan))
    assert memory.advance(1, np.zeros(6)) == 0
