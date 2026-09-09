import pytest

from rosclaw_soccer.providers.g1.vector_contact import PhysicsSampleClock


def test_every_physical_step_counts_once():
    clock = PhysicsSampleClock()
    clock.begin(1, 0)
    for step in range(1, 11):
        clock.observe(1, step)
    assert clock.valid and clock.samples == 10
    clock.begin(2, 10)
    clock.observe(2, 11)
    assert clock.samples == 1


@pytest.mark.parametrize("episode,step", [(1, 10), (1, 0), (2, 1), (True, 1), (1, True)])
def test_policy_rate_sampling_and_implicit_resets_invalidate_episode(episode, step):
    clock = PhysicsSampleClock()
    clock.begin(1, 0)
    with pytest.raises(RuntimeError):
        clock.observe(episode, step)
    assert not clock.valid
    with pytest.raises(RuntimeError):
        clock.observe(1, 1)
    with pytest.raises(ValueError):
        clock.begin(1, 10)
    clock.begin(2, 10)
    clock.observe(2, 11)
    assert clock.valid


def test_contact_evidence_cannot_begin_before_a_declared_episode():
    clock = PhysicsSampleClock()
    with pytest.raises(RuntimeError):
        clock.observe(0, 1)
    with pytest.raises(ValueError):
        clock.begin(0, 0)
