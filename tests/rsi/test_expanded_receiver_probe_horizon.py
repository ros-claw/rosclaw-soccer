import pytest

from scripts.rsi_qualify_relay_snapshot_passer import expanded_probe_end_control, probe_amplitude


@pytest.mark.parametrize(
    "handoff,expected", [(None, 178), ({"control": 170}, 170), ({"control": 210}, 178)]
)
def test_expanded_probe_censors_at_historical_handoff(handoff, expected):
    assert expanded_probe_end_control(handoff) == expected


@pytest.mark.parametrize("handoff", [{}, {"control": True}, {"control": 140}, {"control": 301}])
def test_expanded_probe_rejects_invalid_boundaries(handoff):
    with pytest.raises(ValueError):
        expanded_probe_end_control(handoff)


def test_probe_is_single_bounded_pulse_even_in_long_rollout():
    assert probe_amplitude(140) == 0.0
    assert probe_amplitude(159) == pytest.approx(1.0)
    assert probe_amplitude(177) > 0
    assert probe_amplitude(178) == 0.0
    assert probe_amplitude(216) == 0.0
    with pytest.raises(ValueError):
        probe_amplitude(139)
