import pytest

from scripts.rsi_probe_velocity_prior_training_body import mobility_cases


def test_existing_four_case_protocol_unchanged():
    assert mobility_cases(False) == ((True, 0.0), (True, 0.3), (False, 0.0), (False, 0.3))


def test_authority_bank_balanced_fixed_inside_qualified_command_range():
    cases = mobility_cases(True)
    assert len(cases) == 8 and len(set(cases)) == 8
    assert cases == tuple((home, speed) for home in (True, False) for speed in (0.3, 0.6, 0.8, 1.0))
    assert len(cases) * 2 * 300 * 10 == 48000
    assert max(speed for _, speed in cases) <= 1.0


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_no_implicit_protocol_selection(value):
    with pytest.raises(ValueError):
        mobility_cases(value)
