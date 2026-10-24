import numpy as np
import pytest

from rosclaw_soccer.rsi.refinement_actor_clock import FULL, LEGACY, validate_actor_clock
from rosclaw_soccer.rsi.refinement_onpolicy_learning import (
    full_actor_terminal_returns,
    observed_terminal_returns,
)


def clock(n=389, tail=100, terminal="FULL_MOTION_PLUS100NN_RECOVERY"):
    return dict(
        controller_kind=FULL,
        motion_reference_controls=289,
        observed_recovery_controls=tail,
        total_observed_controls=n,
        deterministic_recovery_excluded_from_actor_rows=False,
        termination=terminal,
    )


@pytest.mark.parametrize("n,tail", [(1, 0), (70, 0), (289, 0), (290, 1), (338, 49), (389, 100)])
def test_actual_failures_in_either_phase_unpadded(n, tail):
    assert validate_actor_clock(clock(n, tail, "BODY_FAILURE"), n)


@pytest.mark.parametrize("n,tail", [(0, 0), (400, 100), (288, 1), (390, 101), (389, 99)])
def test_invented_or_padded_phase_clock_rejected(n, tail):
    with pytest.raises(ValueError):
        validate_actor_clock(clock(n, tail), n)


def test_legacy_cannot_be_relabelled_as_full_or_full_as_legacy():
    assert validate_actor_clock(clock(), 389)
    legacy = clock() | dict(
        controller_kind=LEGACY, deterministic_recovery_excluded_from_actor_rows=True
    )
    assert not validate_actor_clock(legacy, 289)
    with pytest.raises(ValueError):
        validate_actor_clock(legacy, 389)
    with pytest.raises(ValueError):
        validate_actor_clock(clock(), 289)


def test_full_returns_preserve_motion_credit_and_add_direct_recovery_credit_no_double_delay():
    old = observed_terminal_returns(289, 100, -3.0)
    full = full_actor_terminal_returns(389, -3.0)
    np.testing.assert_array_equal(full[:289], old)
    assert full[-1] == -3.0 and len(full[289:]) == 100
    assert full[289] == np.float32(-3 * 0.995**99)
    assert full_actor_terminal_returns(338, -3.0)[-1] == -3.0


@pytest.mark.parametrize("projected", [False, True])
def test_diagnostic_clock_requires_opt_in_and_never_implicitly_becomes_PPO(projected):
    from rosclaw_soccer.rsi.refinement_actor_clock import DIAGNOSTIC, PROJECTED_DIAGNOSTIC

    e = clock() | dict(controller_kind=PROJECTED_DIAGNOSTIC if projected else DIAGNOSTIC)
    with pytest.raises(ValueError):
        validate_actor_clock(e, 389)
    assert validate_actor_clock(e, 389, allow_diagnostic=True)
    with pytest.raises(ValueError):
        validate_actor_clock(e, 389, allow_diagnostic=1)
