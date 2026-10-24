import numpy as np
import pytest

from rosclaw_soccer.rsi.supported_action_effects import supported_action_effects


def test_activation_does_not_imply_motor_change_or_contact_success():
    a = np.zeros((10, 1), np.float32)
    a[2:5] = 1
    x = np.zeros((10, 29), np.float32)
    r = supported_action_effects(a, x, x.copy(), np.ones(29, np.float32), first_foot_substep=70)
    assert r["support_active_rows"] == 3 and r["actual_changed_rows"] == 0
    assert r["support_segments_in_control_rows"] == [[2, 4]]
    assert r["gap_after_last_support_before_contact_controls"] == 2
    assert r["support_active_at_first_contact"] is False
    assert r["physical_success_inferred_from_activation"] is False


def test_multiple_entries_and_effective_action_change_are_counted_separately():
    a = np.zeros((10, 1), np.float32)
    a[2:4] = 0.5
    a[7] = 1
    x = np.zeros((10, 29), np.float32)
    p = x.copy()
    x[2:4, 0] = 0.01
    r = supported_action_effects(a, x, p, np.ones(29, np.float32), first_foot_substep=72)
    assert r["support_active_rows"] == 3 and r["actual_changed_rows"] == 2
    assert r["support_segments_in_control_rows"] == [[2, 3], [7, 7]]
    assert r["support_active_at_first_contact"] is True


def test_reentry_after_contact_does_not_hide_missing_precontact_support():
    activation = np.zeros((10, 1), np.float32)
    activation[2:4] = 1
    activation[8] = 1
    x = np.zeros((10, 29), np.float32)
    result = supported_action_effects(
        activation, x, x.copy(), np.ones(29, np.float32), first_foot_substep=60
    )
    assert result["last_support_control"] == 8
    assert result["last_support_before_contact_control"] == 3
    assert result["gap_after_last_support_before_contact_controls"] == 2


def test_any_changed_action_outside_support_is_an_integrity_failure():
    x = np.zeros((10, 29), np.float32)
    p = x.copy()
    x[0, 0] = 1e-8
    with pytest.raises(ValueError, match="exact"):
        supported_action_effects(
            np.zeros((10, 1), np.float32), x, p, np.ones(29, np.float32), first_foot_substep=None
        )


@pytest.mark.parametrize("fault", ["NaN", "dtype", "activation", "contact"])
def test_bad_physical_diagnostic_contract_is_not_silently_interpreted(fault):
    activation = np.zeros((10, 1), np.float32)
    x = np.zeros((10, 29), np.float32)
    contact = None
    if fault == "NaN":
        x[0, 0] = np.nan
    elif fault == "dtype":
        x = x.astype(np.float64)
    elif fault == "activation":
        activation[0, 0] = 1.1
    else:
        contact = True
    with pytest.raises(ValueError):
        supported_action_effects(
            activation,
            x,
            np.zeros((10, 29), np.float32),
            np.ones(29, np.float32),
            first_foot_substep=contact,
        )
