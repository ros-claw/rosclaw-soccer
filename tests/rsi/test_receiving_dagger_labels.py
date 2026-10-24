import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_dagger_labels import receiving_dagger_label


@pytest.mark.parametrize(
    "functional,strict,plane,use_actual",
    [
        (True, False, False, True),
        (True, False, True, False),
        (True, True, True, True),
        (False, False, False, False),
        (False, False, True, False),
    ],
)
def test_do_not_erase_student_success_with_a_weaker_geometric_teacher(
    functional, strict, plane, use_actual
):
    actual, advice = np.array([0.2, 0, 0], np.float32), np.array([-0.3, 0.1, 0], np.float32)
    label, kind = receiving_dagger_label(
        actual,
        advice,
        observed_safe_functional=functional,
        observed_strict=strict,
        plane_before_contact=plane,
    )
    np.testing.assert_array_equal(label, actual if use_actual else advice)
    assert (kind == "ACTUAL_SUCCESS_SELF_RETENTION") == use_actual
    actual.fill(1)
    advice.fill(1)
    assert label[0] != 1


def test_impossible_strict_success_without_functional_receive_rejected():
    with pytest.raises(ValueError):
        receiving_dagger_label(
            np.zeros(3, np.float32),
            np.zeros(3, np.float32),
            observed_safe_functional=False,
            observed_strict=True,
            plane_before_contact=True,
        )
