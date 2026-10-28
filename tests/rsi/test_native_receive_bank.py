"""The whole declared incoming bank is checked before native data allocation."""

import pytest

from rosclaw_soccer.rsi.native_receive_bank import build_incoming_bank
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse


def test_reject_incomplete_bank_before_model_access():
    with pytest.raises(ValueError, match="entire 256"):
        build_incoming_bank(model=None, params=None, bank={}, plans=[{}] * 8, courses=())


def test_bad_last_course_rejects_all_before_model_access():
    courses = tuple(
        ReceivingCourse("red.finisher", i, 1.25 if i != 255 else 4.0, 0.08) for i in range(256)
    )
    with pytest.raises(ValueError, match="finite bounded physical receiving launch"):
        build_incoming_bank(model=None, params=None, bank={}, plans=[{}] * 8, courses=courses)


def test_duplicate_identity_rejected_before_model_access():
    courses = tuple(ReceivingCourse("red.finisher", 1, 1.25, 0.08) for _ in range(256))
    with pytest.raises(ValueError, match="duplicate player/seed"):
        build_incoming_bank(model=None, params=None, bank={}, plans=[{}] * 8, courses=courses)
