import pytest

from rosclaw_soccer.rsi.cftrack_shared_relay_trial import run_shared_cftrack_trial


@pytest.mark.parametrize("context,side,arm", [(2, 0.0, True), (2, 0.35, 1), (True, 0.35, False)])
def test_unregistered_course_fails_before_allocating_world(context, side, arm):
    class ForbiddenSetup:
        def prepare(self, *args):
            raise AssertionError("invalid declaration allocated physics")

    with pytest.raises(ValueError):
        run_shared_cftrack_trial(
            ForbiddenSetup(),
            object(),
            context=context,
            lateral_m=side,
            use_cftrack=arm,
            policy_factory=lambda: None,
            expected_policy_type=object,
        )
