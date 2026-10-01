import pytest

from scripts.rsi_collect_online_step_validation import baseline_reference_binding


def test_explicit_pilot_and_later_parent_paths_are_not_confused():
    summary = dict(
        physical_executions=12,
        independent_contexts=4,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    assert baseline_reference_binding(
        dict(summary, schema="soccer.rsi.step_motor_physical_pilot.v1")
    ) == ("step-neural", "neural")
    assert baseline_reference_binding(
        dict(summary, schema="soccer.rsi.online_step_motor_physical_validation.v1")
    ) == ("online", "online")
    for extra in (
        dict(schema="unknown"),
        dict(physical_executions=8),
        dict(hardware_authorized=True),
    ):
        with pytest.raises(ValueError):
            baseline_reference_binding(
                dict(summary, schema="soccer.rsi.step_motor_physical_pilot.v1") | extra
            )
