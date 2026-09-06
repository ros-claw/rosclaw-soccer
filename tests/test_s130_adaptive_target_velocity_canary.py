from __future__ import annotations

import inspect

from rosclaw_soccer.training.adaptive_target_velocity_canary import (
    run_adaptive_target_velocity_canary,
    validate_adaptive_target_velocity_canary,
)


def test_adaptive_canary_requires_explicit_lineage_and_external_output() -> None:
    parameters = inspect.signature(run_adaptive_target_velocity_canary).parameters
    for name in (
        "teacher_discovery_report_path",
        "actor_training_report_path",
        "actor_path",
        "handoff_actor_path",
        "output_dir",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert callable(validate_adaptive_target_velocity_canary)
