from __future__ import annotations

import inspect

from rosclaw_soccer.training.adaptive_target_velocity_growth import (
    train_adaptive_target_velocity_actor,
)


def test_adaptive_growth_api_requires_bound_discovery_and_external_output() -> None:
    signature = inspect.signature(train_adaptive_target_velocity_actor)
    assert tuple(signature.parameters) == ("discovery_report_path", "output_dir")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
