from __future__ import annotations

import inspect

from rosclaw_soccer.training.neural_contact_canary import (
    run_neural_contact_canary,
    validate_neural_contact_canary,
)


def test_neural_contact_canary_requires_full_lineage_and_external_output() -> None:
    parameters = inspect.signature(run_neural_contact_canary).parameters
    for name in (
        "teacher_discovery_report_path",
        "actor_training_report_path",
        "actor_path",
        "handoff_actor_path",
        "output_dir",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert callable(validate_neural_contact_canary)
