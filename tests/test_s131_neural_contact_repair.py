from __future__ import annotations

import inspect

from rosclaw_soccer.training.neural_contact_repair_discovery import (
    run_neural_contact_repair_discovery,
)


def test_neural_contact_repair_requires_rejected_holdout_lineage() -> None:
    parameters = inspect.signature(run_neural_contact_repair_discovery).parameters
    for name in (
        "rejected_holdout_report_path",
        "teacher_discovery_report_path",
        "handoff_actor_path",
        "output_dir",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
