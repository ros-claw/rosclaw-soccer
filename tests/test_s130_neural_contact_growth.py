from __future__ import annotations

import inspect

from rosclaw_soccer.training.neural_contact_growth import train_neural_contact_actor


def test_neural_contact_growth_requires_bound_teacher_and_external_output() -> None:
    parameters = inspect.signature(train_neural_contact_actor).parameters
    assert parameters["source_teacher_report_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["output_dir"].kind is inspect.Parameter.KEYWORD_ONLY
