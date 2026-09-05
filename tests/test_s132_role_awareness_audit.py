from __future__ import annotations

import inspect

from rosclaw_soccer.training.role_awareness_audit import (
    RoleCurriculumAssignment,
    run_role_awareness_audit,
    validate_role_awareness_audit,
)


def test_role_awareness_audit_requires_all_physical_sources() -> None:
    parameters = inspect.signature(run_role_awareness_audit).parameters
    for name in (
        "lead_pass_evidence_path",
        "neural_canary_report_path",
        "neural_holdout_report_path",
        "output_dir",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert callable(validate_role_awareness_audit)
    assert RoleCurriculumAssignment.__dataclass_fields__["activation_ceiling"].default == "SIM_ONLY"
