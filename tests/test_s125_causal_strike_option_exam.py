from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.training.causal_strike_option_exam import (
    default_causal_strike_option_development_contexts,
    default_causal_strike_option_holdouts,
    validate_causal_strike_option_retention,
)


def test_s125_holdouts_are_fresh_and_unique() -> None:
    development = default_causal_strike_option_development_contexts()
    holdouts = default_causal_strike_option_holdouts()

    assert len(development) == len(holdouts) == 6
    assert len({context.context_hash for context in holdouts}) == 6
    assert {context.context_hash for context in development}.isdisjoint(
        context.context_hash for context in holdouts
    )
    assert all(context.case_id.startswith("s125.holdout.v1.") for context in holdouts)


def test_rejected_exam_cannot_be_used_as_retention_authority(tmp_path: Path) -> None:
    report = tmp_path / "exam-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.growth.causal_strike_option_exam.v1",
                "status": "REJECTED_CAUSAL_STRIKE_OPTION",
                "sealed": True,
                "promotion_eligible": False,
                "gates": {"option_success_rate": False},
                "report_hash": "sha256:not-an-authority",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="retention authority is invalid"):
        validate_causal_strike_option_retention(report)
