from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw_soccer.training.role_isolated_contact_qualification import (
    _derive_qualification,
    validate_role_isolated_contact_qualification,
)

_QUALIFICATION = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s115-heavy-pitch-qualification-v5/evidence.json"
)


def _probe(*, complete: bool) -> dict[str, object]:
    return {
        "evidence_passed": True,
        "evidence_gates": {"strict_replay": True},
        "plasticity_gates": {
            "candidate_envelope_supported": True,
            "candidate_selected": True,
            "complete_chain_passed": complete,
        },
    }


def test_qualification_rejects_a_control_success_that_fails_sealed_holdout() -> None:
    gates, promoted, status = _derive_qualification(
        control=_probe(complete=True), holdout=_probe(complete=False)
    )

    assert all(value for key, value in gates.items() if key != "holdout_complete_chain_passed")
    assert gates["holdout_complete_chain_passed"] is False
    assert promoted is False
    assert status == "REJECTED_SEALED_HOLDOUT_TASK_FAILURE"


def test_qualification_needs_complete_control_and_holdout_chains() -> None:
    gates, promoted, status = _derive_qualification(
        control=_probe(complete=True), holdout=_probe(complete=True)
    )

    assert all(gates.values())
    assert promoted is True
    assert status == "QUALIFIED_CONTROL_AND_SEALED_HOLDOUT_SIM_ONLY_CANDIDATE"


def test_current_heavy_ball_qualification_rejects_narrow_single_point_success() -> None:
    if not _QUALIFICATION.is_file():
        pytest.skip("current heavy-ball qualification evidence is not present")
    report = validate_role_isolated_contact_qualification(_QUALIFICATION)

    assert report["candidate_promoted"] is False
    assert report["candidate_status"] == "REJECTED_SEALED_HOLDOUT_TASK_FAILURE"
    assert report["gates"]["control_complete_chain_passed"] is True
    assert report["gates"]["holdout_complete_chain_passed"] is False
