from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw_soccer.training.role_isolated_contact_qualification import (
    RoleIsolatedContactQualificationConfig,
    _derive_qualification,
    validate_role_isolated_contact_qualification,
)

_QUALIFICATION = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s116-proprioceptive-recovery-qualification-v2/evidence.json"
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


def test_qualification_rejects_invalid_probe_config_hash() -> None:
    with pytest.raises(ValueError, match="config hash"):
        RoleIsolatedContactQualificationConfig(control_probe_config_hash="sha256:bad")


def test_current_proprioceptive_recovery_qualification_passes_two_bound_contexts() -> None:
    if not _QUALIFICATION.is_file():
        pytest.skip("current proprioceptive-recovery qualification evidence is not present")
    report = validate_role_isolated_contact_qualification(_QUALIFICATION)

    assert report["candidate_promoted"] is True
    assert (
        report["candidate_status"]
        == "QUALIFIED_CONTROL_AND_SEALED_HOLDOUT_SIM_ONLY_CANDIDATE"
    )
    assert report["gates"]["control_complete_chain_passed"] is True
    assert report["gates"]["holdout_complete_chain_passed"] is True
