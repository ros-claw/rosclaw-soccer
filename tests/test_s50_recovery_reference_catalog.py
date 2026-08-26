from __future__ import annotations

from rosclaw_soccer.training.recovery_reference_catalog import (
    PINNED_RECOVERY_REFERENCES,
    audit_recovery_reference_catalog,
)


def test_recovery_references_are_commit_pinned_and_license_aware(tmp_path) -> None:
    assert {reference.name for reference in PINNED_RECOVERY_REFERENCES} == {
        "HumanUP",
        "HoST",
        "HiFAR",
        "AMP_mjlab",
    }
    amp = next(item for item in PINNED_RECOVERY_REFERENCES if item.name == "AMP_mjlab")
    assert amp.license_id == "UNDECLARED"
    assert not amp.code_reuse_allowed
    assert all(len(reference.expected_commit) == 40 for reference in PINNED_RECOVERY_REFERENCES)

    # Missing clones fail closed and are never represented as executed baselines.
    report = audit_recovery_reference_catalog(tmp_path)
    assert not report["all_repositories_qualified"]
    assert report["execution_boundary"] == "SOURCE_AND_LICENSE_AUDIT_ONLY"
    assert report["activation_ceiling"] == "SIM_ONLY"
    assert not report["hardware_authorized"]
    assert all(not row["qualified"] for row in report["references"])
