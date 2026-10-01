"""Strict transport comparison: physics identity is distinct from provenance identity."""

from typing import Any


def compare_transport(
    new_report: dict[str, Any],
    old_report: dict[str, Any],
    new_outcome: dict[str, Any],
    old_outcome: dict[str, Any],
) -> dict[str, Any]:
    """Call only after independently reconstructing BOTH complete command traces.

    Source/report/parent identities legitimately change on a reader-only source
    revision. No other raw-report field or measured outcome may change. The
    command audit includes source_report_hash, so retain both audit identities
    explicitly instead of falsely claiming that provenance hashes are equal.
    """
    provenance = {"source_hash", "report_hash", "parent_report_hash"}
    if new_report.keys() != old_report.keys() or any(
        new_report[k] != old_report[k] for k in new_report.keys() - provenance
    ):
        raise ValueError("transport changed a raw physical report field")
    if new_outcome.keys() != old_outcome.keys() or any(
        new_outcome[k] != old_outcome[k] for k in new_outcome.keys() - {"command_audit_hash"}
    ):
        raise ValueError("transport changed an independently reconstructed physical outcome")
    hashes = [old_outcome["command_audit_hash"], new_outcome["command_audit_hash"]]
    if any(not isinstance(h, str) or not h.startswith("sha256:") or len(h) != 71 for h in hashes):
        raise ValueError("both independently reconstructed audit identities required")
    return dict(
        measured_physical_fields_equal=True,
        reference_command_audit_hash=hashes[0],
        current_command_audit_hash=hashes[1],
        provenance_hashes_equal=hashes[0] == hashes[1],
        interpretation="Audit provenance binds each distinct source report; not a physics metric",
    )
