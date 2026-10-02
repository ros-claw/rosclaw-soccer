from pathlib import Path

import pytest

from scripts import rsi_recover_consolidated_bank_delta as recovery


def commitment():
    return dict(
        schema="soccer.rsi.consolidated_bank_delta_commitment.v1",
        indices=recovery.DECLARED_INDICES,
        physical_report_representation="lossless_gzip_json",
        source_commit="runner",
        runner_hash="hash",
        core_commit="core",
        input_hashes={},
        promotion_authorized=False,
        hardware_authorized=False,
    )


def bind(monkeypatch):
    runner, core = Path("fixture/scripts/runner.py"), Path("fixture-core")
    monkeypatch.setattr(recovery, "_head", lambda p: "core" if p == core else "runner")
    monkeypatch.setattr(Path, "read_bytes", lambda p: b"source")
    monkeypatch.setattr(recovery, "hash_bytes", lambda _: "hash")
    return runner, core


def test_exact_original_source_and_no_promotion_required(monkeypatch):
    runner, core = bind(monkeypatch)
    recovery.check_original_binding(commitment(), runner, core)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_commit", "new-runner"),
        ("runner_hash", "new-source"),
        ("core_commit", "new-core"),
        ("promotion_authorized", True),
        ("hardware_authorized", True),
        ("indices", [6, 21]),
        ("physical_report_representation", "new-format"),
    ],
)
def test_recovery_cannot_change_source_scope_format_or_authority(monkeypatch, key, value):
    runner, core = bind(monkeypatch)
    data = commitment()
    data[key] = value
    with pytest.raises(ValueError, match="original exact"):
        recovery.check_original_binding(data, runner, core)
