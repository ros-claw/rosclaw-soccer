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


def test_only_observed_native_failures_are_explicitly_classified():
    assert (
        recovery.native_failure_kind("Unable to allocate memory of size 671088640")
        == "NATIVE_CONTACT_BUFFER_ALLOCATION_FAILED"
    )
    assert (
        recovery.native_failure_kind("[Fatal] libX11.so.6!XOpenDisplay+0xbf9")
        == "NATIVE_XDISPLAY_STARTUP_CRASH"
    )


@pytest.mark.parametrize(
    "log", ["", "model diverged", "libX11.so.6!XOpenDisplay", "EOF while parsing a value"]
)
def test_uncategorized_or_incomplete_errors_cannot_authorize_recovery(log):
    with pytest.raises(ValueError, match="explicit observed"):
        recovery.native_failure_kind(log)


@pytest.mark.parametrize("link", [False, True])
def test_exact_local_control_reuse_preserves_original_bytes(tmp_path, link):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    payload = source / "physical_trace.npz"
    payload.write_bytes(b"fixture-not-physical-evidence")
    recovery.copy_complete_control(source, destination, link=link)
    reused = destination / payload.name
    assert reused.read_bytes() == payload.read_bytes() == b"fixture-not-physical-evidence"
    assert (reused.stat().st_ino == payload.stat().st_ino) is link
    with pytest.raises(FileExistsError):
        recovery.copy_complete_control(source, destination, link=link)


def test_control_reuse_rejects_nested_symlink(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    (source / "untrusted").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="non-symlink"):
        recovery.copy_complete_control(source, tmp_path / "new", link=True)
