import gzip
import json

import pytest
from rosclaw.growth.shared_proof_payload import canonical_hash, detach_payload

from rosclaw_soccer.rsi.physical_report_io import CPU_MEAN_MODEL_LOCATION, load_physical_report
from rosclaw_soccer.rsi.physical_report_metadata import PhysicalReportMetadataReader


def write_gzip(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)


def fixture(tmp_path):
    folder = tmp_path / "bank" / "case"
    folder.mkdir(parents=True)
    document = dict(
        executed_motor_policy=dict(
            step_motor_proof=dict(model=dict(mean_model=dict(weights=[[1.0, 2.0]], name="数据")))
        ),
        metric=0.7,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    document["report_hash"] = canonical_hash(document)
    envelope, payload = detach_payload(document, CPU_MEAN_MODEL_LOCATION)
    store = folder.parent / ".shared-models"
    store.mkdir()
    payload_path = store / f"{envelope['payload_hash'][7:]}.json.gz"
    write_gzip(payload_path, payload)
    write_gzip(folder / "report.json.gz", envelope)
    return folder, payload_path, document


def test_complete_original_seals_metadata_only_and_owned_results(tmp_path):
    folder, _, document = fixture(tmp_path)
    reader = PhysicalReportMetadataReader()
    result = reader.read(folder / "report.json")
    assert load_physical_report(folder / "report.json") == document
    assert result["root_seal_verified"] is True
    assert result["complete_numerical_document_returned"] is False
    assert result["metadata_not_physical_replay"] is True
    assert result["metadata_with_payload_marker"]["report_hash"] == document["report_hash"]
    result["metadata_with_payload_marker"]["metric"] = 999
    assert reader.read(folder / "report.json")["metadata_with_payload_marker"]["metric"] == 0.7
    assert str(folder / "report.json.gz") in reader.input_file_hashes()


@pytest.mark.parametrize("method", ["read", "restore"])
def test_cached_payload_changes_and_symlinks_rejected(tmp_path, method):
    folder, payload_path, _ = fixture(tmp_path)
    reader = PhysicalReportMetadataReader()
    operation = getattr(reader, method)
    operation(folder / "report.json")
    write_gzip(payload_path, dict(weights=[[999]], name="数据"))
    with pytest.raises(ValueError, match="bytes changed"):
        operation(folder / "report.json")
    with pytest.raises(ValueError, match="input or source"):
        reader.input_file_hashes()
    other = folder.parent / "another.gz"
    payload_path.rename(other)
    payload_path.symlink_to(other)
    with pytest.raises(ValueError, match="non-symlink"):
        operation(folder / "report.json")


@pytest.mark.parametrize("method", ["read", "restore"])
def test_ambiguous_and_legacy_full_reports_are_not_silently_accepted(tmp_path, method):
    folder, _, document = fixture(tmp_path)
    (folder / "report.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="exactly one"):
        getattr(PhysicalReportMetadataReader(), method)(folder / "report.json")
    (folder / "report.json.gz").unlink()
    with pytest.raises(ValueError, match="exact shared"):
        getattr(PhysicalReportMetadataReader(), method)(folder / "report.json")


def test_complete_restore_is_exact_sealed_and_owned_without_metadata_additions(tmp_path):
    folder, payload_path, document = fixture(tmp_path)
    reader = PhysicalReportMetadataReader()
    first = reader.restore(folder / "report.json")
    assert first == load_physical_report(folder / "report.json") == document
    assert set(first) == set(document)
    assert "metadata_with_payload_marker" not in first
    assert (
        canonical_hash({k: v for k, v in first.items() if k != "report_hash"})
        == first["report_hash"]
    )
    first["executed_motor_policy"]["step_motor_proof"]["model"]["mean_model"]["weights"][0][0] = 999
    assert reader.restore(folder / "report.json") == document
    assert len(reader._cache) == 1
    assert str(payload_path) in reader.input_file_hashes()
    # Compact reads keep their old metadata-only meaning after full restoration.
    assert reader.read(folder / "report.json")["complete_numerical_document_returned"] is False


@pytest.mark.parametrize("method", ["read", "restore"])
def test_payload_change_during_cached_verification_is_rejected(tmp_path, method, monkeypatch):
    folder, payload_path, _ = fixture(tmp_path)
    reader = PhysicalReportMetadataReader()
    operation = getattr(reader, method)
    operation(folder / "report.json")
    compiler = next(iter(reader._cache.values()))[0]
    core_method = "verify" if method == "read" else "restore"
    original = getattr(compiler, core_method)

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        write_gzip(payload_path, {"weights": [[999]], "name": "数据"})
        return result

    monkeypatch.setattr(compiler, core_method, changing)
    with pytest.raises(ValueError, match="shared proof file changed during"):
        operation(folder / "report.json")


def test_complete_restore_rejects_envelope_change_during_restoration(tmp_path, monkeypatch):
    folder, _, _ = fixture(tmp_path)
    reader = PhysicalReportMetadataReader()
    reader.restore(folder / "report.json")
    compiler = next(iter(reader._cache.values()))[0]
    original = compiler.restore

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        write_gzip(folder / "report.json.gz", {"changed": True})
        return result

    monkeypatch.setattr(compiler, "restore", changing)
    with pytest.raises(ValueError, match="envelope changed during"):
        reader.restore(folder / "report.json")


def test_complete_restore_rejects_same_json_with_changed_compressed_bytes(tmp_path):
    folder, payload_path, _ = fixture(tmp_path)
    reader = PhysicalReportMetadataReader()
    reader.restore(folder / "report.json")
    original = gzip.decompress(payload_path.read_bytes())
    payload_path.write_bytes(gzip.compress(original, mtime=1))
    with pytest.raises(ValueError, match="cached shared proof bytes changed"):
        reader.restore(folder / "report.json")


@pytest.mark.parametrize("method", ["read", "restore"])
def test_reader_fails_if_observed_source_pin_changes(tmp_path, method):
    folder, _, _ = fixture(tmp_path)
    reader = PhysicalReportMetadataReader()
    source = tmp_path / "source-fixture.py"
    source.write_text("original")
    from rosclaw_soccer.sim.contracts import hash_bytes

    reader._pins[str(source)] = hash_bytes(source.read_bytes())
    source.write_text("changed")
    with pytest.raises(ValueError, match="dependency source changed"):
        getattr(reader, method)(folder / "report.json")


def test_complete_restore_requires_compatible_core_before_payload_allocation(tmp_path, monkeypatch):
    from rosclaw_soccer.rsi import physical_report_metadata as module

    monkeypatch.delattr(module.compiled_module.CompiledSharedMetadata, "restore")
    reader = PhysicalReportMetadataReader()
    with pytest.raises(ValueError, match="Core complete"):
        reader.restore(tmp_path / "absent" / "report.json")
    assert not reader._cache


def test_complete_restore_rejects_missing_root_seal_not_only_envelope(tmp_path):
    folder, payload_path, document = fixture(tmp_path)
    document["report_hash"] = "sha256:" + "0" * 64
    envelope, payload = detach_payload(document, CPU_MEAN_MODEL_LOCATION)
    write_gzip(folder / "report.json.gz", envelope)
    write_gzip(payload_path, payload)
    with pytest.raises(ValueError, match="root seal"):
        PhysicalReportMetadataReader().restore(folder / "report.json")


@pytest.mark.parametrize("method", ["read", "restore"])
def test_same_bytes_symlink_swap_during_verification_rejected(tmp_path, method, monkeypatch):
    folder, payload_path, _ = fixture(tmp_path)
    reader = PhysicalReportMetadataReader()
    operation = getattr(reader, method)
    operation(folder / "report.json")
    compiler = next(iter(reader._cache.values()))[0]
    core_method = "verify" if method == "read" else "restore"
    original = getattr(compiler, core_method)

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        target = tmp_path / "elsewhere.gz"
        payload_path.rename(target)
        payload_path.symlink_to(target)
        return result

    monkeypatch.setattr(compiler, core_method, changing)
    with pytest.raises(ValueError, match="non-symlink shared proof changed during"):
        operation(folder / "report.json")
