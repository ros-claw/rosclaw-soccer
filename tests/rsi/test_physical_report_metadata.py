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


def test_cached_payload_changes_and_symlinks_rejected(tmp_path):
    folder, payload_path, _ = fixture(tmp_path)
    reader = PhysicalReportMetadataReader()
    reader.read(folder / "report.json")
    write_gzip(payload_path, dict(weights=[[999]], name="数据"))
    with pytest.raises(ValueError, match="bytes changed"):
        reader.read(folder / "report.json")
    with pytest.raises(ValueError, match="input or source"):
        reader.input_file_hashes()
    other = folder.parent / "another.gz"
    payload_path.rename(other)
    payload_path.symlink_to(other)
    with pytest.raises(ValueError, match="non-symlink"):
        reader.read(folder / "report.json")


def test_ambiguous_and_legacy_full_reports_are_not_silently_accepted(tmp_path):
    folder, _, document = fixture(tmp_path)
    (folder / "report.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="exactly one"):
        PhysicalReportMetadataReader().read(folder / "report.json")
    (folder / "report.json.gz").unlink()
    with pytest.raises(ValueError, match="exact shared"):
        PhysicalReportMetadataReader().read(folder / "report.json")
