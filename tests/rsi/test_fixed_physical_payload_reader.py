import gzip
import hashlib
import json

import pytest
from rosclaw.growth.shared_proof_payload import canonical_hash, detach_payload

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
from rosclaw_soccer.rsi.physical_report_io import (
    CPU_MEAN_MODEL_LOCATION,
    CPU_MODEL_LOCATION,
    MEAN_MODEL_LOCATION,
    MODEL_LOCATION,
)
from scripts.rsi_atomic_artifacts import write_once


def fixture(root, location=CPU_MEAN_MODEL_LOCATION):
    value = {"schema": "fixture.physical.v1", "ball": [0.0, -0.0, 1.234567891e-20]}
    node = value
    for key in location[:-1]:
        node[key] = {}
        node = node[key]
    node[location[-1]] = {"weights": [0.0, -0.0, 1.234567891e-20] * 20}
    value["report_hash"] = canonical_hash(value)
    envelope, payload = detach_payload(value, location)
    folder = root / "case"
    folder.mkdir()
    store = root / ".shared-models"
    store.mkdir()
    payload_path = store / (envelope["payload_hash"][7:] + ".json.gz")
    write_once(payload_path, payload)
    path = folder / "report.json.gz"
    write_once(path, envelope)
    pinned = "sha256:" + hashlib.sha256(payload_path.read_bytes()).hexdigest()
    return path, payload_path, pinned, value


@pytest.mark.parametrize(
    "location", [MODEL_LOCATION, MEAN_MODEL_LOCATION, CPU_MODEL_LOCATION, CPU_MEAN_MODEL_LOCATION]
)
def test_exact_full_logical_document_and_old_report_seal(tmp_path, location):
    path, payload, pinned, expected = fixture(tmp_path, location)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    assert reader.load_sealed(path.parent / "report.json") == _sealed(path) == expected
    assert _sealed(path, physical_payload_reader=reader) == expected
    loaded = reader.load_sealed(path)
    loaded[location[0]]["step_motor_proof"]["model"].clear()
    assert reader.load_sealed(path) == expected


@pytest.mark.parametrize("field", ["ball", "logical_document_hash", "envelope_hash", "report_hash"])
def test_tampering_rejected_even_with_envelope_resealed(tmp_path, field):
    path, payload, pinned, _ = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    with gzip.open(path, "rt") as stream:
        envelope = json.load(stream)
    if field == "ball":
        envelope["stripped_document"]["ball"][0] = 1e-12
    elif field == "report_hash":
        envelope["stripped_document"]["report_hash"] = "sha256:" + "0" * 64
        stripped = envelope["stripped_document"]
        temporary = json.loads(json.dumps(stripped))
        target = temporary
        for key in envelope["location"][:-1]:
            target = target[key]
        with gzip.open(payload, "rt") as stream:
            target[envelope["location"][-1]] = json.load(stream)
        envelope["logical_document_hash"] = canonical_hash(temporary)
    else:
        envelope[field] = "sha256:" + "0" * 64
    if field != "envelope_hash":
        envelope["envelope_hash"] = canonical_hash(
            {k: v for k, v in envelope.items() if k != "envelope_hash"}
        )
    with gzip.open(path, "wt") as stream:
        json.dump(envelope, stream)
    with pytest.raises(ValueError, match="seal changed"):
        reader.load_sealed(path)


def test_source_bytes_rechecked_after_constructor(tmp_path):
    path, payload, pinned, _ = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    payload.write_bytes(payload.read_bytes() + b"trailing")
    with pytest.raises(ValueError, match="source bytes changed"):
        reader.load_sealed(path)


def test_wrong_external_file_hash_rejected(tmp_path):
    _, payload, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="source bytes changed"):
        FixedPhysicalPayloadReader(payload, expected_file_hash="sha256:" + "0" * 64)


def test_cache_bytes_cannot_mutate_silently(tmp_path):
    path, payload, pinned, _ = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    reader._field._payload_bytes = b"{}"
    with pytest.raises(ValueError, match="unchanged cached payload"):
        reader.load_sealed(path)


def test_symlinked_source_rejected(tmp_path):
    path, payload, pinned, _ = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    target = tmp_path / "moved.json.gz"
    payload.rename(target)
    payload.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        reader.load_sealed(path)


def test_no_fallback_for_ambiguous_reports(tmp_path):
    path, payload, pinned, value = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    write_once(path.parent / "report.json", value)
    with pytest.raises(ValueError, match="exactly one"):
        reader.load_sealed(path)


@pytest.mark.parametrize("field", ["location", "payload_hash", "extra", "marker"])
def test_invalid_envelope_has_no_fallback(tmp_path, field):
    path, payload, pinned, _ = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    with gzip.open(path, "rt") as stream:
        value = json.load(stream)
    if field == "location":
        value[field] = ["ball"]
    elif field == "payload_hash":
        value[field] = "../../escape"
    elif field == "extra":
        value[field] = True
    else:
        value["stripped_document"]["executed_motor_policy"]["step_motor_proof"]["model"][
            "mean_model"
        ] = {}
    value["envelope_hash"] = canonical_hash(
        {k: v for k, v in value.items() if k != "envelope_hash"}
    )
    with gzip.open(path, "wt") as stream:
        json.dump(value, stream)
    with pytest.raises(ValueError):
        reader.load_sealed(path)


def test_reader_cannot_relocate_payload_to_other_store(tmp_path):
    path, payload, pinned, _ = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    other = tmp_path / "other" / "case"
    other.mkdir(parents=True)
    relocated = other / path.name
    relocated.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="exact fixed"):
        reader.load_sealed(relocated)


def test_non_physical_artifacts_cannot_use_reader(tmp_path):
    _, payload, pinned, _ = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    with pytest.raises(ValueError, match="exact opt-in"):
        _sealed(tmp_path / "result.json", physical_payload_reader=reader)


def test_arbitrary_reader_object_rejected(tmp_path):
    path, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="exact opt-in"):
        _sealed(path, physical_payload_reader=object())


def test_source_change_during_restoration_rejected(tmp_path, monkeypatch):
    path, payload, pinned, _ = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    original = reader._field.document_hash_at_path

    def change_after_hash(*args):
        result = original(*args)
        payload.write_bytes(payload.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(reader._field, "document_hash_at_path", change_after_hash)
    with pytest.raises(ValueError, match="source bytes changed"):
        reader.load_sealed(path)


def test_uncompressed_payload_limit_checked(tmp_path, monkeypatch):
    import rosclaw_soccer.rsi.fixed_physical_payload_reader as module

    _, payload, pinned, _ = fixture(tmp_path)
    monkeypatch.setattr(module, "_DECOMPRESSED_LIMIT", 256)
    assert payload.stat().st_size < 256
    with pytest.raises(ValueError, match="bounded complete"):
        FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)


@pytest.mark.parametrize("limit", ["_COMPRESSED_LIMIT", "_CANONICAL_LIMIT"])
def test_other_payload_byte_limits_checked(tmp_path, monkeypatch, limit):
    import rosclaw_soccer.rsi.fixed_physical_payload_reader as module

    _, payload, pinned, _ = fixture(tmp_path)
    monkeypatch.setattr(module, limit, 1)
    with pytest.raises(ValueError, match="bounded"):
        FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)


def test_cache_change_after_last_document_hash_rejected(tmp_path, monkeypatch):
    path, payload, pinned, _ = fixture(tmp_path)
    reader = FixedPhysicalPayloadReader(payload, expected_file_hash=pinned)
    original = reader._field.document_hash_at_path
    calls = 0

    def change_after_hash(*args):
        nonlocal calls
        result = original(*args)
        calls += 1
        if calls == 2:
            reader._field._payload_bytes = b"{}"
        return result

    monkeypatch.setattr(reader._field, "document_hash_at_path", change_after_hash)
    with pytest.raises(ValueError, match="authenticated cached payload bytes"):
        reader.load_sealed(path)
