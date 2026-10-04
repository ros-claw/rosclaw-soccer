import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.physical_report_io import load_physical_report
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once, write_shared_physical_report


def report(index=0):
    value = {
        "schema": "fixture.physical.v1",
        "contact_motor_policy": {
            "step_motor_proof": {"model": {"weights": [0.0, -0.0, 1.234567891e-20] * 100}},
        },
        "body": [index, 0.12345678912345],
        "hardware_authorized": False,
    }
    return {**value, "report_hash": hash_json(value)}


def publish(root: Path, i=0):
    folder = root / f"case{i}"
    folder.mkdir()
    path = folder / "report.json.gz"
    write_shared_physical_report(path, report(i))
    return path


def test_whole_logical_report_and_old_seal_survive_shared_transport(tmp_path):
    path = publish(tmp_path)
    assert load_physical_report(path) == report()
    assert _sealed(path.parent / "report.json") == report()
    assert load_json_artifact(path)["schema"] == "rosclaw.growth.shared_json_payload.v1"
    assert len(list((tmp_path / ".shared-models").iterdir())) == 1
    before = path.read_bytes()
    write_shared_physical_report(path, report())
    assert path.read_bytes() == before


def test_concurrent_reports_share_one_exact_model_without_losing_body_records(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda i: publish(tmp_path, i), range(4)))
    assert len(list((tmp_path / ".shared-models").iterdir())) == 1
    for i, path in enumerate(paths):
        assert _sealed(path) == report(i)


def test_parent_without_model_remains_full_gzip_report(tmp_path):
    folder = tmp_path / "parent"
    folder.mkdir()
    value = {"schema": "fixture.parent", "body": [1.0]}
    value["report_hash"] = hash_json(value)
    write_shared_physical_report(folder / "report.json.gz", value)
    assert _sealed(folder / "report.json") == value
    assert not (tmp_path / ".shared-models").exists()


def test_sampling_views_keep_distinct_seeds_but_share_exact_mean_model(tmp_path):
    original = []
    for i in range(2):
        value = report(i)
        value["contact_motor_policy"]["step_motor_proof"]["model"] = {
            "schema": "fixture.sampling",
            "seed": i + 42,
            "noise": [i * 0.01],
            "mean_model": {"weights": [0.12345678912345] * 100},
        }
        value.pop("report_hash")
        value["report_hash"] = hash_json(value)
        folder = tmp_path / f"sample{i}"
        folder.mkdir()
        write_shared_physical_report(folder / "report.json.gz", value)
        original.append(value)
    assert len(list((tmp_path / ".shared-models").iterdir())) == 1
    for i, value in enumerate(original):
        assert _sealed(tmp_path / f"sample{i}/report.json") == value


def test_missing_or_changed_payload_rejected(tmp_path):
    path = publish(tmp_path)
    payload = next((tmp_path / ".shared-models").iterdir())
    value = load_json_artifact(payload)
    value["weights"][2] += 1e-30
    with gzip.open(payload, "wt", encoding="utf-8") as stream:
        json.dump(value, stream)
    with pytest.raises(ValueError, match="identity changed"):
        load_physical_report(path)
    payload.unlink()
    with pytest.raises(ValueError, match="complete local"):
        load_physical_report(path)


def test_symlinked_payload_is_rejected_before_reading(tmp_path):
    path = publish(tmp_path)
    payload = next((tmp_path / ".shared-models").iterdir())
    target = tmp_path / "elsewhere.json.gz"
    payload.rename(target)
    payload.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_physical_report(path)


def test_payload_reference_cannot_escape_store(tmp_path):
    path = publish(tmp_path)
    envelope = load_json_artifact(path)
    envelope["payload_hash"] = "../../elsewhere"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(envelope, stream)
    with pytest.raises(ValueError, match="content-addressed"):
        load_physical_report(path)


def test_non_model_locations_are_not_a_physical_transport(tmp_path):
    from rosclaw.growth.shared_proof_payload import detach_payload

    folder = tmp_path / "case"
    folder.mkdir()
    envelope, _ = detach_payload({"other": {}}, ("other",))
    write_once(folder / "report.json.gz", envelope)
    with pytest.raises(ValueError, match="exact content-addressed"):
        load_physical_report(folder / "report.json")


@pytest.mark.parametrize("sampling", [False, True])
def test_cpu_executed_policy_has_exact_lossless_shared_transport(tmp_path, sampling):
    values = []
    for i in range(2):
        value = report(i)
        value["executed_motor_policy"] = value.pop("contact_motor_policy")
        if sampling:
            value["executed_motor_policy"]["step_motor_proof"]["model"] = {
                "seed": i + 42,
                "mean_model": {"weights": [0.0, -0.0, 1.234567891e-20] * 100},
            }
        value.pop("report_hash")
        value["report_hash"] = hash_json(value)
        folder = tmp_path / f"cpu{i}"
        folder.mkdir()
        write_shared_physical_report(folder / "report.json.gz", value)
        values.append(value)
    assert len(list((tmp_path / ".shared-models").iterdir())) == 1
    for i, value in enumerate(values):
        assert _sealed(tmp_path / f"cpu{i}/report.json") == value


def test_null_cpu_foundation_policy_keeps_full_report(tmp_path):
    folder = tmp_path / "cpu"
    folder.mkdir()
    value = {"schema": "fixture.foundation", "executed_motor_policy": None}
    value["report_hash"] = hash_json(value)
    write_shared_physical_report(folder / "report.json.gz", value)
    assert _sealed(folder / "report.json") == value
    assert not (tmp_path / ".shared-models").exists()


def test_ambiguous_policy_aliases_rejected_before_storage(tmp_path):
    folder = tmp_path / "cpu"
    folder.mkdir()
    value = report()
    value["executed_motor_policy"] = value["contact_motor_policy"]
    with pytest.raises(ValueError, match="unambiguous"):
        write_shared_physical_report(folder / "report.json.gz", value)
    assert not (folder / "report.json.gz").exists()
    assert not (tmp_path / ".shared-models").exists()
