from pathlib import Path

import pytest
from rosclaw.growth.shared_proof_payload import canonical_hash, detach_payload

from rosclaw_soccer.rsi.first_touch_campaign_worker import qualified_episode
from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
from rosclaw_soccer.rsi.physical_report_io import CPU_MEAN_MODEL_LOCATION
from rosclaw_soccer.sim.contracts import hash_bytes
from scripts.rsi_atomic_artifacts import write_once


def call_with_reader(folder, reader):
    # Refusal must precede access to deliberately absent model/physics inputs.
    return qualified_episode(
        folder,
        0,
        None,
        plan=None,
        manifest=None,
        assets=None,
        source=Path("absent"),
        core=Path("absent"),
        factory=None,
        compiler=None,
        auditor=None,
        physical_payload_reader=reader,
    )


def test_arbitrary_cache_cannot_enter_qualified_episode(tmp_path):
    with pytest.raises(ValueError, match="episode-owned"):
        call_with_reader(tmp_path / "episode", object())


def test_cache_cannot_read_another_experiment_store(tmp_path):
    value = {
        "executed_motor_policy": {"step_motor_proof": {"model": {"mean_model": {"weights": [0.1]}}}}
    }
    value["report_hash"] = canonical_hash(value)
    envelope, payload = detach_payload(value, CPU_MEAN_MODEL_LOCATION)
    store = tmp_path / "one" / ".shared-models"
    store.mkdir(parents=True)
    path = store / (envelope["payload_hash"][7:] + ".json.gz")
    write_once(path, payload)
    reader = FixedPhysicalPayloadReader(path, expected_file_hash=hash_bytes(path.read_bytes()))
    with pytest.raises(ValueError, match="episode-owned"):
        call_with_reader(tmp_path / "another" / "episode", reader)
