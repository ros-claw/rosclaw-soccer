"""Producer provenance and independent reviewer provenance must stay separate."""

import copy
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
    _contract,
    validate_compilation_contract,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@pytest.fixture
def historical(tmp_path):
    soccer, core = tmp_path / "producer-soccer", tmp_path / "producer-core"
    files = (
        soccer / "src/rosclaw_soccer/rsi/producer.py",
        core / "src/rosclaw/growth/memory.py",
        soccer / "src/rosclaw_soccer/sim/contracts.py",
    )
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# synthetic independently selected source\n")
    mean = {"synthetic": True}
    policy = {
        "step_motor_proof": {
            "model": {"schema": "soccer.rsi.recurrent_motor_sampling.v1", "mean_model": mean}
        },
        "recurrent_sampling_motor_proof": {},
    }
    policy["policy_hash"] = hash_json(policy)
    contract = _contract(hash_json(mean), {str(f): hash_bytes(f.read_bytes()) for f in files})
    return (soccer, core), files, contract, policy


def test_explicit_historical_sources_not_current_reviewer(historical):
    roots, _, contract, policy = historical
    validate_compilation_contract(contract, policy, source_roots=roots)
    with pytest.raises(ValueError, match="fixed mean/source"):
        validate_compilation_contract(contract, policy)


@pytest.mark.parametrize("change", ["mean", "pins", "authority", "extra", "missing"])
def test_historical_contract_still_requires_every_binding(historical, change):
    roots, _, contract, policy = historical
    changed = copy.deepcopy(contract)
    if change == "mean":
        changed["complete_canonical_mean_hash"] = "sha256:" + "0" * 64
    elif change == "pins":
        changed["source_pins"][next(iter(changed["source_pins"]))] = "sha256:" + "0" * 64
    elif change == "authority":
        changed["hardware_authorized"] = True
    elif change == "extra":
        changed["source_pins"]["/untrusted/declaration/path"] = "sha256:" + "0" * 64
    else:
        changed["source_pins"].pop(next(iter(changed["source_pins"])))
    with pytest.raises(ValueError, match="fixed mean/source"):
        validate_compilation_contract(changed, policy, source_roots=roots)


def test_changed_or_symlinked_source_rejected(historical):
    roots, files, contract, policy = historical
    original = files[0].read_bytes()
    files[0].write_text("# drift\n")
    with pytest.raises(ValueError, match="fixed mean/source"):
        validate_compilation_contract(contract, policy, source_roots=roots)
    files[0].write_bytes(original)
    files[0].unlink()
    files[0].symlink_to(files[1])
    with pytest.raises(ValueError, match="ordinary complete"):
        validate_compilation_contract(contract, policy, source_roots=roots)


@pytest.mark.parametrize(
    "roots", [(), (Path("relative"), Path("/other")), [Path("/a"), Path("/b")]]
)
def test_invalid_explicit_roots_rejected(historical, roots):
    _, _, contract, policy = historical
    with pytest.raises(ValueError, match="two explicit absolute"):
        validate_compilation_contract(contract, policy, source_roots=roots)


def test_archive_source_option_cannot_skip_independent_reference(tmp_path):
    with pytest.raises(ValueError, match="requires independent sampling reference"):
        audit_cpu_transfer(
            tmp_path / "absent",
            tmp_path / "absent.py",
            sampling_factory_source_roots=(tmp_path, tmp_path),
        )
