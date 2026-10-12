from pathlib import Path

import pytest

from rosclaw_soccer.rsi.first_touch_recovery import WITNESSES, FirstTouchRecoveryLedger
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan


def plan() -> PublicFirstTouchTrainingPlan:
    return PublicFirstTouchTrainingPlan((20261060,), "sha256:" + "a" * 64, 18)


def artifacts(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for key in WITNESSES:
        path = tmp_path / key
        path.write_bytes(key.encode())
        result[key] = path
    return result


def test_reopen_recovers_exact_pending_work_and_is_idempotent(tmp_path: Path) -> None:
    files = artifacts(tmp_path)
    root = tmp_path / "campaign"
    ledger = FirstTouchRecoveryLedger(root, plan())
    assert ledger.status()["state"] == "PREFLIGHT"
    ledger.record_episode(0, files)
    head = ledger.status()["head_hash"]
    ledger.record_episode(0, files)
    assert ledger.status()["head_hash"] == head
    ledger.failure("interrupted worker")
    reopened = FirstTouchRecoveryLedger(root, plan())
    assert reopened.status()["state"] == "RESUMABLE_FAILURE"
    assert reopened.status()["completed_groups"] == [0]
    assert reopened.status()["pending_groups"] == list(range(1, 16))
    reopened.record_episode(1, files)
    assert reopened.status()["state"] == "PARTIAL_DATA"
    assert reopened.status()["training_authorized"] is False


def test_complete_means_all_declared_groups_not_only_successes(tmp_path: Path) -> None:
    files = artifacts(tmp_path)
    ledger = FirstTouchRecoveryLedger(tmp_path / "campaign", plan())
    for group in range(16):
        ledger.record_episode(group, files)
    assert ledger.status()["state"] == "COLLECTED"
    assert ledger.status()["pending_groups"] == []
    with pytest.raises(ValueError):
        ledger.failure("must not reopen complete bank")


def test_mutated_witness_must_not_be_silently_dropped(tmp_path: Path) -> None:
    files = artifacts(tmp_path)
    ledger = FirstTouchRecoveryLedger(tmp_path / "campaign", plan())
    ledger.record_episode(0, files)
    files["trace"].write_text("changed")
    with pytest.raises(ValueError, match="mutation"):
        ledger.status()


def test_fatal_freeze_survives_restart(tmp_path: Path) -> None:
    files = artifacts(tmp_path)
    root = tmp_path / "campaign"
    ledger = FirstTouchRecoveryLedger(root, plan())
    ledger.failure("physical contract mismatch", fatal=True)
    reopened = FirstTouchRecoveryLedger(root, plan())
    assert reopened.status()["state"] == "FATAL_CONTRACT_VIOLATION"
    with pytest.raises(ValueError):
        reopened.record_episode(0, files)


def test_changed_plan_or_reward_cannot_resume_same_campaign(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    FirstTouchRecoveryLedger(root, plan())
    changed = PublicFirstTouchTrainingPlan((20261061,), "sha256:" + "a" * 64, 18)
    with pytest.raises(ValueError):
        FirstTouchRecoveryLedger(root, changed)


@pytest.mark.parametrize("bad", [True, -1, 16, 0.0])
def test_invalid_group_rejected(tmp_path: Path, bad: int) -> None:
    ledger = FirstTouchRecoveryLedger(tmp_path / "campaign", plan())
    with pytest.raises(ValueError):
        ledger.record_episode(bad, artifacts(tmp_path))


def test_torn_journal_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    ledger = FirstTouchRecoveryLedger(root, plan())
    ledger.record_episode(0, artifacts(tmp_path))
    with (root / "journal.jsonl").open("ab") as stream:
        stream.write(b'{"partial":')
    with pytest.raises(ValueError, match="torn"):
        FirstTouchRecoveryLedger(root, plan())


def test_missing_journal_is_not_a_new_campaign(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    ledger = FirstTouchRecoveryLedger(root, plan())
    ledger.record_episode(0, artifacts(tmp_path))
    (root / "journal.jsonl").unlink()
    with pytest.raises(ValueError, match="missing durable journal"):
        FirstTouchRecoveryLedger(root, plan())


def test_witness_and_declaration_tampering_rejected(tmp_path: Path) -> None:
    files = artifacts(tmp_path)
    root = tmp_path / "campaign"
    ledger = FirstTouchRecoveryLedger(root, plan())
    missing = dict(files)
    del missing["replay"]
    with pytest.raises(ValueError):
        ledger.record_episode(0, missing)
    link = tmp_path / "linked"
    link.symlink_to(files["trace"])
    with pytest.raises(ValueError):
        ledger.record_episode(0, dict(files, trace=link))
    (root / "declaration.json").write_text("{}")
    with pytest.raises(ValueError):
        ledger.status()
