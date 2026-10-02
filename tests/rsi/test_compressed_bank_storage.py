from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import rsi_compressed_bank_storage as storage
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_step_motor_pilot import COURSES


def test_budget_includes_largest_each_arm_all_52_logs_and_scratch(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    largest = {}
    for arm, kind in (("reproduction", "parent"), ("warm", "actor"), ("online", "actor")):
        sizes = []
        for i, (seed, lane) in enumerate(COURSES):
            stem = f"seed{seed}-lane{lane}-{arm}-{kind}"
            folder = tmp_path / stem
            folder.mkdir()
            report = folder / "report.json.gz"
            write_once(report, {"fixture": True, "frame": i})
            (folder / "body_trace.npz").write_bytes(b"fixture" * (i + 1))
            log = tmp_path / "logs" / f"{stem}.log"
            log.write_text("native\n" * (i + 1))
            sizes.append(report.stat().st_size + 7 * (i + 1) + log.stat().st_size)
        largest[arm] = max(sizes)
    expected = (52 * sum(largest.values()) * 115 + 99) // 100 + 1024**3
    assert storage.measured_bank_budget(tmp_path) == expected
    store = tmp_path / ".shared-models"
    store.mkdir()
    (store / "fixture.json.gz").write_bytes(b"shared-model" * 100)
    expected_shared = ((52 * sum(largest.values()) + 1200) * 115 + 99) // 100 + 1024**3
    assert storage.measured_bank_budget(tmp_path) == expected_shared
    log.unlink()
    with pytest.raises(ValueError, match="native log"):
        storage.measured_bank_budget(tmp_path)


def test_no_partial_pilot_capacity_estimate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        storage.measured_bank_budget(tmp_path)


def paths(monkeypatch, *, same: bool, system_free: int, evidence_free: int):
    system, evidence = Path("fixture-system"), Path("fixture-evidence")
    monkeypatch.setattr(
        Path, "stat", lambda p: SimpleNamespace(st_dev=1 if p == system or same else 2)
    )
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=system_free if p == system else evidence_free),
    )
    return evidence, system


def test_separate_volume_preserves_both_reserves(monkeypatch) -> None:
    evidence, system = paths(
        monkeypatch, same=False, system_free=100 * 1024**3, evidence_free=4 * 1024**3
    )
    result = storage.capacity_check(evidence, system, 3 * 1024**3)
    assert result["shared_volume"] is False
    assert result["system_reserve_bytes"] == 100 * 1024**3
    with pytest.raises(ValueError, match="reserve"):
        storage.capacity_check(evidence, system, 3 * 1024**3 + 1)


def test_small_evidence_volume_cannot_hide_low_system_space(monkeypatch) -> None:
    evidence, system = paths(
        monkeypatch, same=False, system_free=100 * 1024**3 - 1, evidence_free=50 * 1024**3
    )
    with pytest.raises(ValueError, match="reserve"):
        storage.capacity_check(evidence, system, 3 * 1024**3)


def test_shared_volume_budget_is_added_to_system_reserve(monkeypatch) -> None:
    evidence, system = paths(
        monkeypatch, same=True, system_free=102 * 1024**3, evidence_free=102 * 1024**3
    )
    with pytest.raises(ValueError, match="reserve"):
        storage.capacity_check(evidence, system, 3 * 1024**3)
    assert storage.capacity_check(evidence, system, 2 * 1024**3)["shared_volume"] is True


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_cannot_bypass_gate(tmp_path: Path, budget) -> None:
    with pytest.raises(ValueError, match="positive"):
        storage.capacity_check(tmp_path, tmp_path, budget)
