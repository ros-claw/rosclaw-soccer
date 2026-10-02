"""Measured next-bank capacity; no changes to physical acceptance thresholds."""

import shutil
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.physical_report_io import resolve_physical_report
from scripts.rsi_collect_step_motor_pilot import COURSES
from scripts.rsi_prepare_smooth_memory_round_two import folder_bytes


def measured_bank_budget(pilot_root: Path) -> int:
    """Count complete reports, traces AND native logs, then add growth/scratch."""
    largest = {}
    for arm, kind in (("reproduction", "parent"), ("warm", "actor"), ("online", "actor")):
        sizes = []
        for seed, lane in COURSES:
            stem = f"seed{seed}-lane{lane}-{arm}-{kind}"
            folder = pilot_root / stem
            if resolve_physical_report(folder / "report.json").name != "report.json.gz":
                raise ValueError("measured budget requires complete lossless compressed pilot")
            log = pilot_root / "logs" / f"{stem}.log"
            if not log.is_file():
                raise ValueError("native log must enter measured capacity budget")
            sizes.append(folder_bytes(folder) + log.stat().st_size)
        largest[arm] = max(sizes)
    store = pilot_root / ".shared-models"
    shared = 0
    if store.exists():
        if store.is_symlink() or any(p.is_symlink() for p in store.rglob("*")):
            raise ValueError("local shared proof store required for measured budget")
        shared = folder_bytes(store)
    return ((52 * sum(largest.values()) + shared) * 115 + 99) // 100 + 1024**3


def capacity_check(evidence: Path, system: Path, budget: int) -> dict[str, Any]:
    if type(budget) is not int or budget <= 0:
        raise ValueError("positive measured complete-bank budget required")
    system_reserve, evidence_reserve = 100 * 1024**3, 1024**3
    same_volume = evidence.stat().st_dev == system.stat().st_dev
    system_free = shutil.disk_usage(system).free
    evidence_free = shutil.disk_usage(evidence).free
    required = budget + (system_reserve if same_volume else evidence_reserve)
    if system_free < system_reserve or evidence_free < required:
        raise ValueError("complete bank exceeds evidence/system reserve; no truncated bank")
    return dict(
        system_device=system.stat().st_dev,
        evidence_device=evidence.stat().st_dev,
        system_reserve_bytes=system_reserve,
        evidence_volume_reserve_bytes=system_reserve if same_volume else evidence_reserve,
        measured_bank_budget_bytes=budget,
        system_free_bytes=system_free,
        evidence_free_bytes=evidence_free,
        shared_volume=same_volume,
    )
