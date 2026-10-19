"""Read-only campaign watch; liveness is never learning or qualification evidence.

The agent, not this observer, owns scientific successor decisions. In particular,
an unsafe rollout is data, whereas an integrity failure requires a repair review.
Neither condition authorizes silently retrying an already charged attempt.
"""

import hashlib
import json
import math
import sqlite3
import subprocess
from pathlib import Path
from typing import Any


def collect(unit: str, root: Path) -> dict[str, Any]:
    """Observe a fixed systemd unit and the latest ledger row without changing it."""
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "LoadState",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    service = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    observation: dict[str, Any] = dict(unit=unit, service=service, ledger=None, cpu_ticks=None)
    pid = int(service.get("MainPID", "0"))
    if pid:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            observation["cpu_ticks"] = int(fields[11]) + int(fields[12])
            observation["process_start_ticks"] = int(fields[19])
        except FileNotFoundError:
            observation["process_disappeared_during_read"] = True
    ledger = root / "campaign.sqlite"
    if ledger.is_file():
        with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True, timeout=5) as db:
            row = db.execute(
                "SELECT seq, digest, body FROM events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
        if row is not None:
            state = json.loads(row[2])["state"]
            observation["ledger"] = dict(
                seq=row[0],
                tip=row[1],
                phase=state["phase"],
                attempts=state["attempts"],
                completed=len(state["completed"]),
                active=state["active"],
            )
    # Serial diagnostic runners publish ordinary JSON checkpoints, not SQLite.
    # Metadata change is progress observation, never scientific verification.
    files = sorted(root.glob("*.json"))
    if len(files) > 2048:
        raise ValueError("bounded stage checkpoint directory required")
    metadata = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError("ordinary stage checkpoint required")
        stat = path.stat()
        metadata.append((path.name, stat.st_size, stat.st_mtime_ns))
    if metadata:
        observation["artifacts"] = dict(
            tip=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
            checkpoint_count=len(metadata),
            complete_present=any(name == "complete.json" for name, _, _ in metadata),
            failure_present=any(name == "failure.json" for name, _, _ in metadata),
            contents_verified=False,
        )
    return observation


def assess(
    observation: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    now: float,
    quiet_seconds: float = 1200,
) -> dict[str, Any]:
    """Keep process activity, committed science, and next agent work distinct."""
    if not all(math.isfinite(value) for value in (now, quiet_seconds)) or quiet_seconds <= 0:
        raise ValueError("finite watch clock and positive quiet interval required")
    previous = previous or {}
    if not all(key in previous for key in ("last_ledger_change_unix", "last_cpu_change_unix")):
        previous = {}
    service = observation["service"]
    ledger = observation.get("ledger") or {}
    artifacts = observation.get("artifacts") or {}
    tip = (ledger.get("tip"), artifacts.get("tip"))
    identity = (service.get("MainPID"), observation.get("process_start_ticks"))
    old_service = previous.get("service", {})
    old_identity = (old_service.get("MainPID"), previous.get("process_start_ticks"))
    same_process = identity == old_identity and identity[1] is not None
    ticks = observation.get("cpu_ticks")
    old_ticks = previous.get("cpu_ticks")
    cpu_advanced = (
        same_process and ticks is not None and old_ticks is not None and ticks > old_ticks
    )
    old_tip = (
        (previous.get("ledger") or {}).get("tip"),
        (previous.get("artifacts") or {}).get("tip"),
    )
    committed_at = now if tip != old_tip or not previous else previous["last_ledger_change_unix"]
    active_at = now if cpu_advanced or not same_process else previous["last_cpu_change_unix"]
    # A rollback in wall time must not falsely create a long-idle diagnosis.
    committed_at, active_at = min(now, committed_at), min(now, active_at)
    running = service.get("ActiveState") == "active" and service.get("MainPID") not in (None, "0")
    if service.get("LoadState") != "loaded":
        phase, next_action = "MISSING_UNIT", "REPAIR_MISSING_DECLARED_WORKER"
    elif (
        service.get("ActiveState") == "failed"
        or service.get("Result") not in (None, "success")
        or artifacts.get("failure_present")
    ):
        phase, next_action = "FAILED", "PRESERVE_FAILURE_AND_DIAGNOSE_NO_BLIND_RETRY"
    elif running:
        if now - active_at >= quiet_seconds:
            phase, next_action = (
                "NO_OBSERVED_CPU_PROGRESS",
                "INVESTIGATE_WAIT_OR_HANG_WITHOUT_KILLING",
            )
        elif now - committed_at >= quiet_seconds:
            phase, next_action = "COMPUTING_WITHOUT_NEW_COMMIT", "CHECK_STAGE_PROGRESS_NOT_JUST_CPU"
        else:
            phase, next_action = "RUNNING", "CONTINUE_AGENT_DEVELOPMENT_AND_WAIT_FOR_EVIDENCE"
    elif service.get("ActiveState") in ("activating", "deactivating"):
        phase, next_action = "TRANSITIONING", "RECHECK_BOUNDED_TRANSITION"
    elif artifacts.get("complete_present"):
        phase, next_action = (
            "STAGE_ARTIFACT_READY_FOR_READBACK",
            "AUTHENTICATE_ALL_RESULTS_AND_DECLARE_SCIENTIFIC_SUCCESSOR",
        )
    else:
        phase, next_action = "NO_RUNNING_WORKER", "REVIEW_LEDGER_AND_DECLARE_NEXT_SCIENTIFIC_STAGE"
    return dict(
        **observation,
        observed_unix=now,
        phase=phase,
        next_agent_action=next_action,
        cpu_advanced_since_previous_observation=cpu_advanced,
        last_ledger_change_unix=committed_at,
        last_cpu_change_unix=active_at,
        ledger_integrity_verified=False,
        outcome_verified=False,
        goal_complete=False,
        automatic_retry=False,
        hardware_authorized=False,
    )
