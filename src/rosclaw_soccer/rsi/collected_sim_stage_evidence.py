"""Explicit handoff for a collected SIM process, never a fabricated exit code.

Trusted journal process/manager metadata corroborates completion. The caller
must independently verify complete artifacts, source pins and the whole ledger.
"""

import re
import shlex
from pathlib import Path
from typing import Any


def verify_collected_sim_journal(
    records: list[dict[str, Any]],
    *,
    status: dict[str, str],
    unit: str,
    expected_command: list[str],
    terminal_phase: str,
) -> dict[str, Any]:
    if (
        status.get("LoadState") != "not-found"
        or status.get("ActiveState") != "inactive"
        or status.get("MainPID") != "0"
        or not isinstance(records, list)
        or not 1 <= len(records) <= 10000
        or any(type(row) is not dict for row in records)
    ):
        raise ValueError("actually absent inactive SIM unit and bounded journal required")
    principal = [
        row
        for row in records
        if row.get("_SYSTEMD_USER_UNIT") == unit
        and shlex.split(row.get("_CMDLINE", "")) == expected_command
    ]
    invocations = {r.get("_SYSTEMD_INVOCATION_ID") for r in principal}
    pids = {r.get("_PID") for r in principal}
    boots = {r.get("_BOOT_ID") for r in principal}
    if (
        len(invocations) != 1
        or len(pids) != 1
        or len(boots) != 1
        or any(
            not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{32}", v) is None
            for v in (*invocations, *boots)
        )
        or any(not isinstance(v, str) or not v.isdigit() or int(v) <= 0 for v in pids)
    ):
        raise ValueError("one exact worker command, PID, boot and invocation required")
    invocation, pid, boot = next(iter(invocations)), next(iter(pids)), next(iter(boots))
    assert isinstance(pid, str)  # Validated above; keep the filesystem target typed.
    if Path("/proc", pid).exists():
        raise ValueError("recorded worker PID still exists; no collected-stage handoff")
    phases = [r for r in principal if r.get("MESSAGE") == terminal_phase]
    manager = [
        r
        for r in records
        if r.get("USER_UNIT") == unit and r.get("USER_INVOCATION_ID") == invocation
    ]
    resources = [
        r
        for r in manager
        if r.get("MESSAGE_ID") == "ae8f7b866b0347b9af31fe1c80b127c0"
        and r.get("_COMM") == "systemd"
        and r.get("_SYSTEMD_USER_UNIT") == "init.scope"
        and r.get("_TRANSPORT") == "journal"
        and r.get("_BOOT_ID") == boot
        and r.get("_EXE") in {"/usr/lib/systemd/systemd", "/lib/systemd/systemd"}
    ]
    if len(phases) != 1 or len(resources) != 1:
        raise ValueError("exact terminal phase and trusted manager resource closure required")
    phase, resource = phases[0], resources[0]
    try:
        phase_time = int(phase["__MONOTONIC_TIMESTAMP"])
        resource_time = int(resource["__MONOTONIC_TIMESTAMP"])
        cpu_time = int(resource["CPU_USAGE_NSEC"])
        last_worker_time = max(int(r["__MONOTONIC_TIMESTAMP"]) for r in principal)
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("trusted journal monotonic clocks and CPU accounting required") from error
    if (
        not 0 < phase_time <= last_worker_time < resource_time
        or cpu_time <= 0
        or not phase.get("__CURSOR")
        or not resource.get("__CURSOR")
        or any(
            ("SERVICE_RESULT" in r and r["SERVICE_RESULT"] != "success")
            or ("JOB_RESULT" in r and r["JOB_RESULT"] != "done")
            or ("EXIT_CODE" in r and r["EXIT_CODE"] != "exited")
            or ("EXIT_STATUS" in r and str(r["EXIT_STATUS"]) != "0")
            for r in manager
        )
    ):
        raise ValueError("consistent terminal ordering and no explicit failure required")
    return dict(
        schema="soccer.rsi.collected_sim_stage_handoff.v1",
        unit=unit,
        boot_id=boot,
        invocation_id=invocation,
        recorded_worker_pid=int(pid),
        terminal_phase=terminal_phase,
        phase_cursor=phase["__CURSOR"],
        resource_closure_cursor=resource["__CURSOR"],
        cpu_usage_nsec=cpu_time,
        unit_is_actually_collected=True,
        os_exit_code_observed=False,
        exit_code=None,
        complete_and_ledger_verification_required=True,
        worker_reexecuted=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
