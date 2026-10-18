import ast
import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.collected_sim_stage_evidence import verify_collected_sim_journal
from scripts import rsi_read_mesh_contact_native_pilot as audit
from tests.rsi.test_release_four_full_terminal_readback import fixture as four_fixture


def records(unit="test.service", command=None):
    command = ["python", "-u", "worker.py"] if command is None else command
    return [
        dict(
            _SYSTEMD_USER_UNIT=unit,
            _CMDLINE=shlex.join(command),
            _SYSTEMD_INVOCATION_ID="b" * 32,
            _PID="987654321",
            _BOOT_ID="a" * 32,
            MESSAGE="CONTACT_EXIT_TORQUE_MPC_COMPLETE_NOT_QUALIFIED_TEACHER",
            __MONOTONIC_TIMESTAMP="100",
            __CURSOR="phase",
        ),
        dict(
            USER_UNIT=unit,
            USER_INVOCATION_ID="b" * 32,
            _BOOT_ID="a" * 32,
            _COMM="systemd",
            _SYSTEMD_USER_UNIT="init.scope",
            _TRANSPORT="journal",
            _EXE="/usr/lib/systemd/systemd",
            MESSAGE_ID="ae8f7b866b0347b9af31fe1c80b127c0",
            __MONOTONIC_TIMESTAMP="200",
            __CURSOR="resource",
            CPU_USAGE_NSEC="10000",
        ),
    ]


def validate(rows, status=None):
    return verify_collected_sim_journal(
        rows,
        status=status or dict(LoadState="not-found", ActiveState="inactive", MainPID="0"),
        unit="test.service",
        expected_command=["python", "-u", "worker.py"],
        terminal_phase="CONTACT_EXIT_TORQUE_MPC_COMPLETE_NOT_QUALIFIED_TEACHER",
    )


def test_collected_handoff_explicitly_does_not_invent_exit_code():
    value = validate(records())
    assert value["os_exit_code_observed"] is False
    assert value["exit_code"] is None
    assert value["complete_and_ledger_verification_required"] is True
    assert value["worker_reexecuted"] is False


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_phase",
        "fake_manager",
        "different_invocation",
        "wrong_command",
        "bad_order",
        "failure",
        "cpu",
        "duplicates",
    ],
)
def test_missing_or_conflicting_journal_cannot_bypass_terminal_gate(tamper):
    rows = records()
    if tamper == "missing_phase":
        rows[0]["MESSAGE"] = "STILL_RUNNING"
    elif tamper == "fake_manager":
        rows[1]["_EXE"] = "/usr/bin/logger"
    elif tamper == "different_invocation":
        rows[1]["USER_INVOCATION_ID"] = "c" * 32
    elif tamper == "wrong_command":
        rows[0]["_CMDLINE"] += " --other"
    elif tamper == "bad_order":
        rows[1]["__MONOTONIC_TIMESTAMP"] = "99"
    elif tamper == "failure":
        rows[1]["SERVICE_RESULT"] = "exit-code"
    elif tamper == "cpu":
        rows[1]["CPU_USAGE_NSEC"] = "0"
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        validate(rows)


def test_still_loaded_or_running_process_is_not_a_collected_stage():
    with pytest.raises(ValueError):
        validate(records(), dict(LoadState="loaded", ActiveState="active", MainPID="123"))


def test_full_collected_mpc_reader_still_verifies_all_four_endpoints(tmp_path, monkeypatch):
    output = four_fixture(tmp_path, monkeypatch, torque_mpc=True)
    config = output.parent / "config.json"
    command = [
        sys.executable,
        "-u",
        "scripts/rsi_swing_release_ablation_pilot.py",
        "--config",
        str(config),
    ]
    path = tmp_path / "collected-journal.json"
    path.write_text(json.dumps(records("fake.service", command)))
    monkeypatch.setattr(
        sys,
        "argv",
        sys.argv
        + [
            "--collected-unit-journal",
            str(path),
            "--collected-unit-journal-hash",
            audit.digest(path),
        ],
    )
    original = audit.subprocess.check_output

    def system_output(cmd, **kwargs):
        if cmd[:4] == ["systemctl", "--user", "show", "fake.service"]:
            return "LoadState=not-found\nActiveState=inactive\nMainPID=0\nResult=success\n"
        return original(cmd, **kwargs)

    monkeypatch.setattr(audit.subprocess, "check_output", system_output)
    audit.main()
    result = json.loads(output.read_text())
    assert result["worker_os_exit_code_proven"] is False
    assert result["whole_input_and_witness_files_verified"] is True
    assert len(result["rows"]) == 4
    assert result["collected_sim_stage_handoff"]["exit_code"] is None


def test_journal_helper_does_not_pin_reviewer_package_before_producer_binding():
    helper = (
        Path(audit.__file__).parents[1] / "src/rosclaw_soccer/rsi/collected_sim_stage_evidence.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,runpy;assert 'rosclaw_soccer' not in sys.modules;"
            "runpy.run_path(sys.argv[1]);assert 'rosclaw_soccer' not in sys.modules",
            str(helper),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    main = next(
        n
        for n in ast.parse(Path(audit.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    prefix = []
    for statement in main.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Attribute)
            and isinstance(t.value.value, ast.Name)
            and t.value.value.id == "sys"
            and t.value.attr == "path"
            for t in statement.targets
        ):
            break
        prefix.append(statement)
    assert not any(
        isinstance(n, ast.ImportFrom) and (n.module or "").startswith("rosclaw_soccer")
        for statement in prefix
        for n in ast.walk(statement)
    )
