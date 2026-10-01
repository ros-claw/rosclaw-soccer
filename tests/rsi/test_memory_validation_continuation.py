import json
import subprocess
import sys

import pytest

from scripts.rsi_continue_memory_validation import run_stage


def test_success_is_journaled_only_after_process_exit(tmp_path):
    run_stage(tmp_path, "fixture", [sys.executable, "-c", "print('fixture completed')"], timeout=5)
    assert json.loads((tmp_path / "fixture-finished.json").read_text()) == {"exit_code": 0}
    assert "fixture completed" in (tmp_path / "fixture.log").read_text()
    assert (tmp_path / "fixture-started.json").is_file()


def test_nonzero_process_preserves_log_and_does_not_pass(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_stage(
            tmp_path, "failure", [sys.executable, "-c", "print('reason');raise SystemExit(7)"]
        )
    assert json.loads((tmp_path / "failure-finished.json").read_text()) == {"exit_code": 7}
    assert "reason" in (tmp_path / "failure.log").read_text()


def test_timeout_stops_only_owned_new_process_group_and_keeps_failure(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_stage(
            tmp_path, "timeout", [sys.executable, "-c", "import time;time.sleep(20)"], timeout=0.05
        )
    assert json.loads((tmp_path / "timeout-failed.json").read_text()) == {
        "reason": "OWNED_SIM_STAGE_TIMEOUT"
    }
    assert not (tmp_path / "timeout-finished.json").exists()


def test_stage_cannot_overwrite_or_automatically_retry_a_previous_log(tmp_path):
    command = [sys.executable, "-c", "print('original')"]
    run_stage(tmp_path, "existing", command)
    before = (tmp_path / "existing.log").read_bytes()
    with pytest.raises(FileExistsError):
        run_stage(tmp_path, "existing", command)
    assert (tmp_path / "existing.log").read_bytes() == before
