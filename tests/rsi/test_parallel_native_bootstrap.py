import json
import subprocess
import sys

from scripts.rsi_collect_parallel_motor_experience import BOOTSTRAP


def test_exact_collector_mp_main_and_pickle_bootstrap_allocates_no_physics(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert json.loads(result.stdout) == dict(
        module_registered=True, entry_pickle_roundtrip=True, no_physics_created=True
    )
    assert not list(tmp_path.iterdir())
