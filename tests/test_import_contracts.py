from __future__ import annotations

import subprocess
import sys


def test_world_field_import_is_cold_start_safe() -> None:
    process = subprocess.run(
        [sys.executable, "-c", "from rosclaw_soccer.world.field import G1TrainingGoalSpec"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_g1_namespace_does_not_eagerly_import_growth() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rosclaw_soccer.providers.g1; "
            "assert 'rosclaw_soccer.growth.role_learning' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
