"""Wait in the OLD stable interpreter before entering a new installed venv.

The installer may mutate the new interpreter's site-packages, so it must not
load the training script until the original install unit is actually terminal.
Exactly one bounded SIM smoke is dispatched; a failed child is not retried.
"""

import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-python", type=Path, required=True)
    p.add_argument("--install-unit", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection

    if (
        subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or not a.args
        or a.args[0] != "--"
    ):
        raise ValueError("frozen SIM launcher and explicit child arguments required")
    wait_for_terminal_collection(a.install_unit, wait=True)
    if not a.training_python.is_file():
        raise ValueError("completed installer must provide its declared training interpreter")
    command = [
        str(a.training_python),
        "-u",
        "scripts/rsi_mjlab_motion_learning_smoke.py",
        "--install-unit",
        a.install_unit,
        "--expected-head",
        a.expected_head,
        *a.args[1:],
    ]
    print("INSTALL_TERMINAL_START_ONE_FROZEN_MOTION_LEARNING_SMOKE", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
