"""Wait in the OLD stable interpreter before entering a new installed venv.

The installer may mutate the new interpreter's site-packages, so it must not
load the training script until the original install unit is actually terminal.
Exactly one bounded SIM smoke is dispatched; a failed child is not retried.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-python", type=Path, required=True)
    p.add_argument("--install-unit", required=True)
    p.add_argument("--installation-output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
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
    metadata = subprocess.check_output(
        [
            str(a.training_python),
            "-c",
            "import json,sys;from importlib.metadata import version;"
            "print(json.dumps(dict(python=sys.version,prefix=sys.prefix,packages={p:version(p) "
            "for p in ('mjlab','mujoco','mujoco-warp','warp-lang','torch','rsl-rl-lib')})))",
        ],
        text=True,
    )
    a.installation_output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.installation_output / "complete.json",
        dict(
            status="INSTALL_UNIT_TERMINAL_PACKAGE_METADATA_READ_NOT_PHYSICS_QUALIFICATION",
            source_head=a.expected_head,
            installer_unit=a.install_unit,
            **json.loads(metadata),
            new_native_steps=0,
            new_optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
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
