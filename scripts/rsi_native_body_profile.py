"""Read native physical and layout metadata; never allocate a physical rollout."""

import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("expected-world-hash", "expected-head", "robot-prefix", "ball-body-name"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or digest(a.world) != a.expected_world_hash
    ):
        raise ValueError("immutable declared model, frozen source and unused output required")
    pins = {str(a.world): a.expected_world_hash}
    for name in (
        "scripts/rsi_native_body_profile.py",
        "scripts/rsi_atomic_artifacts.py",
        "scripts/rsi_native_context_recovery_common.py",
        "src/rosclaw_soccer/rsi/native_velocity_transfer.py",
        "src/rosclaw_soccer/providers/g1/joint_contract.py",
    ):
        path = Path.cwd() / name
        pins[str(path)] = digest(path)
    check_pins(pins)
    import mujoco

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

    model = mujoco.MjModel.from_binary_path(str(a.world))
    profile = NativeVelocityLayout(
        model, robot_prefix=a.robot_prefix, ball_body_name=a.ball_body_name
    ).profile()
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(a.output / "profile.json", profile)
    write_once(
        a.output / "complete.json",
        dict(
            status="NATIVE_BODY_METADATA_NOT_MOTION_QUALIFIED",
            input_file_hashes=pins,
            source_head=a.expected_head,
            engine_version=mujoco.__version__,
            profile_hash=digest(a.output / "profile.json"),
            new_native_steps=0,
            new_neural_calls=0,
            new_optimizer_steps=0,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
