"""Sealed private FK audit of a reference reflection; zero dynamics or NN steps."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("world", "motion-csv", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-world-hash", "expected-motion-hash"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ):
        raise ValueError("clean pinned source required before native allocation")
    files = [Path(__file__).resolve(), *Path("src").rglob("*.py")]
    pins = {str(p): digest(p) for p in files}
    for path, expected in (
        (args.world, args.expected_world_hash),
        (args.motion_csv, args.expected_motion_hash),
    ):
        if digest(path) != expected:
            raise ValueError("input identity differs")
        pins[str(path)] = expected
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output / "declaration.json", dict(source_head=head, input_file_hashes=pins))
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_sagittal_reference import mirror_sagittal_reference

    poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)[286:767]
    reflected = mirror_sagittal_reference(poses)
    np.testing.assert_array_equal(mirror_sagittal_reference(reflected), poses)
    model = mujoco.MjModel.from_binary_path(str(args.world))
    original, mirrored = mujoco.MjData(model), mujoco.MjData(model)
    joint_ids = [model.joint(n).id for n in G1_DDS_JOINT_NAMES]
    addresses = model.jnt_qposadr[joint_ids]
    if not np.array_equal(addresses, np.arange(7, 36)):
        raise ValueError("canonical named native joint mapping required")
    reflection = np.diag([1.0, -1.0, 1.0])
    pairs = []
    for i in range(1, model.nbody):
        name = model.body(i).name
        if name.startswith("left_"):
            pairs.append((i, model.body("right_" + name[5:]).id))
    if len(pairs) < 12:
        raise ValueError("named bilateral body coverage required")
    max_position = max_rotation = 0.0
    violations = []
    for frame, (q, mirror_q) in enumerate(zip(poses, reflected, strict=True)):
        for data, pose in ((original, q), (mirrored, mirror_q)):
            data.qpos[:36] = pose  # PRIVATE FK data only; no rollout/simulation state.
            mujoco.mj_forward(model, data)
        for left, right in pairs:
            for first, second in ((left, right), (right, left)):
                max_position = max(
                    max_position,
                    float(
                        np.linalg.norm(mirrored.xpos[second] - reflection @ original.xpos[first])
                    ),
                )
                max_rotation = max(
                    max_rotation,
                    float(
                        np.max(
                            abs(
                                mirrored.xmat[second].reshape(3, 3)
                                - reflection @ original.xmat[first].reshape(3, 3) @ reflection
                            )
                        )
                    ),
                )
        for j, joint in enumerate(joint_ids):
            if model.jnt_limited[joint] and not (
                model.jnt_range[joint, 0] <= mirror_q[7 + j] <= model.jnt_range[joint, 1]
            ):
                violations.append(
                    dict(frame=frame, joint=G1_DDS_JOINT_NAMES[j], value=float(mirror_q[7 + j]))
                )
    for path, expected in pins.items():
        if digest(Path(path)) != expected:
            raise ValueError("source/input changed during audit")
    report = dict(
        source_head=head,
        source_frames=len(poses),
        bilateral_body_pairs=len(pairs),
        maximum_body_position_reflection_error_m=max_position,
        maximum_body_rotation_reflection_error=max_rotation,
        mirrored_native_limit_violations=violations,
        kinematic_augmentation_admissible=max_position <= 1e-6
        and max_rotation <= 1e-6
        and not violations,
        new_native_integration_steps=0,
        new_neural_calls=0,
        optimizer_updates=0,
        private_fk_evaluations=2 * len(poses),
        dynamic_tracking_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "complete.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
