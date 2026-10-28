"""Probe one train-split MotionDecode Ball_Control reference using native FK."""

import argparse
import csv
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset-root", "model", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("clip", "clip-hash", "model-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest

    if a.output.exists() or subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("unused output and frozen source required")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != a.expected_head:
        raise ValueError("exact source head required")
    root = a.dataset_root.resolve()
    clip_path = root / a.clip
    clip = clip_path.resolve()
    if not clip.is_relative_to(root) or clip_path.is_symlink() or not clip.is_file():
        raise ValueError("ordinary in-dataset clip required")
    if digest(clip) != a.clip_hash or digest(a.model) != a.model_hash:
        raise ValueError("exact clip and model hashes required")
    from rosclaw_soccer.training.first_touch_teacher_portfolio import (
        _EXPECTED_COLUMNS,
        _split_for_path,
    )

    if (
        not a.clip.startswith("extracted_s118/3.3.3.Football/3.3.3.4.Ball_Control/")
        or _split_for_path(a.clip) != "train"
    ):
        raise ValueError("explicit original training split ball-control clip required")
    with clip.open(newline="", encoding="utf-8") as file:
        if tuple(next(csv.reader(file))) != _EXPECTED_COLUMNS:
            raise ValueError("canonical root and G1 HG29 CSV header required")

    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.motiondecode_native_reference import native_motion_reference
    from rosclaw_soccer.rsi.native_course_tracking import NativeCourseTracking
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

    values = np.loadtxt(clip, delimiter=",", skiprows=1, dtype=np.float64)
    model = mujoco.MjModel.from_binary_path(str(a.model))
    plan, ref = native_motion_reference(model=model, values=values, source_fps=120.0)
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    tracker = NativeCourseTracking(model, layout, [ref] * 8)
    data = mujoco.MjData(model)
    minimum_score, guard_count = 1.0, 0
    for i in range(len(plan["joint"])):
        data.qpos[:] = plan["native_qpos"][i]
        data.qvel[:] = plan["native_qvel"][i]
        mujoco.mj_forward(model, data)
        score, guard = tracker.measure(data, source=0, frame=i, origin=np.zeros(3))
        minimum_score = min(minimum_score, float(np.min(score)))
        guard_count += int(guard.sum())
    if minimum_score < 0.99999 or guard_count:
        raise ValueError("native FK self-consistency failed")
    if digest(clip) != a.clip_hash or digest(a.model) != a.model_hash:
        raise ValueError("source data changed during probe")
    a.output.mkdir()
    np.savez_compressed(
        a.output / "reference.npz",
        **{"plan_" + k: v for k, v in plan.items()},
        **{"body_" + k: v for k, v in ref.items()},
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="MOTIONDECODE_KINEMATIC_NATIVE_FK_REFERENCE_NOT_TRACKED_SKILL",
            source_head=a.expected_head,
            clip=a.clip,
            clip_hash=a.clip_hash,
            model_hash=a.model_hash,
            license_hash=digest(root / "LICENSE.md"),
            attribution="ChingMu MotionDecode",
            source_fps_assumption=120.0,
            csv_contains_timestamps=False,
            target_fps=50.0,
            source_frames=len(values),
            reference_frames=len(plan["joint"]),
            min_self_tracking=minimum_score,
            self_guard_count=guard_count,
            raw_hash=digest(a.output / "reference.npz"),
            simulator_steps=0,
            ball_contact_truth=False,
            actor_training=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
