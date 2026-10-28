"""Build a preregistered eight-clip train-only MotionDecode FK reference bank.

No contact labels, physical simulation, optimizer updates or promotion. Raw
data stays outside Git and ChingMu research/prototyping license applies.
"""

import argparse
import csv
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset-root", "model", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("model-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--train-offset", type=int, choices=(0, 8), default=0)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest

    if a.output.exists() or subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("unused output and frozen clean source required")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != a.expected_head:
        raise ValueError("exact source head required")
    if digest(a.model) != a.model_hash:
        raise ValueError("exact native model required")
    from rosclaw_soccer.training.first_touch_teacher_portfolio import (
        _EXPECTED_COLUMNS,
        _split_for_path,
    )

    root = a.dataset_root.resolve()
    folder = root / "extracted_s118/3.3.3.Football/3.3.3.4.Ball_Control"
    eligible = [
        p
        for p in sorted(folder.glob("*.csv"))
        if _split_for_path(p.relative_to(root).as_posix()) == "train"
    ]
    selected = eligible[a.train_offset : a.train_offset + 8]
    if len(selected) != 8 or any(
        path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024 for path in selected
    ):
        raise ValueError("first eight bounded non-symlink train clips required")
    hashes = [digest(path) for path in selected]
    if len(set(hashes)) != 8:
        raise ValueError("duplicate exact source files rejected")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="motiondecode-ball-control-train8-native-fk-v1",
            source_head=a.expected_head,
            model_hash=a.model_hash,
            selected=[
                dict(path=path.relative_to(root).as_posix(), hash=h)
                for path, h in zip(selected, hashes, strict=True)
            ],
            selection="eight lexical-order train clips at declared offset; no outcomes",
            train_offset=a.train_offset,
            source_fps_assumption=120.0,
            target_fps=50.0,
            clip_source_independence_unproven=True,
            ball_contact_truth=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )

    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.motiondecode_native_reference import native_motion_reference
    from rosclaw_soccer.rsi.native_course_tracking import NativeCourseTracking
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

    model = mujoco.MjModel.from_binary_path(str(a.model))
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    plans, bodies, records = [], [], []
    for clip in selected:
        with clip.open(encoding="utf-8", newline="") as f:
            if tuple(next(csv.reader(f))) != _EXPECTED_COLUMNS:
                raise ValueError("exact source CSV schema required")
        values = np.loadtxt(clip, delimiter=",", skiprows=1, dtype=np.float64)
        plan, body = native_motion_reference(model=model, values=values, source_fps=120.0)
        if len(plan["joint"]) < 300:
            raise ValueError("complete long reference for every declared clip required")
        plans.append(plan)
        bodies.append(body)
        records.append(
            dict(
                path=clip.relative_to(root).as_posix(),
                source_hash=digest(clip),
                source_frames=len(values),
                reference_frames=len(plan["joint"]),
            )
        )
    tracker = NativeCourseTracking(model, layout, bodies)
    data = mujoco.MjData(model)
    min_score, guards = 1.0, 0
    for source, plan in enumerate(plans):
        for frame in range(len(plan["joint"])):
            data.qpos[:] = plan["native_qpos"][frame]
            data.qvel[:] = plan["native_qvel"][frame]
            mujoco.mj_forward(model, data)
            score, guard = tracker.measure(data, source=source, frame=frame, origin=np.zeros(3))
            min_score = min(min_score, float(score.min()))
            guards += int(guard.sum())
    if (
        min_score < 0.99999
        or guards
        or [digest(path) for path in selected] != hashes
        or digest(a.model) != a.model_hash
    ):
        raise ValueError("all-source self-consistency or input hashes failed")
    for i, (plan, body) in enumerate(zip(plans, bodies, strict=True)):
        path = a.output / f"reference-{i:02d}.npz"
        np.savez_compressed(
            path,
            **{"plan_" + k: v for k, v in plan.items()},
            **{"body_" + k: v for k, v in body.items()},
        )
        records[i]["reference_hash"] = digest(path)
    if subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("source changed during conversion")
    write_once(
        a.output / "complete.json",
        dict(
            status="MOTIONDECODE_TRAIN8_KINEMATIC_FK_NOT_TRACKED_SKILL",
            source_head=a.expected_head,
            model_hash=a.model_hash,
            records=records,
            train_offset=a.train_offset,
            source_fps_assumption=120.0,
            min_self_tracking=min_score,
            guard_count=guards,
            attribution="ChingMu MotionDecode",
            license_hash=digest(root / "LICENSE.md"),
            source_independence_unproven=True,
            ball_contact_truth=False,
            simulator_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
