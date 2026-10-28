"""Build one pinned, noncommercial, explicitly transformed tracking reference."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.cftrack_motion_reference import load_cftrack_motion
from rosclaw_soccer.rsi.omnicontact_motion_retarget import retarget_window
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "geometry", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("geometry-hash", "capture", "foot", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--event", type=int, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.geometry.is_symlink()
        or digest(a.geometry) != a.geometry_hash
        or a.foot not in ("left", "right")
    ):
        raise ValueError("exact geometry receipt and unused external output required")
    geometry = json.loads(a.geometry.read_text())
    if (
        geometry.get("status") != "OMNICONTACT_REFERENCE_FOOT_GEOMETRY_NOT_PHYSICS"
        or geometry.get("promotion_authorized") is not False
        or geometry.get("physics_steps") != 0
    ):
        raise ValueError("reference-only training geometry required")
    candidates = [
        r
        for r in geometry["samples"]
        if r["capture"] == a.capture and r["event_frame"] == a.event and r["foot"] == a.foot
    ]
    if len(candidates) != 1:
        raise ValueError("exact one training motion event required")
    row = candidates[0]
    root = a.dataset_root.resolve()
    source = root / row["source_relative_path"]
    if (
        source.is_symlink()
        or not source.resolve().is_relative_to(root / "npz" / "soccer")
        or digest(source) != row["source_hash"]
    ):
        raise ValueError("unchanged screened training reference required")
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    fps = int(data["fps"][0])
    if fps != row["fps"]:
        raise ValueError("screened source sample rate changed")
    transformed = retarget_window(data, event=a.event, source_fps=fps, ball_lift_m=0.06)
    if digest(source) != row["source_hash"] or digest(a.geometry) != a.geometry_hash:
        raise ValueError("source or geometry changed during transformation")
    a.output.mkdir(parents=True)
    motion = a.output / "reference.npz"
    with motion.open("xb") as stream:
        np.savez_compressed(stream, **transformed)
    validated = load_cftrack_motion(motion)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="RETARGETED_TRAIN_RECEIVING_REFERENCE_NOT_PHYSICS",
            source_head=a.expected_head,
            geometry_hash=a.geometry_hash,
            source_hash=row["source_hash"],
            capture=a.capture,
            event=a.event,
            foot=a.foot,
            source_fps=fps,
            source_window_frames=[a.event - fps, a.event + fps],
            output_fps=50.0,
            output_frames=len(validated["joint_pos"]),
            ball_reference_lift_m=0.06,
            ball_z_min_m=float(validated["object_pos_w"][:, 2].min()),
            table_reference="zero placeholders for ball-only task",
            base_reference="body index zero",
            contact_resampling="nearest source frame",
            quaternion_resampling="shortest arc slerp",
            artifact_hash=digest(motion),
            reference_only_not_actual_state=True,
            physics_steps=0,
            noncommercial_source=True,
            commercial_publicity_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
