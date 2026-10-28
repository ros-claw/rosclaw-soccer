"""Build noncommercial tracker reference from a separately projected FK fit."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.cftrack_motion_reference import load_cftrack_motion
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.reference_leg_reconstruction import (
    reconstruct_right_leg,
    shifted_reference_frames,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def raw_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("scene", "reference", "fit", "projection", "source-receipt", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "scene-sha256",
        "reference-sha256",
        "fit-sha256",
        "projection-sha256",
        "source-receipt-sha256",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--frame-shift", type=int, default=0)
    a = p.parse_args()
    clean(a.expected_head)
    paths = (
        (a.scene, a.scene_sha256),
        (a.reference, a.reference_sha256),
        (a.fit, a.fit_sha256),
        (a.projection, a.projection_sha256),
        (a.source_receipt, a.source_receipt_sha256),
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or raw_hash(path) != expected for path, expected in paths)
    ):
        raise ValueError("unchanged pinned inputs and unused external output required")
    fit = json.loads(a.fit.read_text())
    projection = json.loads(a.projection.read_text())
    source_receipt = json.loads(a.source_receipt.read_text())
    if (
        fit["foot"] != "right"
        or projection["status"] != "SAMPLED_FK_GATES_PASS"
        or projection["input_hashes"] != [a.scene_sha256, a.reference_sha256, a.fit_sha256]
        or source_receipt["artifact_hash"] != "sha256:" + a.reference_sha256
        or source_receipt["noncommercial_source"] is not True
    ):
        raise ValueError("exact projected right-leg train reference lineage required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    motion = load_cftrack_motion(a.reference)
    frames = shifted_reference_frames(
        np.asarray(fit["frames"], dtype=int), a.frame_shift, len(motion["joint_pos"])
    )
    result, checks = reconstruct_right_leg(
        model,
        layout,
        motion,
        frames,
        np.asarray(projection["offsets_rad"], dtype=np.float64),
    )
    # CFtrack cold-start state includes frame-zero qpos, qvel, root, and ball.
    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "object_pos_w",
        "object_quat_w",
        "object_lin_vel_w",
        "object_ang_vel_w",
    ):
        if not np.array_equal(result[key][0], motion[key][0]):
            raise ValueError("frame-zero reference changed: " + key)
    if any(raw_hash(path) != expected for path, expected in paths):
        raise ValueError("inputs changed during reconstruction")
    a.output.mkdir(parents=True)
    artifact = a.output / "reference.npz"
    with artifact.open("xb") as stream:
        np.savez_compressed(stream, **result)
    load_cftrack_motion(artifact)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="RETARGETED_TRAIN_RECEIVING_REFERENCE_NOT_PHYSICS",
            artifact_hash=digest(artifact),
            source_head=a.expected_head,
            source_reference_hash="sha256:" + a.reference_sha256,
            source_receipt_hash="sha256:" + a.source_receipt_sha256,
            projected_fit_hash="sha256:" + a.projection_sha256,
            fit_hash="sha256:" + a.fit_sha256,
            scene_hash="sha256:" + a.scene_sha256,
            ball_reference_lift_m=source_receipt["ball_reference_lift_m"],
            noncommercial_source=True,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
            physics_steps=0,
            reference_only_not_actual_state=True,
            partial_retarget=True,
            reference_correction_frame_shift=a.frame_shift,
            shifted_reference_geometry_checked=a.frame_shift == 0,
            unchanged_unverified_virtual_markers=True,
            frame_zero_all_tracker_initial_fields_bit_exact=True,
            source_optimizer_converged=False,
            reconstruction_checks=checks,
            goal_complete=False,
            warning="not physics or training; unvalidated ee/contact arrays preserved from source",
        ),
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
