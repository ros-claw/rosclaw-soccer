"""Review matched target-filter guard events from saved native states.

This performs forward kinematics only: no new physics, no policy replay, and
no inference that a reference deviation equals a fall or stable recovery.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline-root", "candidate-root", "reference-root", "model", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("baseline-hash", "candidate-hash", "reference-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if a.output.exists() or subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("unused output and frozen reviewer required")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != a.expected_head:
        raise ValueError("exact reviewer source required")
    paths = [
        root / "complete.json" for root in (a.baseline_root, a.candidate_root, a.reference_root)
    ]
    hashes = [a.baseline_hash, a.candidate_hash, a.reference_hash]
    if any(digest(path) != h for path, h in zip(paths, hashes, strict=True)):
        raise ValueError("exact source receipts required")
    baseline, candidate, reference = [json.loads(path.read_text()) for path in paths]
    if (
        baseline.get("status") != "MOTIONDECODE_INCOMING_SINGLE_ACTOR_PROBE_NOT_SKILL"
        or candidate.get("status") != baseline["status"]
        or candidate.get("target_rate_filter") is not True
        or baseline.get("bank_hash") != candidate.get("bank_hash")
        or baseline.get("reference_hash") != a.reference_hash
        or candidate.get("reference_hash") != a.reference_hash
        or reference.get("status") != "MOTIONDECODE_TRAIN8_KINEMATIC_FK_NOT_TRACKED_SKILL"
        or len(reference["records"]) != 8
    ):
        raise ValueError("matched physical study and exact reference bank required")
    pins = {
        str(a.model): reference["model_hash"],
        str(a.baseline_root / "physical.npz"): baseline["raw_hash"],
        str(a.candidate_root / "physical.npz"): candidate["raw_hash"],
        str(a.baseline_root / "incoming-bank.npz"): baseline["bank_hash"],
        str(a.candidate_root / "incoming-bank.npz"): candidate["bank_hash"],
        **{str(path): h for path, h in zip(paths, hashes, strict=True)},
    }
    for i, record in enumerate(reference["records"]):
        pins[str(a.reference_root / f"reference-{i:02d}.npz")] = record["reference_hash"]
    check_pins(pins)

    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.native_course_tracking import TRACKING_BODIES, NativeCourseTracking
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

    model = mujoco.MjModel.from_binary_path(str(a.model))
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    with np.load(a.baseline_root / "physical.npz", allow_pickle=False) as z:
        raw_a = {k: z[k].copy() for k in z.files}
    with np.load(a.candidate_root / "physical.npz", allow_pickle=False) as z:
        raw_b = {k: z[k].copy() for k in z.files}
    with np.load(a.baseline_root / "incoming-bank.npz", allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in z.files}
    if not np.array_equal(raw_a["bank_index"], raw_b["bank_index"]) or not np.array_equal(
        raw_a["observation_before"][0], raw_b["observation_before"][0]
    ):
        raise ValueError("exact paired reset and initial actor observation required")
    refs = []
    for i in range(8):
        with np.load(a.reference_root / f"reference-{i:02d}.npz", allow_pickle=False) as z:
            refs.append(
                {k.removeprefix("body_"): z[k].copy() for k in z.files if k.startswith("body_")}
            )
    tracker = NativeCourseTracking(model, layout, refs)
    names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    ee_ref_ids = [TRACKING_BODIES.index(name) for name in names]
    ee_ids = [model.body("robot/" + name).id for name in names]
    torso = model.body("robot/torso_link").id
    data = mujoco.MjData(model)

    def measure(raw, world, control):
        if control >= len(raw["valid"]) or not raw["valid"][control, world]:
            return None
        data.qpos[:] = raw["qpos"][control, world, -1]
        data.qvel[:] = raw["qvel"][control, world, -1]
        mujoco.mj_forward(model, data)
        index = int(raw["bank_index"][world])
        source = int(bank["source_id"][index])
        frame = int(raw["frame_before"][control, world])
        origin = bank["origin"][index]
        tracking, guards = tracker.measure(data, source=source, frame=frame, origin=origin)
        if not np.array_equal(guards, raw["guards"][control, world]):
            raise ValueError("reconstructed actual guard differs from saved value")
        root_matrix = data.xmat[layout.pelvis].reshape(3, 3)
        ref = refs[source]["position"][frame] + origin
        return dict(
            control=control,
            root_height_m=float(data.xpos[layout.pelvis, 2]),
            root_tilt_degrees=float(np.rad2deg(np.arccos(np.clip(root_matrix[2, 2], -1, 1)))),
            root_speed_mps=float(np.linalg.norm(data.qvel[layout.rv : layout.rv + 3])),
            torso_z_error_m=float(
                data.xpos[torso, 2] - ref[TRACKING_BODIES.index("torso_link"), 2]
            ),
            end_effector_z_errors_m={
                name: float(data.xpos[body, 2] - ref[ref_id, 2])
                for name, body, ref_id in zip(names, ee_ids, ee_ref_ids, strict=True)
            },
            guards=guards.tolist(),
            tracking=tracking.tolist(),
        )

    worlds = []
    for world in range(len(raw_a["bank_index"])):
        candidate_terminal = np.flatnonzero(raw_b["terminated"][:, world])
        baseline_terminal = np.flatnonzero(raw_a["terminated"][:, world])
        t = int(candidate_terminal[0]) if len(candidate_terminal) else None
        worlds.append(
            dict(
                world=world,
                bank_index=int(raw_a["bank_index"][world]),
                baseline_terminal=None if not len(baseline_terminal) else int(baseline_terminal[0]),
                candidate_terminal=t,
                baseline_at_candidate_terminal=None if t is None else measure(raw_a, world, t),
                candidate_at_terminal=None if t is None else measure(raw_b, world, t),
            )
        )
    check_pins(pins)
    if subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("reviewer source changed")
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="TARGET_FILTER_GUARD_SAVED_FK_REVIEW_NOT_REPLAY",
            source_head=a.expected_head,
            input_hashes=pins,
            paired_initial_observation_exact=True,
            worlds=worlds,
            simulator_steps=0,
            no_claim_of_fall_or_recovery=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
