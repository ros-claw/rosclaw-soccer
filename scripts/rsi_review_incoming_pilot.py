"""Measure actual saved incoming PPO trajectories; no physics or selection."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("pilot-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if args.output.exists() or subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("unused output and frozen reviewer required")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("exact reviewer head required")
    path = args.pilot_root / "complete.json"
    if digest(path) != args.pilot_hash:
        raise ValueError("exact completed pilot receipt required")
    report = json.loads(path.read_text())
    if report.get("status") != "INCOMING_PPO_PILOT_REQUIRES_INDEPENDENT_EXAM":
        raise ValueError("experimental pilot required")
    declaration_path = args.pilot_root / "declaration.json"
    declaration = json.loads(declaration_path.read_text())
    if declaration.get("source_head") != report.get("source_head"):
        raise ValueError("pilot source mismatch")
    pins = dict(declaration["input_file_hashes"])
    model_paths = [
        Path(k)
        for k in pins
        if k.endswith("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
    ]
    if len(model_paths) != 1:
        raise ValueError("one original compiled model required")
    for label in ("parent", "candidate"):
        pins[str(args.pilot_root / f"{label}-development.npz")] = report["development"][label][
            "raw_hash"
        ]
    pins[str(args.pilot_root / "candidate.pt")] = report["checkpoint_hash"]
    pins[str(path)] = args.pilot_hash
    check_pins(pins)

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_binary_path(str(model_paths[0]))
    ball_body = model.body("ball/ball").id
    balls = [i for i in range(model.ngeom) if model.geom_bodyid[i] == ball_body]
    feet = [model.body("robot/" + side + "_ankle_roll_link").id for side in ("left", "right")]
    geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] in feet and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    if len(balls) != 1 or not geoms:
        raise ValueError("actual ball and foot collision geometry required")
    worlds = {}
    arrays = {}
    for label in ("parent", "candidate"):
        with np.load(args.pilot_root / f"{label}-development.npz", allow_pickle=False) as data:
            arrays[label] = {k: data[k].copy() for k in data.files}
    a, b = arrays["parent"], arrays["candidate"]
    if not np.array_equal(a["bank_index"], b["bank_index"]) or not np.array_equal(
        a["observation_before"][0], b["observation_before"][0]
    ):
        raise ValueError("same actual initial paired states required")
    for label, raw in arrays.items():
        rows = []
        for world in range(len(raw["bank_index"])):
            data = mujoco.MjData(model)
            closest, closest_step = float("inf"), -1
            valid = raw["valid"][:, world]
            samples = raw["qpos"][:, world].reshape(-1, model.nq)
            for step in range(int(valid.sum()) * 10):
                data.qpos[:] = samples[step]
                mujoco.mj_forward(model, data)
                gap = min(
                    float(mujoco.mj_geomDistance(model, data, balls[0], geom, 10.0, None))
                    for geom in geoms
                )
                if gap < closest:
                    closest, closest_step = gap, step
            forces = raw["forces"][:, world].reshape(-1, 3)[: int(valid.sum()) * 10]

            def first(mask):
                found = np.flatnonzero(mask)
                return int(found[0]) if found.size else None

            rows.append(
                dict(
                    bank_index=int(raw["bank_index"][world]),
                    closest_foot_surface_gap_m=closest,
                    closest_native_step=closest_step,
                    first_foot_gt_10n=first(np.max(forces[:, :2], axis=1) > 10),
                    first_nonfoot_gt_1n=first(forces[:, 2] > 1),
                    healthy=not bool(raw["terminated"][:, world].any()),
                    capture_rewarded=bool(raw["credit_sustained_capture"][:, world].any()),
                    observed_controls=int(valid.sum()),
                )
            )
        worlds[label] = rows
    check_pins(pins)
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="SAVED_TRAJECTORY_GEOMETRY_REVIEW_NOT_REPLAY_OR_SKILL",
            source_head=args.expected_head,
            pilot_hash=args.pilot_hash,
            paired_initial_observation_exact=True,
            input_hashes=pins,
            worlds=worlds,
            no_new_physics=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
