"""Reconstruct selected recorded contact peaks, not a future skill rollout."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics import contact_wrench
from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.providers.g1 import joint_contract
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim import contracts
from rosclaw_soccer.sim.contracts import hash_json


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("unused localization output required")
    source_pins = {
        str(path): digest(path)
        for path in (
            Path(__file__),
            Path(contact_wrench.__file__),
            Path(joint_contract.__file__),
            Path(contracts.__file__),
        )
    }
    original = json.loads(args.branch_report.read_text())
    if (
        original["report_sha256"]
        != hashlib.sha256(
            json.dumps(
                {k: v for k, v in original.items() if k != "report_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("whole original branch report seal required")
    for name, expected in original["pins"].items():
        if digest(Path(name)) != expected:
            raise ValueError("original branch source/input changed")
    paths = list(map(Path, original["pins"]))
    snapshots = [p for p in paths if p.suffix == ".mjb"]
    traces = [p for p in paths if p.suffix == ".npz"]
    if len(snapshots) != 2 or len(traces) != 1:
        raise ValueError("exact two original worlds and one original trace required")
    models = [mujoco.MjModel.from_binary_path(str(path)) for path in snapshots]
    by_world = {
        "old": next(m for m in models if m.npair == 0),
        "mesh": next(m for m in models if m.npair == 2),
    }
    with np.load(traces[0], allow_pickle=False) as archive:
        trace = {k: archive[k].copy() for k in ("canonical_qpos", "canonical_qvel", "torque_nm")}
    groups = sorted({(e["world"], e["other_body"]) for e in original["events"]})
    peaks = [
        max(
            (e for e in original["events"] if (e["world"], e["other_body"]) == key),
            key=lambda e: np.linalg.norm(e["force_world_n"]),
        )
        for key in groups
    ]
    results = []
    steps = 0
    for peak in peaks:
        model = by_world[peak["world"]]
        data = mujoco.MjData(model)
        frame, substep = peak["frame"], peak["substep"]
        data.qpos[:] = trace["canonical_qpos"][frame, 0]
        data.qvel[:] = trace["canonical_qvel"][frame, 0]
        data.time = frame * 0.02
        mujoco.mj_forward(model, data)
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        ids = [np.flatnonzero(model.actuator_trnid[:, 0] == j) for j in joints]
        if any(len(a) != 1 for a in ids):
            raise ValueError("named canonical actuator mapping required")
        actuators = [int(a[0]) for a in ids]
        for i in range(substep + 1):
            data.ctrl[actuators] = trace["torque_nm"][frame, 0, i]
            mujoco.mj_step(model, data)
            steps += 1
        contacts = read_geom_contact_wrenches(model, data, geom_id=int(model.geom("ball_geom").id))
        matches = [
            c
            for c in contacts
            if c.contact_id == peak["contact_id"] and c.other_geom_id == peak["other_geom_id"]
        ]
        if len(matches) != 1 or dataclasses.asdict(matches[0]) != {
            k: tuple(peak[k]) if type(peak[k]) is list else peak[k]
            for k in dataclasses.asdict(matches[0])
        }:
            raise ValueError("selected peak must exactly reproduce actual original native contact")
        side = "left" if peak["other_body"].startswith("left_") else "right"
        foot = model.body(f"{side}_ankle_roll_link").id
        shin = model.body(f"{side}_knee_link").id
        point = np.asarray(peak["point_world_m"])
        ball = data.xpos[model.body("ball").id]
        rows = {}
        for label, body in (("foot", foot), ("shin", shin)):
            rotation = data.xmat[body].reshape(3, 3)
            rows[label] = dict(
                body_name=model.body(body).name,
                body_origin_world_m=data.xpos[body].tolist(),
                ball_center_body_m=(rotation.T @ (ball - data.xpos[body])).tolist(),
                contact_point_body_m=(rotation.T @ (point - data.xpos[body])).tolist(),
                contact_force_body_n=(rotation.T @ np.asarray(peak["force_world_n"])).tolist(),
            )
        results.append(
            dict(
                world=peak["world"],
                other_body=peak["other_body"],
                frame=frame,
                substep=substep,
                normal_force_n=peak["contact_normal_force_n"],
                world_force_norm_n=float(np.linalg.norm(peak["force_world_n"])),
                cached_contact_time_s=float(data.time),
                localization=rows,
            )
        )
    for name, expected in original["pins"].items():
        if digest(Path(name)) != expected:
            raise ValueError("original branch input changed during localization")
    if any(digest(Path(path)) != expected for path, expected in source_pins.items()):
        raise ValueError("localization source changed during bounded diagnostic")
    result = dict(
        schema="soccer.physics.cached_contact_peak_localization.v1",
        source_branch_report_hash=original["report_sha256"],
        source_file_hash=digest(args.branch_report),
        runner_source_hash=digest(Path(__file__)),
        source_pins=source_pins,
        rows=results,
        new_physics_substeps=steps,
        cached_xpos_and_contact_same_assembly=True,
        qpos_after_step_not_used_for_local_transform=True,
        independent_courses=0,
        skill_rollouts=0,
        optimizer_updates=0,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
