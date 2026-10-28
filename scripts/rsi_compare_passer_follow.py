"""Compare the contact-triggered follow teacher with its matched SIM baseline."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.relay_follow_prefix_audit import verify_follow_prefix
from rosclaw_soccer.rsi.relay_return_pair_comparison import compare_return_pair
from rosclaw_soccer.rsi.relay_snapshot_cases import select_relay_snapshot_cases
from rosclaw_soccer.rsi.relay_teammate_successor import teammate_successor
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "baseline", "candidate", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "baseline-hash", "candidate-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external result required")
    paths = [a.baseline / "complete.json", a.candidate / "complete.json"]
    hashes = [a.baseline_hash, a.candidate_hash]
    if any(p.is_symlink() or digest(p) != h for p, h in zip(paths, hashes, strict=True)):
        raise ValueError("bound trial receipts required")
    base, trial = [json.loads(p.read_text()) for p in paths]
    statuses = (
        "RELAY_NONDEFAULT_SMOOTH_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS",
        "RELAY_PASSER_FOLLOW_TEACHER_PHYSICAL_CANDIDATE_NOT_SUCCESS",
    )
    for data, status in zip((base, trial), statuses, strict=True):
        if (
            data.get("status") != status
            or data.get("source_hash") != a.source_hash
            or data.get("audit_hash") != a.audit_hash
            or data.get("smooth_transition_duration") != 10
            or len(data.get("rows", [])) != 24
            or any(
                data.get(k) is not False
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                    "independent_fresh",
                )
            )
        ):
            raise ValueError("matching unpromoted 10-control development receipts required")
    if (
        trial.get("passer_follow_return") is not True
        or trial.get("passer_follow_trigger") != "sim_contact_oracle"
    ):
        raise ValueError("explicit privileged teacher trigger required")
    for key in (
        "snapshot_hash",
        "frontier_hash",
        "role_proof_hash",
        "expanded_velocity_proof_hash",
    ):
        if base.get(key) != trial.get(key):
            raise ValueError("matching physical input lineage required")
    root, audit = a.source / "source", a.source / "audit/complete.json"
    source, _, sh = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if sh != a.source_hash:
        raise ValueError("source evidence mismatch")
    scene = root / "scene.xml"
    xmls = [
        Path(x)
        for x in source["input_file_hashes"]
        if x.endswith("/cached-standard/declared-scene.xml")
    ]
    if (
        len(xmls) != 1
        or digest(xmls[0]) != source["input_file_hashes"][str(xmls[0])]
        or scene.is_symlink()
        or scene.read_text() != shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
    ):
        raise ValueError("exact native scene required")
    scene_hash = digest(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    teammate = NativeVelocityLayout(
        model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    ball_q = teammate.bq
    original = {
        r["index"]: r for r in select_relay_snapshot_cases(source["records"], nondefault=True)
    }
    b_rows = {r["index"]: r for r in base["rows"]}
    c_rows = {r["index"]: r for r in trial["rows"]}
    if len(original) != 24 or set(original) != set(b_rows) or set(b_rows) != set(c_rows):
        raise ValueError("24 distinct matched cases required")
    rows = []
    for index in sorted(original):
        br, cr = b_rows[index], c_rows[index]
        for key in (
            "context",
            "side",
            "receiver_lateral_offset_m",
            "receiving_ready",
            "receiver_velocity_controls",
        ):
            if br[key] != cr[key]:
                raise ValueError("matched causal receiving boundary required")
        arrays = []
        for directory, row in ((a.baseline, br), (a.candidate, cr)):
            name = row["physical_trajectory"]
            path = directory / name
            if (
                Path(name).name != name
                or path.is_symlink()
                or digest(path) != row["physical_trajectory_hash"]
            ):
                raise ValueError("unchanged trajectory required")
            with np.load(path, allow_pickle=False) as data:
                arrays.append({k: data[k] for k in data.files})
            if digest(path) != row["physical_trajectory_hash"]:
                raise ValueError("trajectory changed during analysis")
        prefix = verify_follow_prefix(
            *arrays,
            initial_control=140,
            receiver_velocity_controls=br["receiver_velocity_controls"],
        )
        handoff = original[index]["handoff"]
        if bool(handoff) != br["receiving_ready"]:
            raise ValueError("original handoff mismatch")
        goal = np.asarray(handoff["return_goal"] if handoff else [0.0, 0.0, 0.0])
        measured = compare_return_pair(
            *arrays,
            prefix_controls=br["receiver_velocity_controls"],
            ball_qpos=ball_q,
            return_goal=goal,
        )
        successors = [
            teammate_successor(
                data["qpos"],
                data["qvel"],
                data["ball_forces"],
                return_start=br["receiver_velocity_controls"],
                ball_q=teammate.bq,
                ball_v=teammate.bv,
                teammate_q=teammate.rq,
                teammate_v=teammate.rv,
                timestep=float(model.opt.timestep),
            )
            for data in arrays
        ]
        rows.append(
            dict(
                index=index,
                receiving_ready=bool(handoff),
                baseline_body_failure=br["body_failure"],
                candidate_body_failure=cr["body_failure"],
                baseline_teammate_successor=successors[0],
                candidate_teammate_successor=successors[1],
                **prefix,
                **measured,
            )
        )
    if (
        any(digest(p) != h for p, h in zip(paths, hashes, strict=True))
        or digest(scene) != scene_hash
        or digest(root / "complete.json") != sh
        or digest(audit) != a.audit_hash
    ):
        raise ValueError("input evidence changed during analysis")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="PASSER_FOLLOW_TEACHER_PAIRED_DEVELOPMENT_NOT_INDEPENDENT_REPLAY",
            source_head=a.expected_head,
            source_hash=sh,
            baseline_hash=a.baseline_hash,
            candidate_hash=a.candidate_hash,
            scene_hash=scene_hash,
            rows=rows,
            sim_contact_oracle_used=True,
            independent_fresh=False,
            independent_replay=False,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
