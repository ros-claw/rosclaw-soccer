"""Compare matched archived SIM return trials; no replay or promotion."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.relay_return_pair_comparison import compare_return_pair
from rosclaw_soccer.rsi.relay_snapshot_cases import select_relay_snapshot_cases
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
        raise ValueError("unused external report required")
    paths = (a.baseline / "complete.json", a.candidate / "complete.json")
    hashes = (a.baseline_hash, a.candidate_hash)
    if any(x.is_symlink() or digest(x) != h for x, h in zip(paths, hashes, strict=True)):
        raise ValueError("exact paired trial receipts required")
    baseline, candidate = (json.loads(x.read_text()) for x in paths)
    expected = (
        "RELAY_NONDEFAULT_CAUSAL_RETURN_CHAIN_MATCH_NOT_FOOTBALL_SUCCESS",
        "RELAY_NONDEFAULT_SMOOTH_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS",
    )
    for evidence, status in zip((baseline, candidate), expected, strict=True):
        if (
            evidence.get("status") != status
            or evidence.get("source_hash") != a.source_hash
            or evidence.get("audit_hash") != a.audit_hash
            or any(
                evidence.get(k) is not False
                for k in (
                    "independent_fresh",
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                )
            )
            or len(evidence.get("rows", [])) != 24
        ):
            raise ValueError("exact unpromoted 24-case pair required")
    for key in (
        "snapshot_hash",
        "frontier_hash",
        "role_proof_hash",
        "expanded_velocity_proof_hash",
    ):
        if baseline.get(key) != candidate.get(key):
            raise ValueError("paired input lineage differs")
    root, audit = a.source / "source", a.source / "audit/complete.json"
    source, _, sh = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if sh != a.source_hash:
        raise ValueError("original audited source mismatch")
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
        raise ValueError("original audited scene required")
    scene_hash = digest(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    ball_q = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    ).bq
    source_rows = {
        r["index"]: r for r in select_relay_snapshot_cases(source["records"], nondefault=True)
    }
    b_rows = {r["index"]: r for r in baseline["rows"]}
    c_rows = {r["index"]: r for r in candidate["rows"]}
    if len(source_rows) != 24 or set(source_rows) != set(b_rows) or set(b_rows) != set(c_rows):
        raise ValueError("distinct matched original cases required")
    rows = []
    for index in sorted(source_rows):
        original, b, c = source_rows[index], b_rows[index], c_rows[index]
        for key in (
            "context",
            "side",
            "receiver_lateral_offset_m",
            "receiving_ready",
            "receiver_velocity_controls",
        ):
            if b[key] != c[key]:
                raise ValueError("paired boundary mismatch")
        handoff = original["handoff"]
        if bool(handoff) != b["receiving_ready"]:
            raise ValueError("source causal readiness mismatch")
        arrays = []
        for directory, row in ((a.baseline, b), (a.candidate, c)):
            path = directory / row["physical_trajectory"]
            if (
                path.name != row["physical_trajectory"]
                or path.is_symlink()
                or digest(path) != row["physical_trajectory_hash"]
            ):
                raise ValueError("exact safe physical trajectory required")
            with np.load(path, allow_pickle=False) as data:
                arrays.append({k: data[k] for k in ("qpos", "qvel", "torque", "ball_forces")})
            if digest(path) != row["physical_trajectory_hash"]:
                raise ValueError("trajectory changed during read")
        goal = np.asarray(handoff["return_goal"] if handoff else [0.0, 0.0, 0.0])
        metrics = compare_return_pair(
            *arrays,
            prefix_controls=b["receiver_velocity_controls"],
            ball_qpos=ball_q,
            return_goal=goal,
        )
        rows.append(
            dict(
                index=index,
                receiving_ready=bool(handoff),
                baseline_body_failure=b["body_failure"],
                candidate_body_failure=c["body_failure"],
                **metrics,
            )
        )
    if (
        digest(paths[0]) != hashes[0]
        or digest(paths[1]) != hashes[1]
        or digest(root / "complete.json") != sh
        or digest(audit) != a.audit_hash
        or digest(scene) != scene_hash
    ):
        raise ValueError("paired evidence changed")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="PAIRED_RETURN_SMOOTHING_DEVELOPMENT_DIAGNOSIS_NOT_INDEPENDENT_REPLAY",
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=a.audit_hash,
            baseline_hash=hashes[0],
            candidate_hash=hashes[1],
            scene_hash=scene_hash,
            rows=rows,
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
