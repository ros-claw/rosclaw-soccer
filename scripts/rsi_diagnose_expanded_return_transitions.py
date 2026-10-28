"""Inspect actual handoff discontinuities in 24 archived development cases."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiver_return_transition_diagnostics import return_transition_diagnostics
from rosclaw_soccer.rsi.relay_snapshot_cases import select_relay_snapshot_cases
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    root, audit = a.source / "source", a.source / "audit/complete.json"
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external report required")
    source, _, sh = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if sh != a.source_hash:
        raise ValueError("exact audited development source required")
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
        raise ValueError("exact audited shared native scene required")
    scene_hash = digest(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    cases = select_relay_snapshot_cases(source["records"], nondefault=True)
    rows = []
    for row in cases:
        if row["handoff"] is None:
            rows.append(dict(index=row["index"], handoff=False, termination=row["termination"]))
            continue
        path = bind(root, row, "physics")
        if path is None:
            raise ValueError("original actual physical trajectory required")
        with np.load(path, allow_pickle=False) as data:
            measured = return_transition_diagnostics(
                data["target_b"],
                data["qpos"],
                data["body_failed"],
                np.asarray(row["handoff"]["return_goal"], np.float64),
                handoff_control=row["handoff"]["control"],
                ball_q=layout.bq,
            )
        if (
            digest(path) != row["artifacts"]["physics"]["hash"]
            or abs(
                measured["minimum_return_goal_error_m"]
                - row["result"]["minimum_return_goal_error_3d_m"]
            )
            > 1e-10
        ):
            raise ValueError("unchanged original trajectory and recorded result required")
        rows.append(
            dict(index=row["index"], handoff=True, termination=row["termination"], **measured)
        )
    if (
        digest(root / "complete.json") != a.source_hash
        or digest(audit) != a.audit_hash
        or digest(scene) != scene_hash
    ):
        raise ValueError("source evidence changed during diagnosis")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="EXPANDED_RETURN_TRANSITION_DIAGNOSIS_NOT_CAUSAL",
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=a.audit_hash,
            scene_hash=scene_hash,
            rows=rows,
            independent_fresh=False,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
