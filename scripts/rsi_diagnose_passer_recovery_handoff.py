"""Inspect actual no-handoff cases from the audited two-player recovery study."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiving_handoff_diagnosis import diagnose_no_handoff
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
        raise ValueError("unused external evidence output required")
    source, _, source_hash = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_PASSER_VELOCITY_RECOVERY16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    if source_hash != a.source_hash:
        raise ValueError("exact independently replayed source required")
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
        raise ValueError("exact pinned shared native scene required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for r in source["records"]:
        if r["course"]["arm"]["passer_velocity_recovery"] or r["handoff"] is not None:
            continue
        if r["termination"] != "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300":
            raise ValueError("only fully observed original-horizon no-handoff cases supported")
        path = bind(root, r, "physics")
        if path is None:
            raise ValueError("exact independently replayed physics required")
        with np.load(path, allow_pickle=False) as z:
            result = diagnose_no_handoff(
                z["qpos"],
                z["qvel"],
                z["ball_forces"],
                z["body_failed"],
                receiver_q=layout.rq,
                receiver_v=layout.rv,
                ball_q=layout.bq,
                ball_v=layout.bv,
                excluded_contact=bool(r["result"]["excluded_contact_bodies"]),
            )
        if not result["no_handoff_reconstruction_consistent"]:
            raise ValueError("measured trace disagrees with original handoff contract")
        rows.append(
            dict(
                context=r["course"]["context"],
                side=r["course"]["lateral_m"],
                trial=r["index"],
                physics_hash=digest(path),
                **result,
            )
        )
    if len(rows) != 6 or len({(r["context"], r["side"]) for r in rows}) != 6:
        raise ValueError("all six uniquely declared development failures required")
    if digest(root / "complete.json") != a.source_hash or digest(audit) != a.audit_hash:
        raise ValueError("source changed during diagnosis")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PASSER_RECOVERY_NO_HANDOFF_PHYSICAL_DIAGNOSIS_NOT_INTERVENTION",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            layout=dict(
                receiver_q=layout.rq, receiver_v=layout.rv, ball_q=layout.bq, ball_v=layout.bv
            ),
            summary=dict(
                conditions=len(rows),
                receiver_contact=sum(r["actual_receiver_contact_over_1n"] for r in rows),
                no_receiver_contact=sum(not r["actual_receiver_contact_over_1n"] for r in rows),
            ),
            rows=rows,
            mutually_exclusive_causes=False,
            causal_intervention_performed=False,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(
        json.dumps(
            {
                "conditions": len(rows),
                "contacts": sum(r["actual_receiver_contact_over_1n"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
