"""Find TRAIN next-skill opportunities without changing strict capture grades.

Only a complete independently replayed32-world source is accepted. This does
not advance physics, fit a policy, activate a skill or prove a return pass.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "audit-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("audit-hash", "audit-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    sp, ap, dp = (
        a.source_root / "complete.json",
        a.audit_root / "complete.json",
        a.source_root / "declaration.json",
    )
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (sp, ap, dp))
        or any((p / "failed.json").exists() for p in (a.source_root, a.audit_root))
        or digest(ap) != a.audit_hash
    ):
        raise ValueError("unused output and exact complete source/own audit required")
    source, audit, decl = [json.loads(p.read_text()) for p in (sp, ap, dp)]
    if (
        source["status"] != "PREALIGNED_FOOT_CAPTURE_TEACHER32_COLLECTED_PENDING_OWN_REPLAY"
        or audit["status"] != "PREALIGNED_FOOT_CAPTURE32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
        or audit["source_head"] != a.audit_head
        or audit["producer_hash"] != digest(sp)
        or source["declaration_hash"] != digest(dp)
        or len(source["records"]) != 32
        or len(audit["records"]) != 32
    ):
        raise ValueError("all32 actual own-neural/PD/native/contact trajectories required")
    for row, own in zip(source["records"], audit["records"], strict=True):
        path = a.source_root / row["artifact"]["path"]
        if (
            row["course"] != own["course"]
            or row["index"] != own["index"]
            or row["qualified_capture"] != own["qualified_capture"]
            or path.parent != a.source_root
            or path.is_symlink()
            or digest(path) != row["artifact"]["hash"]
        ):
            raise ValueError("exact unchanged paired source courses and capture grades required")
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.receiving_handoff_opportunity import handoff_windows
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene

    xmls = [
        Path(p)
        for p in source["input_file_hashes"]
        if p.endswith("/cached-standard/declared-scene.xml")
    ]
    if not xmls or digest(xmls[0]) != source["input_file_hashes"][str(xmls[0])]:
        raise ValueError("pinned original standard body scene required")
    xml = shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
    if (a.source_root / "scene.xml").read_text() != xml:
        raise ValueError("exact original coupled scene required")
    m = mujoco.MjModel.from_xml_string(xml)
    robot, ball = int(m.body("b/pelvis").id), int(m.body("soccer_ball").id)
    rj, bj = int(m.body_jntadr[robot]), int(m.body_jntadr[ball])
    records = []
    for row, own in zip(source["records"], audit["records"], strict=True):
        with np.load(a.source_root / row["artifact"]["path"], allow_pickle=False) as z:
            windows, excluded = handoff_windows(
                qpos=z["qpos"],
                qvel=z["qvel"],
                body_failed=z["body_failed"],
                root_qposadr=int(m.jnt_qposadr[rj]),
                root_dofadr=int(m.jnt_dofadr[rj]),
                ball_qposadr=int(m.jnt_qposadr[bj]),
                ball_dofadr=int(m.jnt_dofadr[bj]),
                contacts=own["contact_body_witnesses"],
                prefix="b/",
            )
        records.append(
            dict(
                index=row["index"],
                course=row["course"],
                strict_foot_capture_unchanged=row["qualified_capture"],
                excluded_contact_bodies=excluded,
                candidate_windows=[asdict(w) for w in windows],
                actual_next_pass_verified=False,
            )
        )
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="TRAIN_RECEIVING_HANDOFF_OPPORTUNITIES_NOT_SKILL_SUCCESSES",
            source_head=a.expected_head,
            source_hash=digest(sp),
            own_audit_hash=a.audit_hash,
            records=records,
            summary={
                arm: sum(bool(r["candidate_windows"]) for r in records if r["course"]["arm"] == arm)
                for arm in sorted({r["course"]["arm"] for r in records})
            },
            declaration_hash=digest(dp),
            physical_steps=0,
            optimizer_updates=0,
            methods={
                str(p): digest(p)
                for p in (
                    Path(__file__),
                    Path("src/rosclaw_soccer/rsi/receiving_handoff_opportunity.py"),
                )
            },
            scope=(
                "Exploratory TRAIN next-skill starts only; strict clean-foot failures remain failed. "
                "No return pass, full skill, IFAB referee or promotion claim."
            ),
            research_sources=[
                "https://www.theifab.com/laws/latest/fouls-and-misconduct/",
                "https://www.fifatrainingcentre.com/en/practice/elite-sessions/in-possession/john-peacock-passing-and-receiving.php",
            ],
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
