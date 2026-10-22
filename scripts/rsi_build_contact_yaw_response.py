"""Declare foot-yaw experiments from exact native contact identity/geometry."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "geometry-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("geometry-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen contact-yaw declaration required")

    clean()
    gp, rp, dp = (
        a.geometry_root / "complete.json",
        a.root / "complete.json",
        a.root / "declaration.json",
    )
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (gp, rp, dp))
        or digest(gp) != a.geometry_hash
    ):
        raise ValueError("exact native geometry witness and unused output required")
    g, r, d = [json.loads(x.read_text()) for x in (gp, rp, dp)]
    if (
        g["status"] != "INHERITED_RUN_EXPERT_ALL_TRIAL_WHOLE_NATIVE_OWN_NN_IK_AUDIT_PASS"
        or g["source_hash"] != digest(rp)
        or g["worlds"] != 28
        or g["contact_events_captured"] is not True
        or g["counts"]
        != dict(
            native=r["actual_native_steps"],
            neural=r["actual_neural_calls"],
            kinematic_intent=r["actual_kinematic_intent_calls"],
        )
        or r["declaration_hash"] != digest(dp)
        or d["protocol"] != "paid.TRAIN.inherited_run_actual_state_expert.response_v3"
        or any(g[k] is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("whole28 world own NN and actual contact capture required")
    pins = dict(g["input_file_hashes"])
    for x in (gp, rp, dp):
        pins[str(x)] = digest(x)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    import numpy as np

    from rosclaw_soccer.rsi.contact_yaw_response import contact_yaw_proposals
    from rosclaw_soccer.rsi.run_contact_geometry import summarize_contacts
    from rosclaw_soccer.rsi.run_teacher_objective import run_teacher_rank

    contexts = []
    for ci in range(2):
        tag = f"selected-c{ci}-repeat0"
        row = next(x for x in r["records"] if x["tag"] == tag)
        art = next(x for x in g["contact_event_artifacts"] if x["tag"] == tag)
        path = a.geometry_root / art["path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(a.geometry_root.resolve())
            or digest(path) != art["hash"]
        ):
            raise ValueError("exact confined actual contact event history required")
        pins[str(path)] = art["hash"]
        blob = json.loads(path.read_text())
        goal = np.array(d["source_case"]["goal_world"], np.float64)
        summary = summarize_contacts(blob["events"], goal)
        if (
            summary != blob["summary"]
            or summary != art["summary"]
            or summary["nonfoot_body_names"]
            or not run_teacher_rank(row)[1]
        ):
            raise ValueError("stable useful no-nonfoot TRAIN base with exact geometry required")
        role = 0 if "left" in d["source_case"]["source"] else 1
        bodies = [v for v in summary["bodies"].values() if v["role"] == role]
        if len(bodies) != 1 or bodies[0]["radial_heading_degrees"] is None:
            raise ValueError("one qualified intended foot radial contact measurement required")
        first = row["result"]["postcontact"]["first_foot_contact_substep"]
        proposals, model = contact_yaw_proposals(
            np.array(row["latent"], np.float64),
            center=d["centers"][ci],
            contact_control=first // 10,
            radial_heading_degrees=bodies[0]["radial_heading_degrees"],
        )
        contexts.append(
            dict(
                course=ci,
                center=d["centers"][ci],
                actual_base_tag=tag,
                actual_base_quality=row["healthy_quality"],
                response=model,
                proposed_latents=proposals.tolist(),
                all_proposals_require_new_whole_physics=True,
            )
        )
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="TRAIN_RUN_CONTACT_RESPONSE_DIAGNOSED_NOT_MOTOR_POLICY",
            generator="measured_contact_radial_yaw_inverse.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_hash=digest(rp),
            geometry_hash=a.geometry_hash,
            contexts=contexts,
            physical_worlds_executed=0,
            neural_motor_updates=0,
            learned_actor_claim=False,
            predicted_success_is_evidence=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="CONTACT_YAW_EXPERIMENTS_DECLARED",
                contexts=[x["response"] for x in contexts],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
