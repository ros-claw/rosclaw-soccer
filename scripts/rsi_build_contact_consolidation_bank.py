"""Admit all robust teacher outcomes and exact current-policy old-skill anchors.

Failure trajectories remain negative labelled rows, never positive imitation.
Only observations and executed targets are learned; no teacher hidden state.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("robust-root", "robust-audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("robust-auditor-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.contact_consolidation_data import (
        BANK_STATUS,
        contact_loss_weights,
        own_reference_sequence,
    )
    from rosclaw_soccer.rsi.stroke_robustness import paired_robustness_summary
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_stroke_search_common import open_motor
    from scripts.rsi_stroke_teacher_admission import admit_teacher

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean pinned whole-history bank source required")

    clean()
    rp, ap, dp = (
        a.robust_root / "complete.json",
        a.robust_audit_root / "complete.json",
        a.robust_root / "declaration.json",
    )
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (rp, ap, dp))
        or (a.robust_audit_root / "failed.json").exists()
    ):
        raise ValueError(
            "complete ordinary independently audited cohort and unused output required"
        )
    r, ar, d = [json.loads(p.read_text()) for p in (rp, ap, dp)]
    if (
        ar["status"] != "FIXED_STROKE_PAIRED48_ALL96_OWN_NATIVE_AUDIT_PASS"
        or ar["source_head"] != a.robust_auditor_head
        or ar["source_hash"] != digest(rp)
        or r["declaration_hash"] != digest(dp)
        or ar["worlds"] != 96
        or ar["counts"]["native"] != r["actual_native_steps"]
        or ar["counts"]["neural"] != r["actual_neural_decisions"]
        or ar["summary"] != paired_robustness_summary(r["records"])
        or ar["summary"]["teacher_locally_qualified"] is not True
        or any(
            x["hardware_authorized"] is not False
            or x["promotion_authorized"] is not False
            or x["student_success_claim"] is not False
            or x["PPO_actor_rows"] != 0
            for x in (r, ar)
        )
    ):
        raise ValueError("full paired robustness qualification is required before consolidation")
    pins = dict(ar["input_file_hashes"])

    def receipt(h):
        ps = [Path(p) for p, v in pins.items() if v == h and p.endswith("/complete.json")]
        if len(ps) != 1:
            raise ValueError("unique exact lineage required")
        return ps[0], json.loads(ps[0].read_text())

    sp, search = receipt(d["search_receipt_hash"])
    sap, _ = receipt(d["search_audit_hash"])
    original, training_hash, contexts, _, _ = admit_teacher(
        sp.parent, sap.parent, d["search_audit_hash"]
    )
    for path, h in original.items():
        if path in pins and pins[path] != h:
            raise ValueError("conflicting teacher lineage")
        pins[path] = h
    sd = json.loads((sp.parent / "declaration.json").read_text())
    ep, exam = receipt(sd["source_exam_hash"])
    tp, training = receipt(training_hash)
    _, legacy = receipt(training["data_hash"])
    anchors = [
        x
        for x in exam["records"]
        if x["context"] < 20 and x["repeat"] == 0 and x["healthy_quality"]
    ]
    if len(anchors) != 14:
        raise ValueError("ALL14 actual current-policy successful skills required")
    episodes = [(ep.parent, x, "retained_W6_skill", True) for x in anchors]
    for context in (4, 20, 21):
        tag = f"selected-context{context}-repeat0" if context != 21 else "control-context21-repeat0"
        row = next(x for x in search["records"] if x["tag"] == tag)
        if not row["healthy_quality"]:
            raise ValueError("original three competent teacher witnesses required")
        episodes.append((sp.parent, row, "source_stroke_teacher", True))
    episodes.extend(
        (
            a.robust_root,
            x,
            "robust_stroke_teacher" if x["healthy_quality"] else "failed_teacher_negative",
            x["healthy_quality"],
        )
        for x in r["records"]
        if x["arm"] == "teacher"
    )
    if len(episodes) != 65:
        raise ValueError("all14 anchors,3 source teachers,48 robust outcomes required")
    for root, row, _, _ in episodes:
        for art in row["artifacts"].values():
            path = root / art["path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != art["hash"]
            ):
                raise ValueError("exact confined whole trajectory required")
            pins[str(path)] = art["hash"]
    for path in (rp, ap, dp, tp):
        pins[str(path)] = digest(path)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.contact_weighted_consolidation_bank65.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            warm_start_hash=training_hash,
            robustness_audit_hash=digest(ap),
            records=65,
            positive_teacher_required=True,
            inherited_anchors="ALL14_ACTUAL_W6_SUCCESSFUL_BEHAVIORS_NOT_ORIGINAL_PARENT_TARGETS",
            teacher_hidden_state_supplied_to_learner=False,
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            student_success_claim=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    records = []
    calls = 0
    try:
        import numpy as np

        bench, model = open_motor(pins, training_hash, contexts)
        parent = model.parent
        scale = parent.get_buffer("action_scale").numpy().copy()
        for index, (root, row, role, eligible) in enumerate(episodes):
            raw = {}
            for name, art in row["artifacts"].items():
                with np.load(root / art["path"], allow_pickle=False) as z:
                    raw[name] = {k: z[k].copy() for k in z.files}
            kick = len(raw["kick"]["NN_obs"])
            stand = len(raw["stand"]["NN_obs"]) if "stand" in raw else 0
            if eligible and (stand != 100 or row["result"]["body_failed"]):
                raise ValueError("positive imitation requires complete healthy recovery")
            obs = np.concatenate([raw[name]["NN_obs"] for name in ("kick", "stand") if name in raw])
            targets = np.concatenate(
                [raw[name]["NN_action"] for name in ("kick", "stand") if name in raw]
            )
            nominals, deltas = own_reference_sequence(parent, obs)
            calls += len(obs)
            weights = contact_loss_weights(
                kick, stand, row["result"]["postcontact"]["first_foot_contact_substep"]
            )
            path = a.output / f"sequence{index}.npz"
            np.savez_compressed(
                path,
                observations=obs,
                targets=targets,
                action_scale=scale,
                reference_nominal=nominals,
                reference_delta=deltas,
                loss_weights=weights,
            )
            records.append(
                dict(
                    index=index,
                    role=role,
                    bc_weight=1.0 if eligible else 0.0,
                    rows=len(obs),
                    kick_rows=kick,
                    stand_rows=stand,
                    task_success=bool(row["healthy_quality"]),
                    raw_path=path.name,
                    raw_hash=digest(path),
                    source_root=str(root),
                    source_tag=row.get("tag"),
                    source_context=row.get("context"),
                    source_course_index=row.get("course_index"),
                    measured_failure=None if eligible else row["result"],
                )
            )
        if bench.native_steps != 0:
            raise ValueError("bank reference observer must not integrate physics")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status=BANK_STATUS,
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                records=records,
                parent_lineage_hash=training["parent_lineage_hash"],
                warm_start_hash=training_hash,
                context_witness_hash=legacy["teacher_hash"],
                robustness_audit_hash=digest(ap),
                whole_actual_sequences=True,
                teacher_hidden_state_supplied_to_learner=False,
                learner_hidden_width=256,
                observation_width=160,
                own_reference_queries=calls,
                physical_steps=0,
                optimizer_updates=0,
                PPO_actor_rows=0,
                DEV_fit_rows=0,
                Fresh_fit_rows=0,
                student_success_claim=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed_sequences=len(records),
                own_reference_queries=calls,
                physical_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
