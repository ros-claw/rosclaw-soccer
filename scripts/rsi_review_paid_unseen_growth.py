"""Authenticate a frozen unseen pair and reconstruct its own memory coverage.

Read-only science: no new physics, labels, fitting, promotion, or auto-retry.
"""

import argparse
import json
import subprocess
from pathlib import Path


def validate_rows(producer, audit):
    """Require all paired contexts and independently matching audited outcomes."""
    rows, outcomes = producer["records"], audit["outcomes"]
    expected = [(i, j) for i in range(32) for j in range(2)]
    if (
        [(r["case"], r["repeat"]) for r in rows] != expected
        or [(r["case"], r["repeat"]) for r in outcomes] != expected
        or producer["repeats_exact"] is not True
        or audit["repeats_exact"] is not True
    ):
        raise ValueError("all32 contexts and two exact whole repeats required")
    for r, o in zip(rows, outcomes, strict=True):
        if any(r[k] != o[k] for k in ("case", "repeat", "quality", "healthy_success")):
            raise ValueError("independent audited outcome mismatch")
        if any(type(r[k]) is not bool for k in ("quality", "healthy_success")):
            raise ValueError("boolean outcomes required")
    for i in range(32):
        if any(rows[2 * i][k] != rows[2 * i + 1][k] for k in ("quality", "healthy_success")):
            raise ValueError("paired outcome repeat mismatch")
    return [r for r in rows if r["repeat"] == 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-root", type=Path, required=True)
    p.add_argument("--plan-hash", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    for arm in ("learned", "zero-control"):
        p.add_argument("--" + arm + "-root", type=Path, required=True)
        p.add_argument("--" + arm + "-audit-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen source required")

    clean()
    if a.output.exists():
        raise ValueError("unused review output required")
    from rosclaw_soccer.rsi.paid_growth_memory_data import growth_exam_candidate
    from rosclaw_soccer.rsi.paid_growth_probe import KIND
    from rosclaw_soccer.rsi.paid_neighborhood_sources import load_neighborhood_plan

    _, plan, pins = load_neighborhood_plan(a.plan_root, a.plan_hash)
    if plan.get("plan_kind") != KIND:
        raise ValueError("explicit frozen-pair unseen fixture required")
    sources = {}
    for arm in ("learned", "zero-control"):
        root = getattr(a, arm.replace("-", "_") + "_root")
        ar = getattr(a, arm.replace("-", "_") + "_audit_root")
        rp, ap = root / "complete.json", ar / "complete.json"
        if any(p.is_symlink() or not p.is_file() for p in (rp, ap)) or any(
            (r / "failed.json").exists() for r in (root, ar)
        ):
            raise ValueError("actual closed producer and audit required")
        r, audit = [json.loads(p.read_text()) for p in (rp, ap)]
        spec = plan["locked_models"][arm]
        _, config, extra = growth_exam_candidate(Path(spec["root"]), spec["hash"], plan)
        if (
            r["status"] != "PAID_CONTACT_NEIGHBORHOOD32_ARM_CLOSED_PENDING_AUDIT"
            or audit["status"] != "PAID_CONTACT_NEIGHBORHOOD32_OWN_NN_IK_ALL_NATIVE_AUDIT_PASS"
            or audit["source_hash"] != digest(rp)
            or r["config"] != config
            or r["candidate_hash"] != spec["hash"]
            or r["model_hash"] != spec["hash"]
            or r["candidate_root"] != spec["root"]
            or r["body_scene_hash"] != plan["body_scene_hash"]
            or r["actual_native_steps"] != 248960
            or r["actual_neural_rows"] != 24896
            or audit["counts"]
            != dict(IK=0, R2_planner=0, episodes=64, native=248960, neural=24896, own_neural=24896)
            or audit["own_native_steps"] != 248960
            or audit["own_trajectory_neural_rows"] != 24896
            or any(
                v["plan_hash"] != a.plan_hash
                or v["source_head"] != a.producer_head
                or v["optimizer_updates"] != 0
                or any(v[k] is not False for k in ("hardware_authorized", "promotion_authorized"))
                for v in (r, audit)
            )
        ):
            raise ValueError("exact fixed pair, source, protocol and native ownNN audit required")
        rows = validate_rows(r, audit)
        for mapping in (
            extra,
            r["input_file_hashes"],
            audit["input_file_hashes"],
            {str(rp): digest(rp), str(ap): digest(ap)},
        ):
            for f, h in mapping.items():
                if f in pins and pins[f] != h:
                    raise ValueError("conflicting input commitment")
                pins[f] = h
        sources[arm] = (root, rows)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_compact_diagnosis import (
            compact_own_trace,
            summarize_compact_trace,
        )
        from rosclaw_soccer.rsi.paid_contact_diagnosis import contact_failure_causes
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        spec = plan["locked_models"]["learned"]
        rp = Path(spec["root"]) / "complete.json"
        fitted = json.loads(rp.read_text())
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(fitted["checkpoint_path"]),
            checkpoint_hash=fitted["checkpoint_hash"],
            training_receipt=rp,
            training_receipt_hash=spec["hash"],
            scene_hash=plan["body_scene_hash"],
        )
        rows = []
        root, learned = sources["learned"]
        for r in learned:
            path = root / r["kick"]["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != r["kick"]["raw_hash"]
            ):
                raise ValueError("confined audited own trajectory required")
            pins[str(path)] = r["kick"]["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                trace = compact_own_trace(actor, {k: z[k].copy() for k in z.files})
            summary = summarize_compact_trace(
                trace, first_foot_substep=r["kick"]["postcontact"]["first_foot_contact_substep"]
            )
            rows.append(
                dict(
                    case=r["case"],
                    source_case=r["source_case"],
                    healthy_success=r["healthy_success"],
                    failure_causes=contact_failure_causes(r["kick"]),
                    own_support=summary,
                )
            )
            write_once(a.output / f"case{r['case']}-support.json", rows[-1])
        successes = {
            arm: [r["case"] for r in records if r["healthy_success"]]
            for arm, (_, records) in sources.items()
        }
        gain = sorted(set(successes["learned"]) - set(successes["zero-control"]))
        loss = sorted(set(successes["zero-control"]) - set(successes["learned"]))
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_UNSEEN_GROWTH_PAIR_OWN_SUPPORT_REVIEWED_NOT_PROMOTED",
                source_head=a.expected_head,
                producer_head=a.producer_head,
                input_file_hashes=pins,
                plan_hash=a.plan_hash,
                cases=rows,
                success_ids=successes,
                paired_gains=gain,
                paired_losses=loss,
                unseen_local_improvement_proven=bool(gain) and not loss,
                no_new_physics=True,
                training_labels_created=False,
                optimizer_updates=0,
                formal_Fresh_claim=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error=str(error), automatic_retry=False, hardware_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
