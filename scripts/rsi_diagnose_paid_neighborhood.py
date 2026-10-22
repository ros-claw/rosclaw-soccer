"""Read audited own-policy coverage on all32 perturbations; no new labels or physics."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-root", type=Path, required=True)
    p.add_argument("--plan-hash", required=True)
    for name in ("arm-root", "audit-root"):
        p.add_argument("--" + name, nargs=2, action="append", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen neighborhood diagnostic required")

    clean()
    if a.output.exists():
        raise ValueError("unused diagnostic output required")
    from rosclaw_soccer.rsi.paid_neighborhood_sources import load_audited_neighborhood_arms
    from rosclaw_soccer.rsi.paid_neighborhood_study import neighborhood_config

    roots, audits = [{n: Path(v) for n, v in values} for values in (a.arm_root, a.audit_root)]
    if any(len(v) != 3 for v in (a.arm_root, a.audit_root)) or any(
        set(v) != set(neighborhood_config()["arms"]) for v in (roots, audits)
    ):
        raise ValueError("three unique declared arms and audits required")
    deadline = time.monotonic() + 14400
    while not all((r / "complete.json").exists() for r in (*roots.values(), *audits.values())):
        clean()
        if any((r / "failed.json").exists() for r in (*roots.values(), *audits.values())):
            raise ValueError("source contract failed; preserve, never automatic retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("bounded all-arm own-audit wait")
        time.sleep(10)
    plan, records, pins = load_audited_neighborhood_arms(
        a.plan_root, a.plan_hash, roots, audits, a.producer_head
    )
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_head=a.producer_head,
            plan_hash=a.plan_hash,
            roots={k: str(v) for k, v in roots.items()},
            audits={k: str(v) for k, v in audits.items()},
            input_file_hashes=pins,
            physical_replay_performed=False,
            optimizer_updates=0,
            training_labels_created=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
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
        spec = plan["specifications"]["models"]["learned_memory"]
        sp = Path(spec["root"]) / "complete.json"
        student = json.loads(sp.read_text())
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(student["checkpoint_path"]),
            checkpoint_hash=student["checkpoint_hash"],
            training_receipt=sp,
            training_receipt_hash=spec["hash"],
            scene_hash=plan["body_scene_hash"],
        )
        rows = []
        for i in range(32):
            source = {arm: r["records"][2 * i] for arm, r in records.items()}
            student_row = source["learned_memory"]
            link = student_row["kick"]
            path = roots["learned_memory"] / link["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(roots["learned_memory"].resolve())
                or pins.get(str(path)) != link["raw_hash"]
                or digest(path) != link["raw_hash"]
            ):
                raise ValueError("exact independently audited student trajectory required")
            with np.load(path, allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            trace = compact_own_trace(actor, raw)
            summary = summarize_compact_trace(
                trace, first_foot_substep=link["postcontact"]["first_foot_contact_substep"]
            )
            own = a.output / f"case{i}-own-support.npz"
            np.savez_compressed(own, **trace)
            rows.append(
                dict(
                    case=i,
                    source_case=plan["cases"][i]["source_case"],
                    healthy_success={k: v["healthy_success"] for k, v in source.items()},
                    quality={k: v["quality"] for k, v in source.items()},
                    student_failure_causes=contact_failure_causes(link),
                    own_support=summary,
                    trace_path=str(own),
                    trace_hash=digest(own),
                )
            )
        successes = {arm: [r["case"] for r in rows if r["healthy_success"][arm]] for arm in records}
        teacher_only = [
            r["case"]
            for r in rows
            if r["healthy_success"]["fixed_successful_teacher"]
            and not r["healthy_success"]["learned_memory"]
        ]
        gains = sorted(set(successes["learned_memory"]) - set(successes["frozen_parent"]))
        losses = sorted(set(successes["frozen_parent"]) - set(successes["learned_memory"]))
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CONTACT_NEIGHBORHOOD32_OWN_SUPPORT_DIAGNOSIS",
                source_head=a.expected_head,
                producer_head=a.producer_head,
                input_file_hashes=pins,
                plan_hash=a.plan_hash,
                cases=rows,
                healthy_success_ids=successes,
                learned_vs_parent_gains=gains,
                learned_vs_parent_losses=losses,
                teacher_only_success_ids=teacher_only,
                teacher_only_mean_support_positive_rows=float(
                    np.mean([rows[i]["own_support"]["support_positive_rows"] for i in teacher_only])
                )
                if teacher_only
                else None,
                physical_replay_performed=False,
                optimizer_updates=0,
                training_labels_created=False,
                repeats_are_not_extra_contexts=True,
                statistical_independence_not_established=True,
                Fresh_rows_read=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                optimizer_updates=0,
                physical_replay_performed=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
