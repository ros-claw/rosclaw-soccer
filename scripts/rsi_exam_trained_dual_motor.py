"""Unassisted learned-motor native exam on TRAIN20 retention + two consumed run contexts.

No online IK corrections: the zero intent is an audited identity operation.
Not legacy52, not Fresh, not goal-plane .1m, not continuous4v4.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-kind",
        choices=(
            "supervised",
            "ar1-actor-critic",
            "contact-consolidated",
            "contact-consolidated-v2",
            "team-pass-consolidated-v1",
            "team-pass-corrected-v2",
        ),
        default="supervised",
    )
    for n in ("training-root", "cpu-audit-root", "rehearsal-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("training-hash", "cpu-auditor-head", "rehearsal-hash", "expected-head"):
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
            raise ValueError("clean frozen unassisted native motor exam required")

    clean()
    tp, ap, rp = (
        a.training_root / "complete.json",
        a.cpu_audit_root / "complete.json",
        a.rehearsal_root / "complete.json",
    )
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (tp, ap, rp))
        or digest(tp) != a.training_hash
        or digest(rp) != a.rehearsal_hash
        or (a.cpu_audit_root / "failed.json").exists()
    ):
        raise ValueError("exact closed candidate, CPU audit and baseline required")
    t, ar, r = [json.loads(x.read_text()) for x in (tp, ap, rp)]
    if (
        ar["status"] != "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS"
        or ar.get("checkpoint_kind", "supervised") != a.checkpoint_kind
        or ar["source_head"] != a.cpu_auditor_head
        or ar["source_hash"] != a.training_hash
        or ar["checkpoint_hash"] != t["checkpoint_hash"]
        or ar["data_hash"] != t["data_hash"]
        or ar["own_actor_decisions"] != t["cpu_offline_decisions"]
        or ar["physical_steps_executed"] != 0
        or r["status"] != "ACTUAL_PARENT_TRAIN20_REHEARSAL_COLLECTED_PENDING_NATIVE_AUDIT"
        or len(r["records"]) != 20
    ):
        raise ValueError(
            "independently checked actual candidate and complete original baseline required"
        )
    pins = dict(ar["input_file_hashes"])
    for x in (tp, ap, rp):
        pins[str(x)] = digest(x)
    rd = a.rehearsal_root / "declaration.json"
    if pins.get(str(rd)) != digest(rd) or r["declaration_hash"] != digest(rd):
        raise ValueError("source-bound original case declaration required")
    cases = json.loads(rd.read_text())["cases"]
    data_paths = [
        Path(x) for x, h in pins.items() if h == t["data_hash"] and x.endswith("/complete.json")
    ]
    parent_paths = [
        Path(x)
        for x, h in pins.items()
        if h == t["parent_lineage_hash"] and x.endswith("/complete.json")
    ]
    if len(data_paths) != 1 or len(parent_paths) != 1:
        raise ValueError("exact training data and frozen parent required")
    data = json.loads(data_paths[0].read_text())
    parent_record = json.loads(parent_paths[0].read_text())
    from rosclaw_soccer.rsi.contact_consolidation_data import context_witness_hash

    witness_hash = context_witness_hash(data, a.checkpoint_kind)
    teacher_paths = [
        Path(x) for x, h in pins.items() if h == witness_hash and x.endswith("/complete.json")
    ]
    if len(teacher_paths) != 1:
        raise ValueError("exact consumed run-context witness required")
    teacher_decl = json.loads((teacher_paths[0].parent / "declaration.json").read_text())
    contexts = [dict(kind="old_first_skill", case=c, backoff_m=0.0) for c in cases]
    contexts += [
        dict(kind="consumed_run", case=teacher_decl["source_case"], backoff_m=c["backoff_m"])
        for c in teacher_decl["courses"]
    ]
    if len(contexts) != 22:
        raise ValueError("fixed22 contexts required before execution")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.unassisted_TRAIN20_and_consumed_run2.v1",
            checkpoint_kind=a.checkpoint_kind,
            source_head=a.expected_head,
            input_file_hashes=pins,
            training_hash=a.training_hash,
            cpu_audit_hash=digest(ap),
            baseline_hash=a.rehearsal_hash,
            contexts=contexts,
            repeats=2,
            maximum_native_steps=176000,
            maximum_actor_decisions=17600,
            hidden_width=256,
            online_teacher_enabled=False,
            Fresh_cases=0,
            legacy52_claim=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    calls = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.motor_reference_rollout import collect_parent_reference
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, [x["case"] for x in contexts])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(parent_record["checkpoint_path"]),
            checkpoint_hash=parent_record["checkpoint_hash"],
            training_receipt=parent_paths[0],
            training_receipt_hash=t["parent_lineage_hash"],
            scene_hash=scene,
        )
        actor, _ = load_dual_candidate_for_sim_audit(
            parent,
            parent_lineage_hash=t["parent_lineage_hash"],
            root=a.training_root,
            expected_receipt_hash=a.training_hash,
            checkpoint_kind=a.checkpoint_kind,
        )
        records = []
        for index, ctx in enumerate(contexts):
            previous = None
            for repeat in range(2):
                ep = bench.make(ctx["case"])
                if ctx["backoff_m"]:
                    q, _ = ep.layout.canonical(ep.env.data)
                    w, x, y, z = q[3:7]
                    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                    ep.env.data.qpos[ep.layout.rq : ep.layout.rq + 2] -= ctx[
                        "backoff_m"
                    ] * np.array([np.cos(yaw), np.sin(yaw)])
                    mujoco.mj_forward(ep.env.model, ep.env.data)
                kick, stand, summary, nn = collect_parent_reference(
                    ep,
                    actor,
                    foot="left" if "left" in ctx["case"]["source"] else "right",
                    hidden_width=256,
                )
                calls += nn
                folder = a.output / f"context{index}-repeat{repeat}"
                folder.mkdir()
                blobs = dict(kick=kick)
                if stand is not None:
                    blobs["stand"] = stand
                if previous is not None:
                    if set(blobs) != set(previous):
                        raise ValueError("whole repeat termination differs")
                    for k in blobs:
                        identical(blobs[k], previous[k])
                else:
                    previous = blobs
                artifacts = {}
                for k, v in blobs.items():
                    path = folder / (k + ".npz")
                    np.savez_compressed(path, **v)
                    artifacts[k] = dict(path=str(path.relative_to(a.output)), hash=digest(path))
                raw = {
                    k: np.concatenate([kick[k], stand[k]]) if stand is not None else kick[k]
                    for k in ("qpos", "qvel", "forces", "body_failed")
                }
                f = raw["forces"].reshape(-1, 3)
                post = native_contact_window(
                    force=f[:, :2],
                    canonical_qpos=raw["qpos"].reshape(-1, 43),
                    canonical_qvel=raw["qvel"].reshape(-1, 41),
                    goal=ep.goal,
                    intended_foot=0 if "left" in ctx["case"]["source"] else 1,
                )
                result = dict(
                    body_failed=bool(raw["body_failed"].any()),
                    forbidden=bool((f[:, 2] > 1).any()),
                    nonfoot_impulse_Ns=float(f[:, 2].sum() * 0.002),
                    postcontact=post,
                )
                row = dict(
                    context=index,
                    kind=ctx["kind"],
                    repeat=repeat,
                    result=result,
                    stand=summary,
                    healthy_quality=bool(
                        contact_quality(result)
                        and summary
                        and summary["healthy"]
                        and summary["stability_observed"]
                    ),
                    native_steps=len(raw["qpos"]) * 10,
                    actor_decisions=nn,
                    artifacts=artifacts,
                    unassisted=True,
                )
                write_once(folder / "complete.json", row)
                records.append(row)
                print(
                    json.dumps(
                        dict(
                            context=index,
                            repeat=repeat,
                            quality=row["healthy_quality"],
                            body_failed=result["body_failed"],
                            forbidden=result["forbidden"],
                            lateral=post["goal_lateral_1p2s_m"],
                            native=bench.native_steps,
                        )
                    ),
                    flush=True,
                )
                if bench.native_steps > 176000 or calls > 17600:
                    raise ValueError("bounded actual exam budget exceeded")
        achieved = [
            x["context"]
            for x in records
            if x["repeat"] == 0 and x["kind"] == "old_first_skill" and x["healthy_quality"]
        ]
        protected = [x["case"] for x in r["records"] if x["healthy_quality"]]
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_RECURRENT_NATIVE_TRAIN_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                training_hash=a.training_hash,
                checkpoint_hash=t["checkpoint_hash"],
                records=records,
                actual_native_steps=bench.native_steps,
                actual_actor_decisions=calls,
                recurrent_branch_queries=2 * calls,
                whole_repeats_exact=True,
                old_success_cases=achieved,
                old_protected_cases=protected,
                old_protected_retained=all(x in achieved for x in protected),
                consumed_run_successes=sum(
                    x["healthy_quality"]
                    for x in records
                    if x["kind"] == "consumed_run" and x["repeat"] == 0
                ),
                Fresh_success_claim=False,
                legacy52_claim=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                actual_native_steps=bench.native_steps if bench else 0,
                actual_actor_decisions=calls,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
