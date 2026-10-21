"""Finite three-seed review; fixed seed0 primary, no Fresh access or promotion."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-pipeline", type=Path, required=True)
    for n, count in (
        ("pipelines", 3),
        ("new-exams", 3),
        ("new-audits", 3),
        ("reference-shards", 4),
        ("reference-audits", 4),
    ):
        p.add_argument("--" + n, type=Path, nargs=count, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("frozen reviewer required")

    clean()
    if a.output.exists():
        raise ValueError("unused review output required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            primary_seed=0,
            seeds=[0, 1, 2],
            reference_pipeline=str(a.reference_pipeline),
            pipelines=[str(f) for f in a.pipelines],
            new_exams=[str(f) for f in a.new_exams],
            new_audits=[str(f) for f in a.new_audits],
            reference_shards=[str(f) for f in a.reference_shards],
            reference_audits=[str(f) for f in a.reference_audits],
            criteria=dict(
                all_case_and_kick_quality_retained=True,
                all_stable_bodies_retained=True,
                new_TRAIN_quality_gain_at_least=1,
                require_all_three_seeds=True,
            ),
            Fresh_rows_read=0,
            automatic_retry=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    deadline = time.monotonic() + 7200
    waiting = a.pipelines + a.new_exams + a.new_audits
    while not all((f / "complete.json").exists() for f in waiting):
        clean()
        if any((f / "failed.json").exists() for f in waiting):
            raise ValueError("failed upstream, diagnose without automatic retry")
        # A queue writes failures at the stage boundary as well as in children.
        if any(list(f.glob("*-failed.json")) for f in a.pipelines):
            raise ValueError("failed queue stage")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed audited candidates required")
        time.sleep(10)
    clean()
    from rosclaw_soccer.rsi.paid_router_review import compare_router_frontier

    pins = {}

    def read(root, status):
        rp = root / "complete.json"
        r = json.loads(rp.read_text())
        if (
            rp.is_symlink()
            or (root / "failed.json").exists()
            or r["status"] != status
            or r["hardware_authorized"] is not False
            or r["promotion_authorized"] is not False
        ):
            raise ValueError("closed SIM evidence required")
        for name, h in r.get("input_file_hashes", {}).items():
            if name in pins and pins[name] != h:
                raise ValueError("conflicting source commitments")
            pins[name] = h
        pins[str(rp)] = digest(rp)
        return r

    def pair(root, audit, status, audit_status):
        r = read(root, status)
        ar = read(audit, audit_status)
        rp = root / "complete.json"
        if (
            ar["input_file_hashes"].get(str(rp)) != digest(rp)
            or ar["counts"]["native"] != r["actual_native_steps"]
            or ar["counts"]["neural"] != r["actual_neural_rows"]
            or ar["counts"]["own_neural"] != ar["counts"]["neural"]
            or r.get("summary") != ar.get("summary")
        ):
            raise ValueError("audit must bind this complete own-NN physical experiment")
        return r, ar

    exam_status = "PAID_SUCCESSOR_NEURAL_STUDENT_LIFECYCLE_CLOSED_PENDING_AUDIT"
    audit_status = "PAID_SUCCESSOR_STUDENT_ALL_NATIVE_PRIMITIVES_AND_OWN_NEURAL_AUDIT_PASS"
    references = {}
    for bank in ("train", "dev"):
        root = a.reference_pipeline / (bank + "-exam")
        r, _ = pair(root, a.reference_pipeline / (bank + "-audit"), exam_status, audit_status)
        if r["bank"] != bank or [w["case"] for w in r["worlds"]] != list(range(16)):
            raise ValueError("exact complete original bank required")
        references[bank] = r["worlds"]
    baseline = []
    plan_hash = None
    for i, (root, audit) in enumerate(zip(a.reference_shards, a.reference_audits, strict=True)):
        r, ar = pair(
            root,
            audit,
            "PAID_TRAIN_ANCESTOR_COUNTERFACTUAL_SHARD_PENDING_AUDIT",
            "PAID_TRAIN_ANCESTOR_COUNTERFACTUAL_OWN_NN_AND_PHYSICS_AUDIT_PASS",
        )
        if (
            r["shard"] != i
            or ar["shard"] != i
            or [g["case"] for g in r["groups"]] != list(range(5 * i, 5 * i + 5))
            or (plan_hash is not None and r["plan_hash"] != plan_hash)
        ):
            raise ValueError("all four original TRAIN20 shards required")
        plan_hash = r["plan_hash"]
        baseline.extend(g["arms"]["grown"] for g in r["groups"])
    seeds = []
    for seed, pipeline in enumerate(a.pipelines):
        queue = read(pipeline, "FINITE_SUCCESSOR_LEARNING_QUEUE_CLOSED_REQUIRES_AGENT_REVIEW")
        trained = pipeline / "trained" / "complete.json"
        training = json.loads(trained.read_text())
        if training["config"]["seed_index"] != seed or queue["stages"]["trained"]["hash"] != digest(
            trained
        ):
            raise ValueError("fixed seed order and trained identity required")
        comparisons = {}
        for bank in ("train", "dev"):
            root, audit = pipeline / (bank + "-exam"), pipeline / (bank + "-audit")
            r, _ = pair(root, audit, exam_status, audit_status)
            declaration = json.loads((root / "declaration.json").read_text())
            if (
                declaration["student_hash"] != digest(trained)
                or r["bank"] != bank
                or [w["case"] for w in r["worlds"]] != list(range(16))
            ):
                raise ValueError("same fixed student and complete bank required")
            comparisons[bank] = compare_router_frontier(references[bank], r["worlds"])
        r, _ = pair(
            a.new_exams[seed],
            a.new_audits[seed],
            "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT",
            "PAID_ROUTER_TRAIN20_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS",
        )
        if (
            r["student_hash"] != digest(trained)
            or r["plan_hash"] != plan_hash
            or [w["case"] for w in r["worlds"]] != list(range(20))
        ):
            raise ValueError("exact same student on all20 TRAIN contexts required")
        comparisons["new_train"] = compare_router_frontier(baseline, r["worlds"])
        passed = (
            all(v["retention_pass"] for v in comparisons.values())
            and comparisons["new_train"]["has_actual_quality_growth"]
        )
        seeds.append(dict(seed=seed, comparisons=comparisons, frontier_pass=passed))
    check_pins(pins)
    clean()
    passed = all(s["frontier_pass"] for s in seeds)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "ROUTER_FRONTIER_PASS_PENDING_NEW_FRESH"
                if passed
                else "ROUTER_FRONTIER_REJECTED_KEEP_PARENT"
            ),
            source_head=a.expected_head,
            input_file_hashes=pins,
            seeds=seeds,
            primary_seed=0,
            all_three_seeds_pass=passed,
            Fresh_rows_read=0,
            next_action=(
                "DECLARE_NEW_FRESH_FOR_FIXED_SEED0"
                if passed
                else "DIAGNOSE_COUNTEREXAMPLES_NO_BLIND_EPOCH_INCREASE"
            ),
            optimizer_updates=0,
            own_native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
