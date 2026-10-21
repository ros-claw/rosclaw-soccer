"""Declare all20 NEW TRAIN cases before same-entry ancestor counterfactuals."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("student-root", "source-root", "source-audit-root", "exclude-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("student-hash", "exclude-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or digest(a.student_root / "complete.json") != a.student_hash
    ):
        raise ValueError("frozen model and unused declaration required")
    from rosclaw_soccer.rsi.paid_ancestor_study import (
        SCHEDULES,
        ancestor_train_cases,
        case_condition,
    )
    from rosclaw_soccer.rsi.paid_fresh_evidence import load_fresh_declaration
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate
    from rosclaw_soccer.sim.contracts import hash_json

    standing_gate(a.source_root, a.source_audit_root, variant=1, bank="train")
    receipt = json.loads((a.student_root / "complete.json").read_text())
    source = json.loads((a.source_root / "declaration.json").read_text())
    if (
        receipt["config"]["protocol"] != "acquired-parent-progressive-contact-v1"
        or receipt["config"]["seed_index"] != 0
    ):
        raise ValueError("predeclared frozen grown seed0 required")
    _, excluded = load_fresh_declaration(a.exclude_root, a.exclude_hash)
    excluded_hashes = sorted(hash_json(case_condition(c)) for c in excluded["cases"])
    cases = ancestor_train_cases(source["cases"])
    if any(hash_json(case_condition(c)) in excluded_hashes for c in cases):
        raise ValueError("TRAIN declaration overlaps consumed Fresh; no resampling")
    pins = dict(receipt["input_file_hashes"])
    audit = json.loads((a.source_audit_root / "complete.json").read_text())
    for path, h in audit["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("actor/source lineage conflict")
        pins[path] = h
    for f in (
        a.student_root / "complete.json",
        Path(receipt["checkpoint_path"]),
        a.source_root / "declaration.json",
        a.source_root / "complete.json",
        a.source_audit_root / "complete.json",
        a.exclude_root / "complete.json",
        a.exclude_root / "declaration.json",
    ):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    declaration = dict(
        schema="paid.ancestor_counterfactual.TRAIN20.v1",
        source_head=a.expected_head,
        input_file_hashes=pins,
        cases=cases,
        schedules={k: list(v) for k, v in SCHEDULES.items()},
        repeats=2,
        shards=4,
        cases_per_shard=5,
        student_root=str(a.student_root),
        student_hash=a.student_hash,
        checkpoint_hash=receipt["checkpoint_hash"],
        body_scene_hash=receipt["body_scene_hash"],
        bank="train",
        Fresh_fit_rows=0,
        excluded_context_hashes=excluded_hashes,
        exclusion_source_hash=a.exclude_hash,
        hypotheses=[
            "third-slot R2 rescues at identical entry",
            "whole-R2 gains may require earlier body trajectory changes",
        ],
        teacher_required_before_router_training=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    write_once(a.output / "declaration.json", declaration)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, cases)
        if scene != receipt["body_scene_hash"] or bench.native_steps or bench.neural_rows:
            raise ValueError("zero-step full-case admission required")
        load_paid_student(
            policy,
            checkpoint=Path(receipt["checkpoint_path"]),
            checkpoint_hash=receipt["checkpoint_hash"],
            training_receipt=a.student_root / "complete.json",
            training_receipt_hash=a.student_hash,
            scene_hash=scene,
        )
        for case in cases:
            ep = bench.make(case)
            q, v = ep.layout.canonical(ep.env.data)
            if not np.array_equal(q[36:39], case["ball_world_initial"]) or not np.array_equal(
                v[35:37], case["velocity_world_initial"]
            ):
                raise ValueError("TRAIN entry differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_TRAIN20_ANCESTOR_STUDY_PREREGISTERED_NO_ROLLOUT",
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                cases_preflighted=20,
                own_native_steps=0,
                rollouts=0,
                bank="train",
                Fresh_fit_rows=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                entire_declaration_rejected=True,
                resampled_cases=0,
                hardware_authorized=False,
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
