"""Commit seed0 and all32 new contexts before any Fresh rollout; SIM only."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("student-root", "source-root", "source-audit-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("student-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or digest(a.student_root / "complete.json") != a.student_hash
        or (a.student_root / "failed.json").exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused declaration, frozen candidate and source required")
    from rosclaw_soccer.rsi.paid_fresh_protocol import PROTOCOL, fresh_lifecycle_cases
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    standing_gate(a.source_root, a.source_audit_root, variant=1, bank="train")
    training, source = [
        json.loads((root / "declaration.json").read_text())
        for root in (a.student_root, a.source_root)
    ]
    receipt = json.loads((a.student_root / "complete.json").read_text())
    if (
        receipt["config"]["seed_index"] != 0
        or receipt["config"]["protocol"] != "acquired-parent-progressive-contact-v1"
    ):
        raise ValueError("seed0 fixed before Fresh; no outcome-based seed selection")
    pins = dict(receipt["input_file_hashes"])
    audit = json.loads((a.source_audit_root / "complete.json").read_text())
    for path, h in audit["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("base/candidate lineage conflict")
        pins[path] = h
    for path in (
        a.student_root / "complete.json",
        a.student_root / "declaration.json",
        Path(receipt["checkpoint_path"]),
        a.source_root / "complete.json",
        a.source_root / "declaration.json",
        a.source_audit_root / "complete.json",
    ):
        pins[str(path)] = digest(path)
    cases = fresh_lifecycle_cases(source["cases"])
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    # Publish the entire proposal BEFORE model compilation/admission. If any
    # case fails, this exact declaration remains failed; no silent replacement.
    declaration = dict(
        schema=PROTOCOL,
        source_head=a.expected_head,
        input_file_hashes=pins,
        student_root=str(a.student_root),
        student_hash=a.student_hash,
        checkpoint_hash=receipt["checkpoint_hash"],
        parent_hash=receipt["parent_hash"],
        data_root=training["data_root"],
        data_hash=receipt["data_hash"],
        cases=cases,
        candidate_seed_index=0,
        arms=["r2", "parent", "progressive"],
        repeats=2,
        skills=3,
        standing_controls=100,
        body_resets=0,
        case_generation_seed=2026102207,
        candidate_frozen_before_generation=True,
        candidate_seed_chosen_on_fresh=False,
        Fresh_episodes_fit=0,
        unchanged_body_scene_hash=receipt["body_scene_hash"],
        evaluation=dict(
            candidate_stable_required=32,
            no_case_regression_vs_R2=True,
            no_case_regression_vs_parent=True,
            third_gain_vs_R2_minimum=4,
            third_gain_vs_parent_minimum=1,
            all_independent_audits_required=True,
        ),
        evidence_ceiling=(
            "new contexts in eight known motion families; "
            "not historical52,not novel motions,not match"
        ),
        hardware_authorized=False,
        promotion_authorized=False,
    )
    write_once(a.output / "declaration.json", declaration)
    try:
        from dataclasses import asdict

        import numpy as np
        import torch

        from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, cases)
        if scene != receipt["body_scene_hash"] or bench.native_steps or bench.neural_rows:
            raise ValueError("preflight must preserve body and perform no physics/inference")
        # Exact trained policy admission, independent of any Fresh labels.
        load_paid_student(
            policy,
            checkpoint=Path(receipt["checkpoint_path"]),
            checkpoint_hash=receipt["checkpoint_hash"],
            training_receipt=a.student_root / "complete.json",
            training_receipt_hash=a.student_hash,
            scene_hash=scene,
        )
        native_entries = []
        for case in cases:
            ep = bench.make(case)
            q, v = ep.layout.canonical(ep.env.data)
            if not np.array_equal(q[36:39], case["ball_world_initial"]) or not np.array_equal(
                v[35:37], case["velocity_world_initial"]
            ):
                raise ValueError("explicit Fresh position/velocity not actually applied")
            native_entries.append(
                dict(case=case["case"], position=q[36:39].tolist(), velocity=v[35:37].tolist())
            )
        model = bench.template.model
        ball_id = model.body("soccer_ball").id
        geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == ball_id]
        if len(geoms) != 1:
            raise ValueError("one exact compiled sphere required")
        dimensions = inspect_native_ball_dimensions(model, geom_id=geoms[0])
        if (
            dimensions.radius_m != 0.11
            or dimensions.body_mass_kg != 0.43
            or model.opt.timestep != 0.002
        ):
            raise ValueError("unchanged declared sphere/mass and2ms physics required")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_FRESH_ALL32_CASES_DECLARED_AND_PREFLIGHT_PASS_NO_ROLLOUT",
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                protocol=PROTOCOL,
                native_entries=native_entries,
                compiled_ball_dimensions=asdict(dimensions),
                own_native_steps=0,
                Fresh_episodes_executed=0,
                Fresh_episodes_fit=0,
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
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
