"""Finite response-driven contact teacher: six anchors plus24 own CEM futures."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "world", "motion-csv", "model-root", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-bank-hash", "expected-rejected-head"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--previous-producer", type=Path, action="append", required=True)
    parser.add_argument(
        "--rejected-producer",
        nargs=3,
        action="append",
        required=True,
        metavar=("UNIT", "ROOT", "SHA256"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cluster", type=int, choices=range(4))
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--expected-preflight-hash")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise ValueError("sealed response CEM source required")
    for name in (
        Path(__file__).name,
        "rsi_response_cem_admission.py",
        "rsi_response_cem_builder.py",
        "rsi_probe_contact_geometry.py",
        "rsi_native_context_recovery_common.py",
    ):
        if (
            subprocess.check_output(["git", "show", f"{head}:scripts/{name}"])
            != (Path.cwd() / "scripts" / name).read_bytes()
        ):
            raise ValueError("response CEM bootstrap differs from sealed source")
    if args.preflight_only:
        if any(x is not None for x in (args.cluster, args.preflight, args.expected_preflight_hash)):
            raise ValueError("zero-actor preflight cannot allocate worker")
    elif any(x is None for x in (args.cluster, args.preflight, args.expected_preflight_hash)):
        raise ValueError("worker requires same-source zero-actor proof")
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_response_cem_admission import admit

    bank, courses, previous, pins = admit(args)
    if not args.preflight_only:
        if digest(args.preflight / "complete.json") != args.expected_preflight_hash:
            raise ValueError("same-source preflight identity differs")
        declaration = json.loads((args.preflight / "declaration.json").read_text())
        proof = json.loads((args.preflight / "complete.json").read_text())
        if (
            declaration["source_head"] != head
            or declaration["input_file_hashes"] != pins
            or proof["new_actor_allocations"] != 0
            or proof["new_native_steps"] != 0
            or proof["new_model_calls"] != 0
            or proof["bank_hash"] != bank["bank_hash"]
            or proof["declaration_hash"] != digest(args.preflight / "declaration.json")
            or (args.preflight / "failure.json").exists()
        ):
            raise ValueError("exact same-source full-bank zero-actor contract required")
        pins.update({str(p): digest(p) for p in args.preflight.iterdir() if p.is_file()})
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.bounded_response_cem16.v1",
        source_head=head,
        input_file_hashes=pins,
        bank_hash=bank["bank_hash"],
        courses=[asdict(c) for c in courses],
        cluster=args.cluster,
        anchor_count=6,
        laws=[0, 3, 4],
        iterations_per_law=2,
        new_samples_per_iteration=4,
        initial_std_m=0.015,
        minimum_std_m=0.003,
        maximum_draws_per_batch=256,
        maximum_candidates_per_course=30,
        full_bank_native_upper_bound=1392000,
        current_source_changes_explicit_forecast_budget_not_default_motor_control=True,
        anchors_all18_identity_required=True,
        pose_phase_and_body_unchanged=True,
        optimizer_steps=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    results = []
    try:
        if args.preflight_only:
            check_pins(pins)
            write_once(
                args.output / "complete.json",
                dict(
                    source_head=head,
                    bank_hash=bank["bank_hash"],
                    declaration_hash=digest(args.output / "declaration.json"),
                    new_actor_allocations=0,
                    new_native_steps=0,
                    new_model_calls=0,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
            )
            print("FULL16_RESPONSE_CEM_ADMITTED_ZERO_ACTORS", flush=True)
            return
        import numpy as np

        from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
        from scripts.rsi_probe_contact_geometry import run_course
        from scripts.rsi_response_cem_builder import build_candidates

        choices = [Path(p) for p in pins if Path(p).name == "entry-affordance.json"]
        if len(choices) != 1:
            raise ValueError("one bound original affordance required")
        stored = json.loads(choices[0].read_text())
        fields = {
            k: stored[k]
            for k in (
                "ball_offset_xy_m",
                "historical_frame",
                "historical_forward_speed_mps",
                "historical_artifact_hash",
            )
        }
        fields["ball_offset_xy_m"] = tuple(fields["ball_offset_xy_m"])
        affordance = SkillEntryAffordance(**fields)
        if affordance.contract_hash != stored["contract_hash"]:
            raise ValueError("same original entry affordance required")
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        for item in courses:
            if item.parent_index == args.cluster:
                results.append(
                    run_course(
                        args,
                        item,
                        previous[args.cluster],
                        poses,
                        affordance,
                        pins,
                        candidate_builder=build_candidates,
                    )
                )
        if len(results) != 4 or any(
            len(r["candidates"]) != 30 or len(r["proposals"]) != 6 for r in results
        ):
            raise ValueError("whole fixed four-course30-candidate cluster required")
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                cluster=args.cluster,
                source_head=head,
                bank_hash=bank["bank_hash"],
                actual_native_steps=sum(r["actual_native_steps"] for r in results),
                forecast_native_steps=sum(r["forecast_native_steps"] for r in results),
                qualified=sum(r["score"]["qualified"] for r in results),
                old_teacher_success_lost=sum(r["old_teacher_success_lost"] for r in results),
                old_teacher_failure_rescued=sum(r["old_teacher_failure_rescued"] for r in results),
                added_unsafe_vs_old_teacher=sum(r["added_unsafe_vs_old_teacher"] for r in results),
                cem_distribution_updates=24,
                neural_optimizer_updates=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json",
            dict(
                error=repr(error),
                completed_courses=len(results),
                completed_actual_steps=sum(r["actual_native_steps"] for r in results),
                completed_forecast_steps=sum(r["forecast_native_steps"] for r in results),
                incomplete_course_accounting_in_case_failure=True,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
