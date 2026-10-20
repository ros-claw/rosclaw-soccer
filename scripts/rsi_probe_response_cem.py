"""Finite response-driven contact teacher: six anchors plus24 own CEM futures."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=("response_cem", "matched_entry"), default="response_cem"
    )
    parser.add_argument("--prior-gate", type=Path)
    parser.add_argument("--expected-prior-gate-hash")
    parser.add_argument("--prior-gate-unit")
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
        "rsi_matched_entry_builder.py",
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

    if args.study == "matched_entry" and any(
        x is None for x in (args.prior_gate, args.expected_prior_gate_hash, args.prior_gate_unit)
    ):
        raise ValueError("matched entry requires the closed rejected CEM gate")
    bank, courses, previous, pins = admit(args)
    schema = "soccer.rsi.bounded_response_cem16.v1"
    if args.study == "matched_entry":
        if any(
            x is None
            for x in (args.prior_gate, args.expected_prior_gate_hash, args.prior_gate_unit)
        ):
            raise ValueError("matched entry requires the closed rejected CEM gate")
        terminal = dict(
            line.split("=", 1)
            for line in subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    args.prior_gate_unit,
                    "-p",
                    "MainPID",
                    "-p",
                    "Result",
                    "-p",
                    "SubState",
                    "-p",
                    "ExecMainStatus",
                ],
                text=True,
            ).splitlines()
        )
        if (
            terminal != dict(MainPID="0", Result="success", SubState="exited", ExecMainStatus="0")
            or digest(args.prior_gate / "complete.json") != args.expected_prior_gate_hash
        ):
            raise ValueError("prior CEM gate must be closed and authenticated")
        gate = json.loads((args.prior_gate / "complete.json").read_text())
        if (
            gate["status"] != "CEM_FAMILY_REJECTED_STOP"
            or gate["producer_claimed_qualified"] != 6
            or gate["launched_reviews"]
        ):
            raise ValueError("explicit no-gain CEM rejection required for this pivot")
        gd = json.loads((args.prior_gate / "declaration.json").read_text())
        check_pins(gd["input_file_hashes"])
        pins.update(gd["input_file_hashes"])
        pins.update({str(p): digest(p) for p in args.prior_gate.iterdir() if p.is_file()})
        protocol = Path.cwd() / "docs/rsi/matched-entry-bridge-preregister-20261020.zh-CN.md"
        pins[str(protocol)] = digest(protocol)
        schema = "soccer.rsi.matched_entry_bridge16.v1"
    if not args.preflight_only:
        if digest(args.preflight / "complete.json") != args.expected_preflight_hash:
            raise ValueError("same-source preflight identity differs")
        declaration = json.loads((args.preflight / "declaration.json").read_text())
        proof = json.loads((args.preflight / "complete.json").read_text())
        if (
            declaration["source_head"] != head
            or declaration["schema"] != schema
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
        schema=schema,
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
    if args.study == "matched_entry":
        for key in (
            "iterations_per_law",
            "new_samples_per_iteration",
            "initial_std_m",
            "minimum_std_m",
            "maximum_draws_per_batch",
            "pose_phase_and_body_unchanged",
        ):
            declaration.pop(key)
        declaration.update(
            maximum_candidates_per_course=12,
            full_bank_native_upper_bound=585600,
            matched_modes=["matched_position", "matched_velocity"],
            entry_interval_domain=[10, 50],
            modified_joint_count=23,
            maximum_joint_reference_correction_rad=0.12,
            new_modes_zero_C_only=True,
            physical_body_and_history_handoff_unchanged=True,
            source_playback_unstretched=True,
            publicity_license_authorized=False,
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
            print("FULL16_" + args.study.upper() + "_ADMITTED_ZERO_ACTORS", flush=True)
            return
        import numpy as np

        from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
        from scripts.rsi_probe_contact_geometry import run_course

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
        kwargs = {}
        if args.study == "response_cem":
            from scripts.rsi_response_cem_builder import build_candidates

            builder = build_candidates
        else:
            import mujoco

            from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
            from rosclaw_soccer.sim.contracts import hash_json
            from scripts.rsi_matched_entry_builder import choose_matched_candidate, make_builder

            model = mujoco.MjModel.from_binary_path(str(args.world))
            limits = model.jnt_range[[model.joint(n).id for n in G1_DDS_JOINT_NAMES]].copy()
            builder = make_builder(poses, limits, hash_json(declaration))
            kwargs = dict(
                forecast_budget=12, final_choice=choose_matched_candidate, label="MATCHED_ENTRY"
            )
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
                        candidate_builder=builder,
                        **kwargs,
                    )
                )
        expected = (
            (lambda r: len(r["candidates"]) == 30 and len(r["proposals"]) == 6)
            if args.study == "response_cem"
            else (
                lambda r: (
                    6 <= len(r["candidates"]) <= 12
                    and len(r["candidates"]) % 2 == 0
                    and len(r["proposals"]) == 3
                )
            )
        )
        if len(results) != 4 or not all(expected(r) for r in results):
            raise ValueError("whole fixed four-course declared-candidate cluster required")
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
                cem_distribution_updates=24 if args.study == "response_cem" else 0,
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
