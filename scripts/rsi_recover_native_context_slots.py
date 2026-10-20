"""Fill ONLY21 never-started slots; preserve original faults, controller and physics."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inventory",
        "bank",
        "actor-source",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "fault-review",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-actor-head",
        "expected-inventory-hash",
        "expected-fault-review-hash",
        "fault-review-unit",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--previous-producer", type=Path, action="append", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cluster", type=int, choices=(1, 2, 3))
    parser.add_argument("--recovery-preflight", type=Path)
    parser.add_argument("--expected-recovery-preflight-hash")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise ValueError("sealed recovery source required")
    for name in (
        Path(__file__).name,
        "rsi_native_context_recovery_common.py",
        "rsi_native_context_failure_trace.py",
    ):
        if (
            subprocess.check_output(["git", "show", f"{head}:scripts/{name}"])
            != (Path.cwd() / "scripts" / name).read_bytes()
        ):
            raise ValueError(
                "recovery bootstrap must equal its sealed Git blob before helper imports"
            )

    from scripts.rsi_native_context_recovery_common import (
        admit_inventory,
        cause_chain,
        check_pins,
        digest,
        policy_leaf_reason,
    )

    if args.preflight_only:
        if any(
            x is not None
            for x in (args.cluster, args.recovery_preflight, args.expected_recovery_preflight_hash)
        ):
            raise ValueError("zero-actor preflight cannot allocate a cluster")
    elif any(
        x is None
        for x in (args.cluster, args.recovery_preflight, args.expected_recovery_preflight_hash)
    ):
        raise ValueError("recovery worker requires an explicit cluster and zero-actor proof")
    inventory, original_bank, pins, courses = admit_inventory(args)
    terminal = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                args.fault_review_unit,
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
    if terminal != {
        "MainPID": "0",
        "Result": "success",
        "SubState": "exited",
        "ExecMainStatus": "0",
    }:
        raise ValueError("independent original fault review must be successfully closed")
    if digest(args.fault_review / "complete.json") != args.expected_fault_review_hash:
        raise ValueError("independent fault review identity differs")
    rd = json.loads((args.fault_review / "declaration.json").read_text())
    rc = json.loads((args.fault_review / "complete.json").read_text())
    expected_faults = [
        s["slot"] for s in inventory["slots"] if s["status"] == "FAULT_REQUIRES_REVIEW"
    ]
    if (
        rc["inventory_hash"] != digest(args.inventory)
        or rc["additional_recorded_native_steps"] != 1600
        or rc["additional_real_planner_calls"] != 8
        or [r["slot"] for r in rc["results"]] != expected_faults
        or not all(
            r["all_prefix_physics_bit_exact"]
            and r["all_recorded_planner_outputs_bit_exact"]
            and r["finite_policy_guard_rejection_reproduced"]
            for r in rc["results"]
        )
        or rd["input_file_hashes"].get(str(args.inventory)) != digest(args.inventory)
        or (args.fault_review / "failure.json").exists()
    ):
        raise ValueError("all original fault physics/model/guard proofs required")
    check_pins(rd["input_file_hashes"])
    pins.update(rd["input_file_hashes"])
    pins.update({str(p): digest(p) for p in args.fault_review.rglob("*") if p.is_file()})
    for p in [args.world, args.motion_csv] + [
        args.model_root / name
        for name in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if pins.get(str(p)) != digest(p):
            raise ValueError("same world, motion source and foundation assets required")
    previous = {}
    for root in args.previous_producer:
        p = root / "complete.json"
        if pins.get(str(p)) != digest(p):
            raise ValueError("same bound earlier nominal comparison required")
        d = json.loads(p.read_text())
        if d["cluster"] in previous:
            raise ValueError("unique nominal comparison clusters required")
        previous[d["cluster"]] = (root, d)
    if set(previous) != set(range(4)):
        raise ValueError("all four original comparison clusters required")
    master = None
    if not args.preflight_only:
        if (
            digest(args.recovery_preflight / "complete.json")
            != args.expected_recovery_preflight_hash
        ):
            raise ValueError("recovery master proof identity differs")
        master = json.loads((args.recovery_preflight / "complete.json").read_text())
        md = json.loads((args.recovery_preflight / "declaration.json").read_text())
        if (
            master["source_head"] != head
            or master["new_native_steps"] != 0
            or master["new_actor_allocations"] != 0
            or md["input_file_hashes"] != pins
            or master["pending_slots"] != inventory["unexecuted_slots"]
            or master["declaration_hash"] != digest(args.recovery_preflight / "declaration.json")
            or (args.recovery_preflight / "failure.json").exists()
        ):
            raise ValueError("exact same-source zero-actor recovery commitment required")
        pins.update({str(p): digest(p) for p in args.recovery_preflight.iterdir() if p.is_file()})

    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    pending = [s for s in inventory["slots"] if s["status"] == "NOT_EXECUTED"]
    selected = (
        pending if args.preflight_only else [s for s in pending if s["cluster"] == args.cluster]
    )
    if len(pending) != 21 or (
        not args.preflight_only and len(selected) != {1: 6, 2: 9, 3: 6}[args.cluster]
    ):
        raise ValueError("only exact never-started original slots may be selected")
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.native_context_missing_slot_recovery.v1",
        source_head=head,
        actor_source_head=args.expected_actor_head,
        input_file_hashes=pins,
        inventory_hash=digest(args.inventory),
        bank_hash=inventory["bank_hash"],
        cluster=args.cluster,
        pending_slots=inventory["unexecuted_slots"],
        selected_slots=[s["slot"] for s in selected],
        original_fault_slots_never_retried=expected_faults,
        original_charged_native_steps=73600,
        total_planned_new_native_upper_bound=63000,
        actor_control_source_byte_exact=True,
        new_control_parameters_or_exam=False,
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
                    declaration_hash=digest(args.output / "declaration.json"),
                    pending_slots=inventory["unexecuted_slots"],
                    new_native_steps=0,
                    new_actor_allocations=0,
                    new_model_calls=0,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
            )
            print("ONLY21_MISSING_SLOTS_ADMITTED_ZERO_ACTORS", flush=True)
            return
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
        from scripts.rsi_native_context_failure_trace import validate_failure_trace
        from scripts.rsi_probe_measured_context_native import run_arm

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
            raise ValueError("original affordance identity differs")
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        model = mujoco.MjModel.from_binary_path(str(args.world))
        for slot in selected:
            check_pins(pins)
            # Old actor path checked before imports. No old failed instance is resumed.
            item = next(
                c
                for c in courses
                if c.parent_index == slot["cluster"] and c.variant_index == slot["variant"]
            )
            root, record = previous[item.parent_index]
            nominal = record["results"][item.variant_index]["results"][0]
            with np.load(
                root / f"variant{item.variant_index}" / nominal["artifact"], allow_pickle=False
            ) as a:
                gold = {k: a[k] for k in a.files}
            original_declaration = json.loads(
                (Path(slot["source_artifact_dir"]).parent / "declaration.json").read_text()
            )
            folder = args.output / f"variant{item.variant_index}-{slot['arm']}"
            if folder.exists():
                raise ValueError("each missing slot may be attempted once only")
            try:
                row = run_arm(
                    args,
                    item,
                    slot["arm"],
                    gold,
                    poses,
                    affordance,
                    hash_json(original_declaration),
                )
            except Exception as error:
                reason = policy_leaf_reason(error)
                if reason is None:
                    raise
                proof = validate_failure_trace(folder, model, reason)
                row = dict(
                    slot=slot["slot"],
                    status="POLICY_REFERENCE_REJECTED",
                    cause_chain=cause_chain(error),
                    trace_validation=proof,
                    native_steps=proof["native_steps"],
                    full_h300_quality_observed=False,
                    task_qualified=False,
                    old_actor_or_fault_retried=False,
                    partial_failure_hash=digest(folder / "failure.json"),
                    promotion_authorized=False,
                    hardware_authorized=False,
                )
            else:
                row = dict(
                    slot=slot["slot"],
                    status="COMPLETE",
                    native_steps=row["actual_integrated_native_steps"],
                    full_h300_quality_observed=True,
                    producer_claimed_qualified=row["score"]["qualified"],
                    actor_complete_hash=digest(folder / "complete.json"),
                    old_actor_or_fault_retried=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                )
            row["artifact_hashes"] = {
                str(p.relative_to(args.output)): digest(p) for p in folder.iterdir() if p.is_file()
            }
            results.append(row)
            write_once(args.output / f"slot-v{item.variant_index}-{slot['arm']}.json", row)
            print(
                "MISSING_SLOT_TERMINAL",
                slot["slot"],
                row["status"],
                row["native_steps"],
                flush=True,
            )
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                source_head=head,
                inventory_hash=digest(args.inventory),
                cluster=args.cluster,
                new_native_steps=sum(r["native_steps"] for r in results),
                new_full_h300_cases=sum(r["status"] == "COMPLETE" for r in results),
                new_policy_reference_rejections=sum(
                    r["status"] == "POLICY_REFERENCE_REJECTED" for r in results
                ),
                remaining_selected_slots=0,
                original_cases_retried=0,
                optimizer_steps=0,
                workflow_complete_not_physical_candidate_qualified=True,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json",
            dict(
                error=repr(error),
                cause_chain=cause_chain(error),
                status="FATAL_CONTRACT_VIOLATION",
                completed_slot_receipts=len(results),
                completed_slot_native_steps=sum(r["native_steps"] for r in results),
                incomplete_slot_counts_in_actor_failure=True,
                unexecuted_slots_are_not_verified=True,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
