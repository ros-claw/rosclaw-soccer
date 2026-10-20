"""Strict 48-slot accounting only, not independent physics/model verification."""


def summarize_slots(rows):
    """Refuse missing/duplicate/wrong identities; retain unobserved fault quality."""
    arms = ("baseline", "blend_only", "measured_blend")
    expected = [f"c{c}:v{v}:{a}" for c in range(4) for v in range(4) for a in arms]
    if len(rows) != 48 or [r["slot"] for r in rows] != expected:
        raise ValueError("exact ordered original48 slots required")
    stats = {}
    for arm in arms:
        selected = [r for r in rows if r["slot"].endswith(":" + arm)]
        for row in selected:
            if row["status"] == "COMPLETE":
                if row["native_steps"] != 3000 or type(row["qualified"]) is not bool:
                    raise ValueError("complete H300 task needs recorded qualification")
            elif row["status"] == "POLICY_REFERENCE_REJECTED":
                if (
                    type(row["native_steps"]) is not int
                    or not 200 <= row["native_steps"] < 3000
                    or row["native_steps"] % 200
                    or row["qualified"] is not None
                ):
                    raise ValueError("partial guard rejection has no H300 quality label")
            else:
                raise ValueError("unexecuted or unknown slots cannot close accounting")
        stats[arm] = dict(
            attempted_slots=16,
            full_h300_cases=sum(r["status"] == "COMPLETE" for r in selected),
            guard_rejected_quality_unobserved=sum(
                r["status"] == "POLICY_REFERENCE_REJECTED" for r in selected
            ),
            producer_claimed_qualified=sum(r["qualified"] is True for r in selected),
            native_steps=sum(r["native_steps"] for r in selected),
        )
    comparisons = {}
    for arm in arms[1:]:
        base = [r for r in rows if r["slot"].endswith(":baseline")]
        changed = [r for r in rows if r["slot"].endswith(":" + arm)]
        comparisons[arm] = dict(
            mutually_complete_pairs=sum(
                a["qualified"] is not None and b["qualified"] is not None
                for a, b in zip(base, changed, strict=True)
            ),
            observed_rescues=sum(
                a["qualified"] is False and b["qualified"] is True
                for a, b in zip(base, changed, strict=True)
            ),
            observed_regressions=sum(
                a["qualified"] is True and b["qualified"] is False
                for a, b in zip(base, changed, strict=True)
            ),
            baseline_success_with_changed_guard_rejection=sum(
                a["qualified"] is True and b["qualified"] is None
                for a, b in zip(base, changed, strict=True)
            ),
        )
    return dict(
        attempted_slots=48,
        by_arm=stats,
        paired_comparisons=comparisons,
        total_integrated_native_steps=sum(r["native_steps"] for r in rows),
        remaining_unexecuted_slots=0,
        producer_claims_only=True,
        independent_complete_physics_review=False,
        optimizer_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )


def main():
    import argparse
    import json
    import subprocess
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "bank", "actor-source", "core-source", "fault-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-actor-head",
        "expected-inventory-hash",
        "expected-fault-review-hash",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--recovery", nargs=2, action="append", required=True, metavar=("UNIT", "ROOT")
    )
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise ValueError("sealed summary source required")
    for name in (Path(__file__).name, "rsi_native_context_recovery_common.py"):
        if (
            subprocess.check_output(["git", "show", f"{head}:scripts/{name}"])
            != (Path.cwd() / "scripts" / name).read_bytes()
        ):
            raise ValueError("summary bootstrap differs from sealed source")
    from scripts.rsi_native_context_recovery_common import admit_inventory, check_pins, digest

    inventory, _, pins, _ = admit_inventory(args)
    if digest(args.fault_review / "complete.json") != args.expected_fault_review_hash:
        raise ValueError("fault review identity differs")
    proof = json.loads((args.fault_review / "complete.json").read_text())
    if proof["inventory_hash"] != digest(args.inventory):
        raise ValueError("fault review belongs to different inventory")
    reviewed = {r["slot"]: r for r in proof["results"]}
    pins.update({str(p): digest(p) for p in args.fault_review.rglob("*") if p.is_file()})
    new = {}
    clusters = set()
    units = set()
    for unit, raw_root in args.recovery:
        root = Path(raw_root)
        terminal = dict(
            line.split("=", 1)
            for line in subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
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
        if terminal != dict(MainPID="0", Result="success", SubState="exited", ExecMainStatus="0"):
            raise ValueError("recovery producer is live or failed")
        if (root / "failure.json").exists():
            raise ValueError("fatal recovery failure cannot close inventory")
        complete = json.loads((root / "complete.json").read_text())
        declaration = json.loads((root / "declaration.json").read_text())
        cluster = complete["cluster"]
        if (
            cluster not in (1, 2, 3)
            or cluster in clusters
            or unit in units
            or complete["inventory_hash"] != digest(args.inventory)
            or complete["source_head"] != declaration["source_head"]
            or declaration["actor_source_head"] != args.expected_actor_head
            or complete["remaining_selected_slots"] != 0
            or complete["original_cases_retried"] != 0
            or complete["optimizer_steps"] != 0
            or [r["slot"] for r in complete["results"]] != declaration["selected_slots"]
            or complete["new_native_steps"] != sum(r["native_steps"] for r in complete["results"])
        ):
            raise ValueError("exact closed recovery cluster commitment required")
        check_pins(declaration["input_file_hashes"])
        pins.update(declaration["input_file_hashes"])
        clusters.add(cluster)
        units.add(unit)
        for row in complete["results"]:
            if row["slot"] in new or row["old_actor_or_fault_retried"]:
                raise ValueError("duplicate or retried original case")
            for relative, expected_hash in row["artifact_hashes"].items():
                path = root / relative
                if (
                    not path.resolve().is_relative_to(root.resolve())
                    or digest(path) != expected_hash
                ):
                    raise ValueError("recovery artifact differs or escapes root")
            new[row["slot"]] = dict(row, source_root=str(root))
        pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
    if clusters != {1, 2, 3} or set(new) != set(inventory["unexecuted_slots"]):
        raise ValueError("all21 missing-only recovery receipts required")
    rows = []
    for slot in inventory["slots"]:
        if slot["status"] == "NOT_EXECUTED":
            item = new[slot["slot"]]
            status = item["status"]
            qualified = item.get("producer_claimed_qualified") if status == "COMPLETE" else None
            steps = item["native_steps"]
            root = item["source_root"]
        else:
            status = slot["status"]
            if status == "FAULT_REQUIRES_REVIEW":
                r = reviewed[slot["slot"]]
                if not all(
                    r[k] is True
                    for k in (
                        "all_prefix_physics_bit_exact",
                        "all_recorded_planner_outputs_bit_exact",
                        "finite_policy_guard_rejection_reproduced",
                    )
                ):
                    raise ValueError("original fault requires exact independent proof")
                status = "POLICY_REFERENCE_REJECTED"
            qualified = slot.get("producer_claimed_qualified") if status == "COMPLETE" else None
            steps = slot["native_steps"]
            root = slot["source_artifact_dir"]
        rows.append(
            dict(
                slot=slot["slot"],
                status=status,
                qualified=qualified,
                native_steps=steps,
                source_root=root,
            )
        )
    summary = summarize_slots(rows)
    check_pins(pins)
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=head,
            input_file_hashes=pins,
            inventory_hash=digest(args.inventory),
            read_only_accounting=True,
        ),
    )
    write_once(args.output / "complete.json", dict(summary, slots=rows))
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
