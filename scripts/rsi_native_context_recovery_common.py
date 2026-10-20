"""Admission/data utilities; importing this file never allocates an actor."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

POLICY_REASONS = frozenset(
    (
        "original planner context outside native joint limits; no repair",
        "proposed measured-joint context outside native joint limits; no clipping",
    )
)
ARMS = ("baseline", "blend_only", "measured_blend")


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def check_pins(pins):
    for p, h in pins.items():
        if digest(Path(p)) != h:
            raise ValueError("pinned source/assets/evidence changed: " + p)


def cause_chain(error):
    chain = []
    seen = set()
    while error is not None:
        if id(error) in seen or len(chain) >= 16:
            raise ValueError("cyclic or oversized exception cause chain")
        seen.add(id(error))
        chain.append(dict(type=type(error).__name__, message=str(error)))
        error = error.__cause__
    return chain


def policy_leaf_reason(error):
    cause_chain(error)
    while error.__cause__ is not None:
        error = error.__cause__
    return str(error) if type(error) is ValueError and str(error) in POLICY_REASONS else None


def admit_inventory(args):
    """Check all sources BEFORE project imports; then admit the complete bank."""
    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("sealed new source and unused output required")
    if digest(args.inventory) != args.expected_inventory_hash:
        raise ValueError("48-slot inventory identity differs")
    inventory = json.loads(args.inventory.read_text())
    if (
        inventory["schema"] != "soccer.rsi.native_ablation_recovery_inventory.v1"
        or inventory["source_actor_head"] != args.expected_actor_head
        or subprocess.check_output(
            ["git", "-C", str(args.actor_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != args.expected_actor_head
        or inventory["actual_charged_native_steps"] != 73600
    ):
        raise ValueError("exact sealed actor source and full preserved inventory required")
    bank = json.loads((args.bank / "declaration.json").read_text())
    if (
        inventory["input_file_hashes"].get(str(args.bank / "declaration.json"))
        != digest(args.bank / "declaration.json")
        or bank["source_head"] != args.expected_actor_head
        or bank["bank_hash"] != inventory["bank_hash"]
    ):
        raise ValueError("original source-bound zero-actor bank required")
    pins = {
        **bank["input_file_hashes"],
        **inventory["input_file_hashes"],
        str(args.inventory): digest(args.inventory),
    }
    check_pins(pins)
    for folder in ("src", "scripts"):
        for old in (args.actor_source / folder).rglob("*.py"):
            current = Path.cwd() / old.relative_to(args.actor_source)
            if digest(current) != digest(old):
                raise ValueError("recovery cannot change any existing actor/helper module")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source):
        for p in folder.rglob("*.py"):
            pins[str(p)] = digest(p)
    for unit, expected in inventory["producer_terminals"].items():
        actual = dict(
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
        if actual != expected or actual["MainPID"] != "0":
            raise ValueError("original terminal producer changed or is live")

    from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = close_contact_perturbations()
    if preflight_receiving_courses(courses) != inventory["bank_hash"]:
        raise ValueError("full16-course admission differs")
    expected = [(item, arm) for item in courses for arm in ARMS]
    if len(inventory["slots"]) != 48:
        raise ValueError("all48 original slots required")
    counts = dict(COMPLETE=0, FAULT_REQUIRES_REVIEW=0, NOT_EXECUTED=0)
    native = 0
    for row, (item, arm) in zip(inventory["slots"], expected, strict=True):
        condition = asdict(item) | {"course": list(item.course)}
        if (
            row["slot"] != f"c{item.parent_index}:v{item.variant_index}:{arm}"
            or row["cluster"] != item.parent_index
            or row["variant"] != item.variant_index
            or row["arm"] != arm
            or row["course"] != condition
            or type(row["native_steps"]) is not int
            or row["native_steps"] < 0
        ):
            raise ValueError("exact ordered condition/arm/slot identity required")
        folder = Path(row["source_artifact_dir"])
        declaration_path = folder.parent / "declaration.json"
        if str(declaration_path) not in inventory["input_file_hashes"]:
            raise ValueError("slot must belong to a bound original cluster")
        declaration = json.loads(declaration_path.read_text())
        if declaration["cluster"] != item.parent_index or declaration["courses"] != bank["courses"]:
            raise ValueError("original cluster course commitment differs")
        if folder.name != f"variant{item.variant_index}-{arm}":
            raise ValueError("slot path differs")
        status = row["status"]
        if status not in counts:
            raise ValueError("unknown slot status")
        counts[status] += 1
        native += row["native_steps"]
        if status == "NOT_EXECUTED":
            if folder.exists() or row["native_steps"] != 0:
                raise ValueError("missing slot may not hide an attempted directory")
        else:
            name = "complete.json" if status == "COMPLETE" else "failure.json"
            if str(folder / name) not in inventory["input_file_hashes"]:
                raise ValueError("existing slot requires a pinned terminal receipt")
    if (
        counts != {"COMPLETE": 24, "FAULT_REQUIRES_REVIEW": 3, "NOT_EXECUTED": 21}
        or counts != inventory["status_counts"]
        or native != 73600
    ):
        raise ValueError("original complete/fault/missing accounting differs")
    if inventory["unexecuted_slots"] != [
        s["slot"] for s in inventory["slots"] if s["status"] == "NOT_EXECUTED"
    ]:
        raise ValueError("pending commitment differs")
    return inventory, bank, pins, courses
