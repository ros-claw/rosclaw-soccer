"""New-experiment admission; never weakens the older recovery byte contract."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path


def admit(args):
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("sealed source and unused output required")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts"], text=True
    ).splitlines()
    if any(p.endswith(".py") for p in untracked):
        raise ValueError("unsealed Python source refused")
    if digest(args.bank / "complete.json") != args.expected_bank_hash:
        raise ValueError("original zero-actor bank identity differs")
    bank = json.loads((args.bank / "declaration.json").read_text())
    proof = json.loads((args.bank / "complete.json").read_text())
    if (
        bank["bank_hash"] != proof["bank_hash"]
        or proof["new_native_steps"] != 0
        or proof["actual_actor_allocations"] != 0
        or proof["new_model_calls"] != 0
        or proof["source_head"] != bank["source_head"]
        or proof["declaration_hash"] != digest(args.bank / "declaration.json")
    ):
        raise ValueError("original full-bank zero-actor proof required")
    pins = dict(bank["input_file_hashes"])
    check_pins(pins)
    pins.update({str(p): digest(p) for p in args.bank.iterdir() if p.is_file()})
    for p in [args.world, args.motion_csv] + [
        args.model_root / s
        for s in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if pins.get(str(p)) != digest(p):
            raise ValueError("original world/motion/foundation identity differs")
    previous = {}
    for root in args.previous_producer:
        p = root / "complete.json"
        if pins.get(str(p)) != digest(p):
            raise ValueError("all original teacher comparison artifacts must be bound")
        complete = json.loads(p.read_text())
        if complete["cluster"] in previous:
            raise ValueError("unique original teacher clusters required")
        previous[complete["cluster"]] = (root, complete)
        pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
    if set(previous) != set(range(4)):
        raise ValueError("all four original clusters required")
    rejected = set()
    for unit, path, expected_hash in args.rejected_producer:
        root = Path(path)
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
        if (
            terminal != dict(MainPID="0", Result="success", SubState="exited", ExecMainStatus="0")
            or digest(root / "complete.json") != expected_hash
        ):
            raise ValueError("preceding experiment must be closed and authenticated")
        d = json.loads((root / "complete.json").read_text())
        if (
            d["cluster"] in rejected
            or d["source_head"] != args.expected_rejected_head
            or any(root.rglob("failure.json"))
        ):
            raise ValueError("unique complete preceding geometry clusters required")
        old = json.loads((root / "declaration.json").read_text())
        check_pins(old["input_file_hashes"])
        pins.update(old["input_file_hashes"])
        pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
        rejected.add(d["cluster"])
    if rejected != set(range(4)):
        raise ValueError("all four preceding geometry terminal proofs required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    protocol = (
        Path.cwd() / "docs/rsi/contact-geometry-result-and-response-cem-next-20261020.zh-CN.md"
    )
    pins[str(protocol)] = digest(protocol)
    # Project imports only after all sources and assets have been pinned.
    from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = close_contact_perturbations()
    if (
        preflight_receiving_courses(courses) != bank["bank_hash"]
        or [asdict(c) | {"course": list(c.course)} for c in courses] != bank["courses"]
    ):
        raise ValueError("whole16 original course bank required before actor allocation")
    return bank, courses, previous, pins
