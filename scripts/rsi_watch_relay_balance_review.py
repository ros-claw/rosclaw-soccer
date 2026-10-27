"""After every physical relay balance replay closes, run complete paired review."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        nargs=5,
        action="append",
        required=True,
        metavar=("CURRENT", "AUDIT", "AUDIT_UNIT", "PRIOR", "PRIOR_AUDIT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    for name in ("expected-head", "source-head", "prior-head"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    units = [group[2] for group in args.group]
    if (
        len(args.group) != 3
        or len(set(units)) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= args.timeout_sec <= 43200
        or any(
            not u.startswith("rosclaw-team-") or not u.endswith(".service") or "/" in u
            for u in units
        )
    ):
        raise ValueError("three bounded SIM units and unused review output required")
    deadline = time.monotonic() + args.timeout_sec
    admitted = None
    while admitted is None:
        pending, groups = False, []
        for current, audit_root, unit, prior, prior_audit_root in args.group:
            query = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "-p",
                    "ActiveState",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            state = dict(line.split("=", 1) for line in query.stdout.splitlines() if "=" in line)
            if query.returncode or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys():
                raise ValueError("replay process cannot be verified")
            if state["ActiveState"] == "active":
                pending = True
                continue
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("failed replay: paired review blocked")
            cp, ap, pp, pap = (
                Path(p) / "complete.json" for p in (current, audit_root, prior, prior_audit_root)
            )
            if any(not p.is_file() or p.is_symlink() for p in (cp, ap, pp, pap)):
                raise ValueError("all complete immutable sources required")
            c, a, p, pa = (json.loads(x.read_text()) for x in (cp, ap, pp, pap))
            if (
                c.get("source_head") != args.source_head
                or p.get("source_head") != args.prior_head
                or c.get("status")
                != "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                or p.get("status") != "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                or any(
                    v.get("status") != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
                    for v in (a, pa)
                )
                or a.get("producer_hash") != digest(cp)
                or pa.get("producer_hash") != digest(pp)
            ):
                raise ValueError("exact source and own replay identity required")
            groups.append((current, str(ap), prior, str(pap), digest(ap), digest(pap)))
        if not pending:
            admitted = groups
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded replay wait expired; no retry")
        time.sleep(15)
    command = [
        sys.executable,
        "scripts/rsi_review_relay_balance_counterfactual.py",
        "--output",
        str(args.output),
        "--expected-head",
        args.expected_head,
    ]
    for group in admitted:
        command.extend(("--group", *group))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
