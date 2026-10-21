"""One bounded dependency chain, NOT an autonomous scientific agent.

Wait for the declared native Torch experiment to terminate normally, then run
the predeclared comparison and independent replay. Never restart training or
physics on failure, select weights, or promote a policy.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def producer_finished(state):
    if state.get("ActiveState") == "failed" or state.get("Result") not in ("", "success"):
        raise ValueError("declared producer failed; no automatic physics retry")
    if state.get("MainPID") != "0":
        return False
    if state.get("SubState") in ("exited", "dead"):
        if state.get("ExecMainCode") != "1" or state.get("ExecMainStatus") != "0":
            raise ValueError("normal successful producer exit required")
        return True
    return False


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unit", required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--candidate-head", required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen follower code required")
    pins = {str(a.baseline / "complete.json"): a.baseline_hash}
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    declaration = a.candidate / "declaration.json"
    declared = json.loads(declaration.read_text())
    if (
        declared["source_head"] != a.candidate_head
        or declared["inference_backend"] != "torch"
        or len(declared["cases"]) != 16
    ):
        raise ValueError("exact already-declared Torch full16 producer required")
    pins[str(declaration)] = digest(declaration)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            arguments=vars(a)
            | {"candidate": str(a.candidate), "baseline": str(a.baseline), "output": str(a.output)},
            input_file_hashes=pins,
            timeout_seconds=1800,
            scientific_successor_automated=False,
            agent_wakeup_provided_by_this_script=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        deadline = time.monotonic() + 1800
        properties = (
            "ActiveState",
            "SubState",
            "Result",
            "MainPID",
            "ExecMainCode",
            "ExecMainStatus",
        )
        while True:
            state = dict(
                line.split("=", 1)
                for line in subprocess.check_output(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        a.unit,
                        *["--property=" + key for key in properties],
                    ],
                    text=True,
                ).splitlines()
            )
            if producer_finished(state):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded producer wait expired, no physics restart")
            time.sleep(2)
        check_pins(pins)
        complete = a.candidate / "complete.json"
        receipt = json.loads(complete.read_text())
        if (
            receipt["status"] != "PAID_RESEARCH_NATIVE_TORCH_PROBE_NOT_PROMOTED"
            or not receipt["strict_replays_exact"]
            or not receipt["whole_repeats_exact"]
        ):
            raise ValueError("closed native Torch teacher required")
        h = digest(complete)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.rsi_compare_paid_backends",
                "--baseline",
                str(a.baseline),
                "--baseline-hash",
                a.baseline_hash,
                "--candidate",
                str(a.candidate),
                "--candidate-hash",
                h,
                "--expected-head",
                a.expected_head,
                "--output",
                str(a.output / "comparison"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.rsi_audit_paid_teacher",
                "--root",
                str(a.candidate),
                "--hash",
                h,
                "--expected-head",
                a.expected_head,
                "--output",
                str(a.output / "native-audit"),
            ],
            check=True,
        )
        check_pins(pins)
        comparison = json.loads((a.output / "comparison/complete.json").read_text())
        audit = json.loads((a.output / "native-audit/complete.json").read_text())
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_TORCH_FOLLOWUP_CHAIN_CLOSED_NOT_PROMOTED",
                input_file_hashes=pins,
                producer_terminal=state,
                candidate_hash=h,
                comparison_status=comparison["status"],
                independent_audit_status=audit["status"],
                native_audit_hash=digest(a.output / "native-audit/complete.json"),
                comparison_hash=digest(a.output / "comparison/complete.json"),
                agent_wakeup_provided_by_this_script=False,
                scientific_successor_automated=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), goal_complete=False, hardware_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
