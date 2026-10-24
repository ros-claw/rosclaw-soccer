"""Once-only declared whole-native replay after a complete teacher pilot.

Resolve the newly written receipt hash only after its source/config/ancestry
commitments match the preregistered pilot. No retry or scientific selection.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rosclaw_soccer.rsi.joint_delivery_teacher import pilot_config
from scripts.rsi_native_context_recovery_common import digest


def audit_command(
    *,
    source_root: Path,
    producer_head: str,
    expert_own_root: Path,
    expert_own_hash: str,
    state_id: int,
    expected_head: str,
    output: Path,
) -> list[str]:
    rp, dp = source_root / "complete.json", source_root / "declaration.json"
    if (
        output.exists()
        or source_root.is_symlink()
        or any(p.is_symlink() for p in (rp, dp))
        or (source_root / "failed.json").exists()
    ):
        raise ValueError("closed ordinary source and unused whole-own output required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    cfg = pilot_config(state_id)
    if (
        r["status"] != "JOINT_DELIVERY_TEACHER38_COLLECTED_PENDING_ALL_OWN"
        or r["declaration_hash"] != digest(dp)
        or r["config"] != cfg
        or d["config"] != cfg
        or r["input_file_hashes"] != d["input_file_hashes"]
        or any(
            v["source_head"] != producer_head
            or v["expert_own_hash"] != expert_own_hash
            or v["preflight_only"] is not False
            for v in (r, d)
        )
        or Path(r["expert_own_root"]).resolve() != expert_own_root.resolve()
        or len(r["records"]) != 38
        or r["optimizer_updates"] != 0
        or r["counts"]["native_steps"] > cfg["maximum_native_steps"]
        or r["counts"]["actor_calls"] > cfg["maximum_actor_calls"]
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
        or r["Fresh_verified"] is not False
        or r["team_game_verified"] is not False
    ):
        raise ValueError("exact complete unpromoted38-world pilot predecessor required")
    return [
        sys.executable,
        "-u",
        "-m",
        "scripts.rsi_run_joint_delivery_pilot",
        "--expert-own-root",
        str(expert_own_root),
        "--expert-own-hash",
        expert_own_hash,
        "--state-id",
        str(state_id),
        "--expected-head",
        expected_head,
        "--source-root",
        str(source_root),
        "--source-hash",
        digest(rp),
        "--producer-head",
        producer_head,
        "--output",
        str(output),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source-root", "expert-own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("producer-head", "expert-own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    command = audit_command(**vars(a))
    print(
        json.dumps(
            dict(
                stage="STARTING_DECLARED_JOINT_DELIVERY_ALL38_OWN",
                automatic_retry=False,
                scientific_decision_automated=False,
                hardware_authorized=False,
            )
        ),
        flush=True,
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
