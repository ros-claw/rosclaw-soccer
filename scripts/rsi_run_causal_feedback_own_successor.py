"""One declared successor: authenticate terminal TRAIN10 then replay all worlds.

Not an open-ended agent scheduler. No retry, training launch, physics allocation
or policy promotion occurs here. The own runner independently checks primitives.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rosclaw_soccer.rsi.causal_feedback_pilot import pilot_config
from scripts.rsi_native_context_recovery_common import digest

# Exact source-pinned ancestry can exceed16MiB. Keep a finite allocation bound;
# this raises no world/action budget and does not bypass any source validation.
MAX_DOCUMENT_BYTES = 128 * 1024 * 1024


def resolve_source(
    root: Path,
    *,
    producer_head: str,
    expert_root: Path,
    expert_hash: str,
    state: int,
    anticipatory: bool = False,
) -> str:
    cfg = pilot_config(state, anticipatory=anticipatory)
    cp, dp = root / "complete.json", root / "declaration.json"
    if (
        root.is_symlink()
        or (root / "failed.json").exists()
        or any(
            p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_DOCUMENT_BYTES
            for p in (cp, dp)
        )
    ):
        raise ValueError("closed bounded ordinary causal feedback source required")
    source, declaration = [json.loads(p.read_text()) for p in (cp, dp)]
    if (
        source["status"] != "CAUSAL_FOOT_BALL_TRAIN10_COLLECTED_PENDING_OWN"
        or source["source_head"] != producer_head
        or declaration["source_head"] != producer_head
        or source["config"] != cfg
        or declaration["config"] != cfg
        or source["expert_own_hash"] != expert_hash
        or declaration["expert_own_hash"] != expert_hash
        or declaration["expert_own_root"] != str(expert_root)
        or declaration["preflight_only"] is not False
        or declaration["source_hash"] is not None
        or source["declaration_hash"] != digest(dp)
        or source["input_file_hashes"] != declaration["input_file_hashes"]
        or len(source["records"]) != 10
        or source["optimizer_updates"] != 0
        or any(
            type(source["counts"].get(k)) is not int or not 0 < source["counts"][k] <= bound
            for k, bound in (("native_steps", 40000), ("actor_calls", 8000))
        )
        or any(
            source[k] is not False
            for k in (
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "Fresh_verified",
                "team_game_verified",
            )
        )
    ):
        raise ValueError("exact predeclared completed10-world feedback study required")
    return digest(cp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "expert-own-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("producer-head", "expert-own-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    p.add_argument("--anticipatory", action="store_true")
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused declared own successor output required")
    source_hash = resolve_source(
        a.source_root,
        producer_head=a.producer_head,
        expert_root=a.expert_own_root,
        expert_hash=a.expert_own_hash,
        state=a.state_id,
        anticipatory=a.anticipatory,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_run_causal_feedback_pilot",
            "--expert-own-root",
            str(a.expert_own_root),
            "--expert-own-hash",
            a.expert_own_hash,
            "--state-id",
            str(a.state_id),
            "--expected-head",
            a.expected_head,
            "--producer-head",
            a.producer_head,
            "--source-root",
            str(a.source_root),
            "--source-hash",
            source_hash,
            "--output",
            str(a.output),
            *(["--anticipatory"] if a.anticipatory else []),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
