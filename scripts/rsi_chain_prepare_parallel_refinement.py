"""Bind a closed preregistered96 own-native producer before preparing gen2 data."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--new-own-root", type=Path, required=True)
    p.add_argument("--new-own-producer-head", required=True)
    p.add_argument("--full-recovery", action="store_true")
    a, rest = p.parse_known_args()
    from rosclaw_soccer.rsi.parallel_refinement_batch import FULL_OWN
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from scripts.rsi_prepare_refinement_continual import main as prepare

    root = a.new_own_root
    if root.is_symlink() or (root / "failed.json").exists():
        raise ValueError("ordinary closed96 own producer required")
    path = root / "complete.json"
    h = file_hash(path)
    r = json.loads(path.read_text())
    if (
        r["status"]
        != (FULL_OWN if a.full_recovery else "GEN1_PARALLEL_OWN_POLICY_BATCH_ALL_NATIVE_LAW_MATCH")
        or r["source_head"] != a.new_own_producer_head
        or r["independent_whole_world_replay"] is not True
        or r["sampling_law_audit_passed"] is not True
        or r["config"]["seeds_per_initial"] != 32
    ):
        raise ValueError("exact full new-policy96 own native/RNG producer required")
    sys.argv = [
        sys.argv[0],
        *rest,
        "--parallel-batch",
        *(["--full-recovery"] if a.full_recovery else []),
        "--new-own-root",
        str(root),
        "--new-own-hash",
        h,
    ]
    prepare()


if __name__ == "__main__":
    main()
