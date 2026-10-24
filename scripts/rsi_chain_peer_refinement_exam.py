"""Admit a closed peer/gen1 PPO producer, then run the paired native exam.

Receipt hash is derived only after the exact producer closes. Output remains
once-only and the downstream strict loader checks actual checkpoint/data/state.
This command never retries a failed producer or changes learned weights.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--training-producer-head", required=True)
    p.add_argument("--continual", action="store_true")
    p.add_argument("--continuing", action="store_true")
    p.add_argument("--full-recovery", action="store_true")
    a, rest = p.parse_known_args()
    if sum((a.continual, a.continuing, a.full_recovery)) > 1:
        p.error("choose one exact continuation checkpoint family")
    if "--training-hash" in rest:
        p.error("derived exact predeclared producer hash must not be overridden")
    from rosclaw_soccer.rsi import continuing_refinement_contract as continuing
    from rosclaw_soccer.rsi import refinement_continual_checkpoint as continual
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from rosclaw_soccer.rsi.refinement_onpolicy_learning import PEER_KIND, STATUS
    from scripts.rsi_run_refinement_paired_exam import main as exam

    tp = a.training_root / "complete.json"
    if a.training_root.is_symlink() or (a.training_root / "failed.json").exists():
        raise ValueError("ordinary closed predeclared producer required")
    exact_hash = file_hash(tp)
    trained = json.loads(tp.read_text())
    kind, status = (
        (continuing.FULL_KIND, continuing.FULL_STATUS)
        if a.full_recovery
        else (
            (continuing.KIND, continuing.STATUS)
            if a.continuing
            else ((continual.KIND, continual.STATUS) if a.continual else (PEER_KIND, STATUS))
        )
    )
    if (
        trained["source_head"] != a.training_producer_head
        or trained["checkpoint_kind"] != kind
        or trained["status"] != status
        or trained["new_actor_steps"] < 1
    ):
        raise ValueError("exact actually updated peer-PPO producer required")
    sys.argv = [
        sys.argv[0],
        *rest,
        "--training-root",
        str(a.training_root),
        "--training-hash",
        exact_hash,
    ]
    exam()


if __name__ == "__main__":
    main()
