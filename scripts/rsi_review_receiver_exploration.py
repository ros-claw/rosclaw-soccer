"""Compare audited seeded receiver motor exploration with zero-identity parents."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.relay_follow_comparison import pair_metrics
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report
from scripts.rsi_review_whole_receiver_identity import handoff_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("ROOT", "AUDIT", "IDENTITY", "IDENTITY_AUDIT"),
    )
    p.add_argument("--identity-review", type=Path, required=True)
    p.add_argument("--identity-review-hash", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--identity-producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.group) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.identity_review.is_symlink()
        or digest(a.identity_review) != a.identity_review_hash
    ):
        raise ValueError("three exact paired studies and qualified identity required")
    identity = json.loads(a.identity_review.read_text())
    if (
        identity.get("status") != "RELAY_WHOLE_RECEIVER_ZERO_IDENTITY_QUALIFIED"
        or identity.get("identity_passed") is not True
        or identity.get("conditions") != 24
        or identity.get("own_physics_and_neural_replay_complete") is not True
        or any(
            identity.get(k) is not False
            for k in ("trained", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("strict own-replayed untrained zero identity required")
    blocks, rows, frozen = set(), [], []
    for sr, ar, ir, ia in a.group:
        root, own, old, old_own = map(Path, (sr, ar, ir, ia))
        dec = json.loads((root / "declaration.json").read_text())
        old_dec = json.loads((old / "declaration.json").read_text())
        block = dec["development_context_block"]
        if (
            block in blocks
            or block not in (0, 1, 2)
            or old_dec["development_context_block"] != block
        ):
            raise ValueError("matching distinct context blocks required")
        blocks.add(block)
        ah, ih = digest(own), digest(old_own)
        source, _, sh = report(
            root, own, ah, "NEURAL_RELAY_RECEIVER_EXPLORATION32_PENDING_OWN_REPLAY", worlds=32
        )
        prior, _, ph = report(
            old, old_own, ih, "NEURAL_RELAY_WHOLE_RECEIVER_IDENTITY16_PENDING_OWN_REPLAY", worlds=16
        )
        if (
            source["source_head"] != a.producer_head
            or prior["source_head"] != a.identity_producer_head
            or dec.get("identity_review_hash") != a.identity_review_hash
        ):
            raise ValueError("exact exploration and qualified identity producers required")
        current = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["exploration_seed"],
            ): r
            for r in source["records"]
        }
        parents = {
            (r["course"]["context"], r["course"]["lateral_m"]): r
            for r in prior["records"]
            if r["course"]["arm"]["receiver_whole"] is True
        }
        if len(current) != 32 or len(parents) != 8:
            raise ValueError("complete paired seed factorial required")
        for condition, historical in sorted(parents.items()):
            base = current[(*condition, None)]
            if any(
                base[k] != historical[k]
                for k in ("termination", "controls", "native_steps", "result")
            ) or not handoff_equal(root, base, old, historical):
                raise ValueError("zero-sampling candidate changed the proven parent outcome")
            for name, old_art in historical["artifacts"].items():
                new_art = base["artifacts"][name]
                if (
                    (old_art is None) != (new_art is None)
                    or old_art is not None
                    and old_art["hash"] != new_art["hash"]
                ):
                    raise ValueError("zero-sampling parent trajectory or history changed")
                if old_art is not None:
                    bind(root, base, name)
                    bind(old, historical, name)
            old_metrics = measured(root, base)
            for seed in (92117, 92143, 92171):
                candidate = current[(*condition, seed)]
                if (
                    candidate["artifacts"]["initial"]["hash"]
                    != base["artifacts"]["initial"]["hash"]
                ):
                    raise ValueError("exploration and zero parent must share exact initial physics")
                sampled = bind(root, candidate, "receiver_velocity")
                if sampled is None:
                    raise ValueError("complete actual sampled own receiver history required")
                with np.load(sampled, allow_pickle=False) as f:
                    n = len(f["control"])
                    if (
                        n < 1
                        or f["SAM_executed_action"].shape != (n, 29)
                        or f["SAM_logp"].shape != (n,)
                        or f["SAM_active"].shape != (n,)
                        or not np.isfinite(f["SAM_executed_action"]).all()
                        or not np.isfinite(f["SAM_logp"]).all()
                        or not f["SAM_active"].all()
                        or not np.array_equal(f["control"], np.arange(n))
                    ):
                        raise ValueError(
                            "complete actually sampled bounded receiver controls required"
                        )
                    changed = int(np.any(f["SAM_executed_action"] != f["action"], axis=1).sum())
                    if changed != n:
                        raise ValueError(
                            "every declared stochastic action must be actually sampled"
                        )
                new_metrics = measured(root, candidate)
                rows.append(
                    dict(
                        block=block,
                        context=condition[0],
                        side=condition[1],
                        seed=seed,
                        stochastic_controls=n,
                        changed_controls=changed,
                        baseline=old_metrics,
                        candidate=new_metrics,
                        changes=pair_metrics(old_metrics, new_metrics),
                    )
                )
        frozen.extend(
            ((root / "complete.json", sh), (own, ah), (old / "complete.json", ph), (old_own, ih))
        )
    if (
        blocks != {0, 1, 2}
        or len(rows) != 72
        or digest(a.identity_review) != a.identity_review_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("all 24 paired conditions, 3 seeds and unchanged receipts required")
    summaries = {}
    for seed in (92117, 92143, 92171):
        subset = [r for r in rows if r["seed"] == seed]
        counts = {key: sum(r["changes"][key] for r in subset) for key in subset[0]["changes"]}
        counts["stochastic_controls"] = sum(r["stochastic_controls"] for r in subset)
        counts["candidate_handoffs"] = sum(r["candidate"]["handoff"] for r in subset)
        counts["candidate_precise_healthy_returns"] = sum(
            r["candidate"]["return_0p1m_and_healthy"] for r in subset
        )
        summaries[str(seed)] = counts
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_RECEIVER_EXPLORATION_PAIRED_NOT_TRAINED",
            source_head=a.expected_head,
            identity_review_hash=a.identity_review_hash,
            conditions=24,
            summaries=summaries,
            records=rows,
            zero_parent_byte_identical=True,
            own_physics_neural_and_density_audited=True,
            optimizer_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
