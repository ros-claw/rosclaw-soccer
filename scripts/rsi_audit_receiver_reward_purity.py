"""Read each actual receiver sampling prefix without changing historical rewards."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.receiver_reward_purity import (
    receiver_contact_prefix,
    summarize_reward_purity,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    for name in ("dataset", "dataset-audit", "review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "dataset-hash",
        "dataset-audit-hash",
        "review-hash",
        "source-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    dp = a.dataset / "complete.json"
    pins = (
        (dp, a.dataset_hash),
        (a.dataset_audit, a.dataset_audit_hash),
        (a.review, a.review_hash),
    )
    if (
        len(a.pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in pins)
    ):
        raise ValueError("all independently audited inputs and unused external output required")
    dataset, dataset_audit, review = (json.loads(p.read_text()) for p, _ in pins)
    if (
        dataset.get("status") != "RECEIVER_ONPOLICY_PHYSICAL_EPISODES_NOT_TRAINED"
        or dataset.get("review_hash") != a.review_hash
        or dataset_audit.get("status") != "RECEIVER_ONPOLICY_READBACK_AND_DENSITY_NOT_TRAINED"
        or dataset_audit.get("dataset_hash") != a.dataset_hash
        or dataset_audit.get("review_hash") != a.review_hash
        or dataset_audit.get("exact_inputs_actions_rewards_and_source_replayed") is not True
        or review.get("status") != "RELAY_RECEIVER_EXPLORATION_PAIRED_NOT_TRAINED"
        or review.get("own_physics_neural_and_density_audited") is not True
        or any(
            d.get("hardware_authorized") is not False or d.get("promotion_authorized") is not False
            for d in (dataset, dataset_audit, review)
        )
    ):
        raise ValueError("exact replayed and density-audited receiver data required")
    sources = {x["block"]: x for x in review["source_pins"]}
    rows = {(r["context"], r["side"], r["seed"]): r for r in dataset["records"]}
    if len(sources) != 3 or len(rows) != 72:
        raise ValueError("complete sampled conditions and source identities required")
    result, seen, blocks, frozen = [], set(), set(), []
    for root_name, audit_name in a.pair:
        root, audit = Path(root_name), Path(audit_name)
        declaration = json.loads((root / "declaration.json").read_text())
        block = declaration["development_context_block"]
        if block in blocks or block not in sources:
            raise ValueError("unique reviewed source blocks required")
        blocks.add(block)
        pin = sources[block]
        source, _, sh = report(
            root,
            audit,
            pin["own_audit_hash"],
            "NEURAL_RELAY_RECEIVER_EXPLORATION32_PENDING_OWN_REPLAY",
            worlds=32,
        )
        if (
            source["source_head"] != a.source_head
            or sh != pin["source_hash"]
            or digest(root / "declaration.json") != source["declaration_hash"]
        ):
            raise ValueError("exact original source and declaration required")
        for trial in source["records"]:
            seed = trial["course"]["arm"]["exploration_seed"]
            if seed is None:
                continue
            key = (trial["course"]["context"], trial["course"]["lateral_m"], seed)
            if key in seen or key not in rows:
                raise ValueError("unique sampled condition and seed required")
            seen.add(key)
            row = rows[key]
            physics, velocity = bind(root, trial, "physics"), bind(root, trial, "receiver_velocity")
            if (
                physics is None
                or velocity is None
                or row["source_block"] != block
                or row["source_trial"] != trial["index"]
                or row["actual"] != measured(root, trial)
                or digest(physics) != row["source_physics_hash"]
                or digest(velocity) != row["source_velocity_hash"]
            ):
                raise ValueError("exact original actual sampled trajectory required")
            with (
                np.load(physics, allow_pickle=False) as f,
                np.load(velocity, allow_pickle=False) as v,
            ):
                n = len(v["control"])
                if (
                    n != row["controls"]
                    or not np.array_equal(v["control"], np.arange(n))
                    or not v["SAM_active"].all()
                ):
                    raise ValueError("complete actual sampled control prefix required")
                contact = receiver_contact_prefix(f["ball_forces"], n)
            if (
                contact["nonfoot_contact_over_1n"]
                != row["receiver_nonfoot_contact_during_sampling"]
            ):
                raise ValueError("recorded contamination label differs from raw physics")
            actual = row["actual"]
            expected_reward = (
                2.0 * actual["return_0p1m_and_healthy"]
                + actual["handoff"]
                + 0.25 * actual["outgoing_foot_strike"]
                - 2.0 * actual["body_failure"]
            )
            if row["terminal_reward"] != expected_reward:
                raise ValueError("historical terminal reward must not change")
            result.append(
                dict(
                    context=key[0],
                    side=key[1],
                    seed=seed,
                    handoff=actual["handoff"],
                    precise_healthy=actual["return_0p1m_and_healthy"],
                    reward=expected_reward,
                    physics_hash=digest(physics),
                    **contact,
                )
            )
        frozen.extend(((root / "complete.json", sh), (audit, pin["own_audit_hash"])))
    if (
        blocks != set(sources)
        or seen != set(rows)
        or any(digest(p) != h for p, h in (*pins, *frozen))
    ):
        raise ValueError("all unchanged source physical histories required")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_SAMPLING_PREFIX_REWARD_PURITY_READBACK",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            dataset_audit_hash=a.dataset_audit_hash,
            review_hash=a.review_hash,
            episodes=len(result),
            summary=summarize_reward_purity(result),
            records=result,
            no_nonfoot_after_prefix_proven=False,
            historical_rewards_modified=False,
            independent_new_physics=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
