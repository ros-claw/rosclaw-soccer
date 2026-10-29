"""Export audited own-action return-player samples, preserving all failed cases."""

import argparse
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
from rosclaw_soccer.rsi.return_player_outcome import (
    discounted_sample_returns,
    return_player_outcome,
)
from rosclaw_soccer.rsi.return_player_sampling_audit import audit_return_player_sampling
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("source-hash", "audit-hash", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    source, _, source_hash = report(
        a.source,
        a.audit,
        a.audit_hash,
        "NEURAL_RELAY_RETURN_PLAYER_OWN_SAMPLING16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    declaration = a.source / "declaration.json"
    if (
        source_hash != a.source_hash
        or source.get("source_head") != a.producer_head
        or digest(declaration) != source["declaration_hash"]
    ):
        raise ValueError("exact pinned producer and source required")
    pins = source["input_file_hashes"]
    configs = [x for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    if len(configs) != 1:
        raise ValueError("unique original action codec required")
    params = VelocityPolicyParameters.load(Path(configs[0]), expected_hash=pins[configs[0]])
    scale = np.asarray(params.action_scale, np.float32)
    staged, records = [], []
    for row in source["records"]:
        if row["course"]["arm"].get("passer_sampling_seed") is None:
            continue
        if row["course"]["arm"]["passer_sampling_seed"] != 94111:
            raise ValueError("declared pilot seed required")
        vp, pp = bind(a.source, row, "passer_velocity"), bind(a.source, row, "physics")
        if vp is None or pp is None:
            raise ValueError("complete own actual control and physical history required")
        with np.load(vp, allow_pickle=False) as v, np.load(pp, allow_pickle=False) as physical:
            n = row["controls"]
            if not np.array_equal(v["control"], np.arange(n)):
                raise ValueError("full world-aligned own controls required")
            audit_result = audit_return_player_sampling(v, scale)
            indices = np.flatnonzero(v["sampling_active"])
            record = dict(
                index=row["index"],
                course=row["course"],
                termination=row["termination"],
                sampled_controls=len(indices),
                source_velocity_hash=digest(vp),
                source_physics_hash=digest(pp),
                outcome=None,
                path=None,
            )
            if len(indices):
                start, end = int(indices[0]), int(indices[-1])
                if physical["ball_forces"].shape[0] != n or physical["body_failed"].shape[0] != n:
                    raise ValueError("complete full-horizon physical outcomes required")
                for i in indices:
                    if not np.array_equal(v["sampling_inputs"][i, :98], v["obs"][i]):
                        raise ValueError("sample must use the player's actual observation")
                    if not np.array_equal(v["sampling_action"][i], v["action_memory_after"][i]):
                        raise ValueError("sample must be the actual committed action")
                    if not np.array_equal(
                        target29(v["sampling_action"][i], params), v["actual_target"][i]
                    ):
                        raise ValueError("sampled action must be the actual physical target")
                outcome = return_player_outcome(
                    physical["ball_forces"], physical["body_failed"], sample_start=start
                )
                values = dict(
                    inputs=v["sampling_inputs"][indices].copy(),
                    reference=v["sampling_reference"][indices].copy(),
                    old_mean=v["sampling_mean"][indices].copy(),
                    sampled_action=v["sampling_action"][indices].copy(),
                    old_logp=v["sampling_logp"][indices].copy(),
                    rng_before=v["sampling_rng_before"][indices].copy(),
                    rng_after=v["sampling_rng_after"][indices].copy(),
                    returns=discounted_sample_returns(outcome, sample_end=end),
                    action_scale=scale,
                )
                record.update(
                    outcome=outcome, sample_end=end, density_error=audit_result["max_logp_error"]
                )
                staged.append((record, values))
            records.append(record)
    if (
        len(records) != 8
        or sum(r["sampled_controls"] for r in records)
        != source["counts"]["passer_sampled_controls"]
    ):
        raise ValueError("all eight sampled arms and actual sample count required")
    if digest(a.source / "complete.json") != a.source_hash or digest(a.audit) != a.audit_hash:
        raise ValueError("sources changed during export")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    for record, values in staged:
        path = a.output / f"episode-{record['index']:02d}.npz"
        np.savez_compressed(path, **values)
        record.update(path=path.name, hash=digest(path))
    write_once(
        a.output / "complete.json",
        dict(
            status="RETURN_PLAYER_OWN_SAMPLED_DATA_RESEARCH_ONLY",
            source_head=a.expected_head,
            producer_head=a.producer_head,
            source_hash=a.source_hash,
            replay_hash=a.audit_hash,
            records=records,
            sampled_episodes=len(staged),
            sampled_controls=sum(r["sampled_controls"] for r in records),
            reward_profile="post-sampling-contact-proxy-full-horizon-v1",
            gamma=0.99,
            sampled_conditions=len(staged),
            independent_replicates_proven=False,
            repeated_seed_across_conditions=True,
            optimizer_authorized=False,
            possession_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
