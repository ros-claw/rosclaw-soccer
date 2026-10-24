"""Admit ALL independently replayed own stochastic trajectories before PPO."""

import json
from pathlib import Path

from scripts.rsi_native_context_recovery_common import digest


def admit(source_root, own_root, own_hash, state):
    from rosclaw_soccer.rsi.stochastic_refinement_pilot import pilot_config, verify_study
    from scripts.rsi_joint_delivery_pilot_common import admit as admit_behavior

    sp, op, dp = (
        source_root / "complete.json",
        own_root / "complete.json",
        source_root / "declaration.json",
    )
    if (
        any(r.is_symlink() or (r / "failed.json").exists() for r in (source_root, own_root))
        or any(
            p.is_symlink() or not p.is_file() or p.stat().st_size > 128 * 1024 * 1024
            for p in (sp, op, dp)
        )
        or digest(op) != own_hash
    ):
        raise ValueError("closed exact own-stochastic source/audit pair required")
    source, own, declared = [json.loads(p.read_text()) for p in (sp, op, dp)]
    cfg = pilot_config(state)
    if (
        source["status"] != "OWN_STOCHASTIC_REFINEMENT24_COLLECTED_PENDING_NATIVE_LAW_AUDIT"
        or own["status"] != "OWN_STOCHASTIC_REFINEMENT24_ALL_NATIVE_PRIVATE_RNG_LAW_MATCH"
        or own["source_hash"] != digest(sp)
        or own["config"] != source["config"]
        or source["config"] != cfg
        or own["source_head"] != source["source_head"]
        or source["source_head"] != declared["source_head"]
        or own["independent_whole_world_replay"] is not True
        or own["sampling_law_audit_passed"] is not True
        or own["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
        or own["counts"] != source["counts"]
        or source["declaration_hash"] != digest(dp)
        or declared["preflight_only"] is not False
        or declared["config"] != cfg
        or source["expert_own_hash"] != own["expert_own_hash"]
        or source["expert_own_hash"] != declared["expert_own_hash"]
        or source["input_file_hashes"] != declared["input_file_hashes"]
        or any(
            v[k] is not False
            for v in (source, own)
            for k in (
                "actor_training_admitted",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "Fresh_verified",
                "team_game_verified",
            )
        )
        or source["optimizer_updates"] != 0
        or own["optimizer_updates"] != 0
    ):
        raise ValueError("all24 actual private native/law histories required before training")
    pins, scope, bank, pf, behavior_root, baseline = admit_behavior(
        Path(declared["expert_own_root"]), declared["expert_own_hash"], cfg
    )
    courses = [baseline["course"], *source["preflight"]["validation_courses"]]
    verify_study(source, courses)
    for k, v in own["input_file_hashes"].items():
        if k in pins and pins[k] != v:
            raise ValueError("conflicting experience ancestry")
        pins[k] = v
    for p in (sp, op, dp):
        pins[str(p)] = digest(p)
    for row in source["records"]:
        if set(row["artifacts"]) != {"initial", "physical", "A", "B", "sampling"}:
            raise ValueError("all raw stochastic primitives required")
        for a in row["artifacts"].values():
            p = source_root / a["path"]
            if (
                p.is_symlink()
                or p.resolve().parent != (source_root / f"trial{row['index']}").resolve()
                or digest(p) != a["hash"]
            ):
                raise ValueError("exact confined own-executed primitive required")
            pins[str(p)] = a["hash"]
    return source, own, pins, scope, bank, pf, behavior_root


def episodes(source_root, source, scale, *, advantage_mode="zero-v1"):
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.refinement_onpolicy_learning import (
        joint_delivery_reward,
        observed_terminal_returns,
    )

    result = []
    metadata = (
        "controller_kind",
        "sampling_law",
        "assisted",
        "external_teacher",
        "authority_rejected",
        "all_actor_actions_actually_executed",
        "natural_terminal_observed",
        "termination",
        "motion_reference_controls",
        "observed_recovery_controls",
        "total_observed_controls",
        "deterministic_recovery_excluded_from_actor_rows",
    )
    for r in source["records"]:
        with np.load(source_root / r["artifacts"]["A"]["path"], allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        with np.load(source_root / r["artifacts"]["sampling"]["path"], allow_pickle=False) as z:
            s = {k: z[k].copy() for k in z.files}
        n, total = r["stochastic_controls"], r["controls"]
        if (
            s["active"].dtype != np.bool_
            or s["active"].shape != (total,)
            or not s["active"][:n].all()
            or s["active"][n:].any()
            or not np.array_equal(a["standing"], ~s["active"])
        ):
            raise ValueError("only contiguous actually sampled motion rows are actor rows")
        if any(len(v) != total for v in a.values()) or any(len(v) != total for v in s.values()):
            raise ValueError("all complete motion/recovery evidence required")
        reward = joint_delivery_reward(r["outcome"])
        returns = observed_terminal_returns(n, r["stand_controls"], reward)
        row = {k: r[k] for k in metadata}
        row.update(
            index=r["index"],
            terminal_reward=reward,
            observations=torch.from_numpy(a["NN_obs"][:n]),
            old_means=torch.from_numpy(a["NN_action"][:n]),
            actions=torch.from_numpy(a["executed_action"][:n]),
            reference_nominal=torch.from_numpy(a["NN_reference_nominal"][:n]),
            action_scale=scale.clone(),
            old_logp=torch.from_numpy(s["logp"][:n]),
            old_values=torch.zeros(n),
            returns=torch.from_numpy(returns),
            advantages=torch.from_numpy(returns.copy()),
        )
        result.append(row)
    if advantage_mode == "peer-loo-v2":
        from rosclaw_soccer.rsi.trajectory_peer_baseline import peer_return_baselines

        groups = [r["artifacts"]["initial"]["hash"] for r in source["records"]]
        from collections import Counter

        if sorted(Counter(groups).values()) != [8, 8, 8]:
            raise ValueError("exact3 physical initial x8 independent rollouts required for peers")
        baselines, advantages = peer_return_baselines([r["returns"] for r in result], groups)
        for row, baseline, advantage, group in zip(
            result, baselines, advantages, groups, strict=True
        ):
            row.update(advantages=advantage, peer_baseline=baseline, initial_state_hash=group)
    elif advantage_mode != "zero-v1":
        raise ValueError("explicit declared advantage derivation required")
    values = torch.cat([r["advantages"] for r in result])
    mean, std = values.mean(), values.std(unbiased=False).clamp_min(1e-6)
    for r in result:
        r["advantages"] = (r["advantages"] - mean) / std
    return result
