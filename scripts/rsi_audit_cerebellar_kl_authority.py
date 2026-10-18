"""Analytic same-input torque-change ceilings from the existing PPO KL gates.

This audits learning authority, not physical reachability. Tiny policy changes
can have large closed-loop contact effects; no outcome is inferred here.
"""

import argparse
import math
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi import cerebellar_actor_critic as learner
from rosclaw_soccer.rsi import cerebellar_torque_policy as policy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once


def budget_report():
    """tanh is 1-Lipschitz, and same-covariance Gaussian KL bounds mean L2."""
    config = learner.CONFIG
    sigma = config["latent_std"]
    maximum_cap = float(np.max(policy.CAPS))
    rows = []
    for name in (
        "maximum_rollout_mean_kl",
        "maximum_rollout_state_kl",
        "maximum_anchor_mean_kl",
        "maximum_anchor_state_kl",
    ):
        kl = config[name]
        bound = maximum_cap * sigma * math.sqrt(2 * kl)
        rows.append(
            dict(
                gate=name,
                kl=kl,
                latent_mean_l2_bound=sigma * math.sqrt(2 * kl),
                torque_vector_l2_bound_nm=bound,
                bound_kind="mean_torque_vector_norm"
                if "mean_kl" in name
                else "per_state_torque_vector_norm",
                reference="fixed_initial_actor"
                if "anchor" in name
                else "current_batch_behavior_actor",
            )
        )
    # Tightness/sanity at zero mean along a maximum-cap coordinate, same input.
    maximum = next(row for row in rows if row["gate"] == "maximum_anchor_state_kl")
    latent_delta = maximum["latent_mean_l2_bound"]
    actual = maximum_cap * math.tanh(latent_delta)
    if not 0 < actual <= maximum["torque_vector_l2_bound_nm"]:
        raise ValueError("declared Gaussian-to-torque analytic bound violated")
    result = dict(
        schema="soccer.rsi.cerebellar_kl_action_authority.v1",
        source_pins={
            str(Path(module.__file__)): hash_bytes(Path(module.__file__).read_bytes())
            for module in (learner, policy)
        },
        optimizer_configuration=dict(config),
        latent_sigma=sigma,
        maximum_torque_cap_nm=maximum_cap,
        all_29_caps_nm=policy.CAPS.tolist(),
        rows=rows,
        zero_mean_single_joint_anchor_extremum_nm=actual,
        anchor_held_fixed_across_rounds=True,
        pointwise_same_feature_only=True,
        body_feedback_amplification_bounded_here=False,
        gaussian_sample_amplitude_bounded_here=False,
        physical_ineffectiveness_proven=False,
        limits_changed=False,
        new_physical_substeps=0,
        new_optimizer_updates=0,
        physical_skill_qualified=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("unused analytic evidence output required")
    result = budget_report()
    write_once(args.output, result)
    print(result["report_hash"], result["rows"], flush=True)


if __name__ == "__main__":
    main()
