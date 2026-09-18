"""Assemble an explicit private-player candidate without activating it.

Per-role learning need not overwrite the other seven players. This operation
copies already learned weights; it is neither another optimizer step nor proof
that the combined team retains its skills. Callers authenticate selection
evidence and must run fresh team physics before considering any promotion.
"""

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.sim.contracts import hash_json


def assemble_private_candidate(
    parent: NearBallResidualPolicy,
    replacements: Mapping[str, NearBallResidualPolicy],
    *,
    selection_evidence_hash: str,
) -> tuple[NearBallResidualPolicy, dict[str, Any]]:
    if (
        not isinstance(parent, NearBallResidualPolicy)
        or not isinstance(replacements, Mapping)
        or not replacements
        or len(replacements) > 8
        or not isinstance(selection_evidence_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", selection_evidence_hash) is None
    ):
        raise ValueError("explicit content-bound private candidate composition required")
    selected = dict(replacements)
    for agent, source in selected.items():
        if (
            not isinstance(agent, str)
            or agent not in parent.agent_ids
            or not isinstance(source, NearBallResidualPolicy)
            or source.agent_ids != parent.agent_ids
            or source.body_hash != parent.body_hash
            or source.observation_contract != parent.observation_contract
        ):
            raise ValueError("source must match the same player, body, roster and observations")
    weights = {key: value.copy() for key, value in parent.weights.items()}
    for agent, source in sorted(selected.items()):
        column = parent.agent_ids.index(agent)
        for key in weights:
            weights[key][column] = source.weights[key][column]
    candidate = NearBallResidualPolicy(
        parent.agent_ids,
        parent.body_hash,
        parent.generation + 1,
        parent.policy_hash,
        weights,
        parent.observation_contract,
    )
    changed = [
        agent
        for column, agent in enumerate(parent.agent_ids)
        if any(
            not np.array_equal(parent.weights[key][column], candidate.weights[key][column])
            for key in weights
        )
    ]
    report = dict(
        schema="soccer.private_policy_composition.v1",
        parent_hash=parent.policy_hash,
        candidate_hash=candidate.policy_hash,
        source_policy_hashes={
            agent: source.policy_hash for agent, source in sorted(selected.items())
        },
        selection_evidence_hash=selection_evidence_hash,
        changed_agent_ids=changed,
        unchanged_agent_ids=[agent for agent in parent.agent_ids if agent not in changed],
        optimizer_steps=0,
        selection_evidence_authenticated=False,
        requires_team_replay=True,
        activation_ceiling="SIM_ONLY",
        promoted=False,
    )
    report["manifest_hash"] = hash_json(report)
    return candidate, report
