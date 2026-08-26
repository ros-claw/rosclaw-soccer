from __future__ import annotations

import hashlib

from rosclaw.continual.contracts import PolicyVersion
from rosclaw.continual.individual_scope import (
    FrozenPartnerSet,
    IndividualGrowthScope,
    IndividualPromotionEvidence,
)

from rosclaw_soccer.growth.role_learning import SoccerRole
from rosclaw_soccer.skills.team.player_lineage import SoccerPlayerProfile, SoccerTeamRoster


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _policy(role: SoccerRole, version: int, parent: PolicyVersion | None = None) -> PolicyVersion:
    return PolicyVersion(
        version=version,
        artifact_hash=_hash(f"{role}.policy.{version}"),
        parent_version_hash=None if parent is None else parent.version_hash,
        controller_snapshot_hash=_hash(f"{role}.controller"),
        body_hash=_hash("g1.body"),
        safety_kernel_hash=_hash("g1.safety"),
        observation_names=("proprio",),
        residual_action_names=("residual",),
    )


def _player(role: SoccerRole) -> SoccerPlayerProfile:
    champion = _policy(role, 0)
    archetype = {
        SoccerRole.SHOOTER: "finisher",
        SoccerRole.PASSER: "playmaker",
        SoccerRole.GOALKEEPER: "goalkeeper",
    }[role]
    growth = IndividualGrowthScope(
        agent_id=f"soccer.{archetype}",
        body_hash=champion.body_hash,
        body_state_hash=_hash(f"{role}.body-state"),
        foundation_policy_hash=_hash("athlete.foundation.v1"),
        personal_adapter_hash=_hash(f"{role}.adapter"),
        role_policy_hash=_hash(f"{role}.role"),
        residual_policy_hash=_hash(f"{role}.residual"),
        capability_profile_hash=_hash(f"{role}.capability"),
        career_lineage_hash=_hash(f"{role}.career"),
        personal_memory_namespace=f"soccer.{archetype}.memory",
        failure_memory_namespace=f"soccer.{archetype}.failure",
        parent_policy=champion,
        champion_policy=champion,
    )
    return SoccerPlayerProfile(
        role=role,
        growth=growth,
        retained_skill_hashes=(_hash(f"{role}.retained"),),
    )


def test_roster_shares_foundation_but_not_private_memory_or_champions() -> None:
    roster = SoccerTeamRoster(
        roster_version=0,
        tactical_policy_hash=_hash("tactics.v0"),
        players=tuple(_player(role) for role in SoccerRole),
    )

    assert len({player.growth.foundation_policy_hash for player in roster.players}) == 1
    assert len({player.growth.personal_memory_namespace for player in roster.players}) == 3
    assert len({player.growth.champion_policy.version_hash for player in roster.players}) == 3


def test_promoting_goalkeeper_preserves_both_forward_lineages() -> None:
    roster = SoccerTeamRoster(
        roster_version=0,
        tactical_policy_hash=_hash("tactics.v0"),
        players=tuple(_player(role) for role in SoccerRole),
    )
    keeper = roster.player(SoccerRole.GOALKEEPER)
    candidate = _policy(SoccerRole.GOALKEEPER, 1, keeper.growth.champion_policy)
    partners = FrozenPartnerSet(
        focal_agent_id=keeper.player_id,
        partners=(),
        numerical_contract_hash=_hash("numerical"),
        scenario_contract_hash=_hash("scenario"),
    )
    staged = keeper.growth.stage_candidate(candidate, partners=partners)
    evidence = IndividualPromotionEvidence(
        agent_id=keeper.player_id,
        parent_policy_hash=keeper.growth.champion_policy.version_hash,
        candidate_policy_hash=candidate.version_hash,
        frozen_partner_snapshot_hash=partners.snapshot_hash,
        matched_seed_commitment_hash=_hash("seeds"),
        gate_report_hash=_hash("gate"),
        personal_memory_namespace=keeper.growth.personal_memory_namespace,
        retention_passed=True,
        safety_passed=True,
        team_compatibility_passed=True,
    )
    promoted_keeper = SoccerPlayerProfile(
        role=keeper.role,
        growth=staged.promote_candidate(evidence),
        retained_skill_hashes=keeper.retained_skill_hashes,
    )

    updated = roster.replace_promoted_player(promoted_keeper)

    assert updated.player(SoccerRole.GOALKEEPER).growth.generation == 1
    for role in (SoccerRole.PASSER, SoccerRole.SHOOTER):
        assert updated.player(role) == roster.player(role)
