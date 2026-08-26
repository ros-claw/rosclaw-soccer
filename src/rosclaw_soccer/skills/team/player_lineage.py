"""Soccer-owned player profiles backed by ROSClaw individual growth scopes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from rosclaw.continual.individual_scope import IndividualGrowthScope

from rosclaw_soccer.growth.role_learning import SoccerRole
from rosclaw_soccer.sim.contracts import hash_json

_ARCHETYPE = {
    SoccerRole.SHOOTER: "finisher",
    SoccerRole.PASSER: "playmaker",
    SoccerRole.GOALKEEPER: "goalkeeper",
}


@dataclass(frozen=True)
class SoccerPlayerProfile:
    """One player's private career state plus shared athlete foundation."""

    role: SoccerRole
    growth: IndividualGrowthScope
    retained_skill_hashes: tuple[str, ...]
    schema_version: str = "rosclaw_soccer.player_profile.v1"

    def __post_init__(self) -> None:
        if self.growth.agent_id != f"soccer.{self.archetype}":
            raise ValueError("player identity must be stable for its soccer archetype")
        if not self.retained_skill_hashes or any(
            not value.startswith("sha256:") for value in self.retained_skill_hashes
        ):
            raise ValueError("player profile requires content-addressed retained skills")

    @property
    def player_id(self) -> str:
        return str(self.growth.agent_id)

    @property
    def archetype(self) -> str:
        return str(_ARCHETYPE[self.role])

    @property
    def profile_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role.value,
            "archetype": self.archetype,
            "growth": self.growth.to_dict(),
            "retained_skill_hashes": list(self.retained_skill_hashes),
        }


@dataclass(frozen=True)
class SoccerTeamRoster:
    """Team versioning kept separate from each player's private promotion."""

    roster_version: int
    tactical_policy_hash: str
    players: tuple[SoccerPlayerProfile, ...]
    previous_roster_hash: str | None = None
    schema_version: str = "rosclaw_soccer.team_roster.v1"

    def __post_init__(self) -> None:
        if self.roster_version < 0:
            raise ValueError("team roster version must be non-negative")
        if not self.tactical_policy_hash.startswith("sha256:"):
            raise ValueError("team roster requires a tactical policy hash")
        if self.previous_roster_hash is not None and not self.previous_roster_hash.startswith(
            "sha256:"
        ):
            raise ValueError("previous roster must be content addressed")
        if self.roster_version == 0 and self.previous_roster_hash is not None:
            raise ValueError("initial roster cannot have a previous version")
        if self.roster_version > 0 and self.previous_roster_hash is None:
            raise ValueError("updated roster requires its previous roster hash")
        expected_roles = set(SoccerRole)
        if {player.role for player in self.players} != expected_roles or len(self.players) != len(
            expected_roles
        ):
            raise ValueError("team roster requires one independent player per soccer role")
        if len({player.player_id for player in self.players}) != len(self.players):
            raise ValueError("team roster player identities must be unique")
        if len({player.growth.personal_memory_namespace for player in self.players}) != len(
            self.players
        ):
            raise ValueError("team roster player memory namespaces must be private")
        if len({player.growth.foundation_policy_hash for player in self.players}) != 1:
            raise ValueError("team roster players must share one athlete foundation")

    @property
    def roster_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def player(self, role: SoccerRole) -> SoccerPlayerProfile:
        return next(player for player in self.players if player.role is role)

    def replace_promoted_player(self, promoted: SoccerPlayerProfile) -> SoccerTeamRoster:
        """Create a new roster without changing any other player's lineage."""

        current = self.player(promoted.role)
        if current.player_id != promoted.player_id:
            raise ValueError("promoted player identity changed")
        if current.growth.foundation_policy_hash != promoted.growth.foundation_policy_hash:
            raise ValueError("individual promotion cannot mutate the shared foundation")
        if promoted.growth.generation != current.growth.generation + 1:
            raise ValueError("promoted player generation must increment by one")
        if promoted.growth.last_promotion_evidence_hash is None:
            raise ValueError("promoted player has no individual promotion evidence")
        updated = tuple(
            promoted if player.role is promoted.role else player for player in self.players
        )
        return replace(
            self,
            roster_version=self.roster_version + 1,
            players=updated,
            previous_roster_hash=self.roster_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "roster_version": self.roster_version,
            "tactical_policy_hash": self.tactical_policy_hash,
            "players": [player.to_dict() for player in self.players],
            "previous_roster_hash": self.previous_roster_hash,
        }


__all__ = ["SoccerPlayerProfile", "SoccerTeamRoster"]
