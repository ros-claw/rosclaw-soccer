"""Strict player-card loader for academy-facing identity and capability state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_MAX_PLAYER_CARD_BYTES = 256 * 1024


@dataclass(frozen=True)
class PlayerCard:
    """Public, non-secret snapshot of one academy player's growth state."""

    player_id: str
    display_name: str
    embodiment: str
    academy_age: int
    age_title: str
    certification_status: str
    activation_ceiling: str
    generation: int
    parent_generation: str | None
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    next_exam: str
    schema_version: str = "rosclaw_soccer.player_card.v1"

    def __post_init__(self) -> None:
        if not self.player_id or not self.display_name:
            raise ValueError("player identity must be non-empty")
        if not 0 <= self.academy_age <= 18:
            raise ValueError("academy age must be in [0, 18]")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("the public academy activation ceiling must remain SIM_ONLY")
        if self.generation < 0:
            raise ValueError("player generation must be non-negative")
        if not self.strengths or not self.weaknesses:
            raise ValueError("player card must expose both strengths and weaknesses")
        if self.schema_version != "rosclaw_soccer.player_card.v1":
            raise ValueError("unsupported player-card schema")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_player_card(path: Path) -> PlayerCard:
    """Load one bounded JSON player card and reject unknown fields."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size > _MAX_PLAYER_CARD_BYTES:
        raise ValueError("player card is missing or exceeds the size limit")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("player card must be a JSON object")
    allowed = set(PlayerCard.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"player card contains unknown fields: {sorted(unknown)}")
    raw["strengths"] = tuple(raw.get("strengths", ()))
    raw["weaknesses"] = tuple(raw.get("weaknesses", ()))
    return PlayerCard(**raw)


__all__ = ["PlayerCard", "load_player_card"]
