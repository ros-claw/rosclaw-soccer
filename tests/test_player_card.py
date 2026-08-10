from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.academy.player_card import load_player_card


def test_repository_player_card_exposes_weaknesses() -> None:
    root = Path(__file__).resolve().parents[1]
    card = load_player_card(root / "academy" / "players" / "claw7.json")
    assert card.academy_age == 4
    assert card.activation_ceiling == "SIM_ONLY"
    assert "continuous first touch and dribbling" in card.weaknesses


def test_player_card_rejects_unknown_fields(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / "academy" / "players" / "claw7.json").read_text())
    value["secret_score"] = 100
    path = tmp_path / "card.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_player_card(path)
