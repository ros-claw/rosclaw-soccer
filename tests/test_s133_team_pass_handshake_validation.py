from __future__ import annotations

import inspect

from rosclaw_soccer.training.team_pass_handshake_validation import (
    validate_team_pass_handshake_discovery,
)


def test_team_handshake_validator_requires_explicit_path() -> None:
    parameters = inspect.signature(validate_team_pass_handshake_discovery).parameters
    assert tuple(parameters) == ("path",)
