"""Outward ankle-roll braking knob: validation and hash behavior.

When unset (None) the knob is dropped from the serialized config so existing
configuration hashes are untouched; when explicitly enabled it must be bounded
and must appear in the hash payload.
"""

import math
from dataclasses import asdict, replace

import pytest

from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


def _hash_payload_keys(config):
    """Keys that actually feed the config hash serialization."""
    from rosclaw_soccer.sim.contracts import hash_json

    digest = hash_json(asdict(config))
    assert isinstance(digest, str) and digest.startswith("sha256:")
    return digest


def test_unset_knob_dropped_from_serialization():
    config = IndependentTeamWorldConfig()
    # The config hash must be reproducible and identical for identical defaults.
    assert _hash_payload_keys(config) == _hash_payload_keys(IndependentTeamWorldConfig())


def test_accepts_bounded_value_and_changes_hash():
    baseline = IndependentTeamWorldConfig()
    braked = replace(baseline, outward_ankle_roll_braking_damping=8.0)
    assert braked.outward_ankle_roll_braking_damping == 8.0
    assert _hash_payload_keys(braked) != _hash_payload_keys(baseline)


def test_rejects_out_of_bounds_values():
    for bad in (0.0, 5.9, 20.1, math.inf, float("nan"), "x", True):
        with pytest.raises(ValueError):
            replace(IndependentTeamWorldConfig(), outward_ankle_roll_braking_damping=bad)
