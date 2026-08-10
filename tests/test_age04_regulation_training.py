from __future__ import annotations

from dataclasses import dataclass

import pytest

from rosclaw_soccer.training.age04_regulation import (
    Age04RegulationCurriculum,
    _config_from_json,
)


def test_regulation_curriculum_binds_eight_unique_probes() -> None:
    curriculum = Age04RegulationCurriculum()

    assert len(curriculum.teacher_force_pairs_n) == 8
    assert curriculum.teacher_force_pairs_n[:2] == ((250.0, 250.0), (245.0, 245.0))
    assert curriculum.precision_radius_m == 0.10
    assert (curriculum.target_y_m, curriculum.target_z_m) == (1.32, 1.04)


def test_regulation_curriculum_rejects_duplicate_probe() -> None:
    with pytest.raises(ValueError, match="unique"):
        Age04RegulationCurriculum(teacher_force_pairs_n=((10.0, 10.0),) * 8)


def test_seed_config_discards_old_schema_and_restores_tuple_fields() -> None:
    @dataclass(frozen=True)
    class Example:
        joint_gain_scales: tuple[float, ...]
        schema_version: str = "new"

    result = _config_from_json(
        Example,
        {"joint_gain_scales": [0.5, 1.0], "schema_version": "old", "removed": True},
    )

    assert result == Example(joint_gain_scales=(0.5, 1.0))
