"""Joint projection must not silently rewrite the frozen foundation."""

import numpy as np
import pytest

from rosclaw_soccer.sim.joint_target_projection import project_modified_joint_targets


def test_projects_only_modified_joint_and_reports_magnitude() -> None:
    result = project_modified_joint_targets(
        target_rad=np.asarray((2.0, 0.7, -0.2)),
        limits_rad=np.asarray(((-1.0, 1.0), (-0.5, 0.6), (-0.5, 0.5))),
        modified_indices=(1,),
    )
    np.testing.assert_allclose(result.target_rad, (2.0, 0.6, -0.2))
    assert result.projection_count == 1
    assert result.maximum_projection_rad == pytest.approx(0.1)


def test_excessive_projection_rejected() -> None:
    with pytest.raises(ValueError, match="excessive"):
        project_modified_joint_targets(
            target_rad=np.asarray((0.7876,)),
            limits_rad=np.asarray(((-0.873, 0.5236),)),
            modified_indices=(0,),
        )


@pytest.mark.parametrize(
    ("target", "limits", "indices"),
    [
        ((float("nan"),), ((-1.0, 1.0),), (0,)),
        ((0.0,), ((1.0, -1.0),), (0,)),
        ((0.0,), ((-1.0, 1.0),), (1,)),
        ((0.0,), ((-1.0, 1.0),), (0, 0)),
    ],
)
def test_invalid_projection_contract_rejected(target: tuple, limits: tuple, indices: tuple) -> None:
    with pytest.raises(ValueError):
        project_modified_joint_targets(
            target_rad=np.asarray(target, dtype=float),
            limits_rad=np.asarray(limits, dtype=float),
            modified_indices=indices,
        )
