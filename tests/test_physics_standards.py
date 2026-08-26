from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.physics.reality_pack import run_reality_pack
from rosclaw_soccer.physics.standards import IFABRegulationSpec


def test_regulation_dimensions_and_ball_are_explicit() -> None:
    spec = IFABRegulationSpec()

    assert spec.field_length_m == 105.0
    assert spec.field_width_m == 68.0
    assert spec.goal_inside_width_m == 7.32
    assert spec.goal_inside_height_m == 2.44
    assert spec.goal_frame_diameter_m <= 0.12
    assert 0.68 <= 2.0 * 3.141592653589793 * spec.ball_radius_m <= 0.70
    assert 0.410 <= spec.ball_mass_kg <= 0.450
    assert spec.to_dict()["net_depth_is_normative"] is False


def test_non_regulation_goal_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 7.32 x 2.44"):
        replace(IFABRegulationSpec(), goal_inside_width_m=3.0)


def test_reality_pack_outputs_must_be_external(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_reality_pack(output_dir=tmp_path / "evidence", source_checkout=tmp_path)


@pytest.mark.integration
def test_reality_pack_ball_rolls_and_net_retains(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    checkout = Path(__file__).resolve().parents[1]
    report = run_reality_pack(
        output_dir=tmp_path / "reality-pack",
        source_checkout=checkout,
    )

    assert report.passed
    assert Path(report.output_path).is_file()
    assert Path(report.curves_path).is_file()
    cases = {case.case_id: case for case in report.cases}
    assert cases["ball_roll"].metrics["rotation_observed"] is True
    assert float(cases["ball_roll"].metrics["stable_slip_ratio"]) <= 0.12
    assert cases["goal_net"].metrics["retained_in_goal"] is True
