from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rosclaw_soccer.training.three_role_save_portfolio as portfolio_module
from rosclaw_soccer.skills.team.shared_world import G1GoalkeeperConfig
from rosclaw_soccer.training.three_role_save_portfolio import (
    ThreeRoleSaveLane,
    ThreeRoleSavePortfolioConfig,
    run_three_role_save_portfolio_evidence,
    three_role_save_lane_kwargs,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def test_save_portfolio_is_sim_only_and_requires_diverse_lanes() -> None:
    config = ThreeRoleSavePortfolioConfig()
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.hardware_authorized is False
    assert len(config.lanes) == 4
    assert config.minimum_contact_span_m == pytest.approx(0.75)
    with pytest.raises(ValueError, match="at least three"):
        replace(config, lanes=config.lanes[:2])
    with pytest.raises(ValueError, match="unique"):
        replace(config, lanes=(config.lanes[0], config.lanes[0], config.lanes[1]))
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)


def test_save_lane_rejects_unqualified_goalkeeper_pocket() -> None:
    with pytest.raises(ValueError, match="goalkeeper pocket"):
        ThreeRoleSaveLane("bad-lane", "BAD", -0.8, 0.0)
    with pytest.raises(ValueError, match="offsets"):
        ThreeRoleSaveLane("bad-lane", "BAD", -1.3, -1.2)
    with pytest.raises(ValueError, match="initial lateral"):
        G1GoalkeeperConfig(initial_lateral_position_m=1.51)


def test_save_lane_translation_preserves_local_strike_pocket(tmp_path: Path) -> None:
    artifacts = tuple(tmp_path / name for name in ("striker", "goalkeeper", "gmt", "skill"))
    for path in artifacts:
        path.write_text("bound", encoding="utf-8")
    lane = ThreeRoleSaveLane("center-channel", "CENTER", -0.45, -0.35)
    kwargs = three_role_save_lane_kwargs(
        lane=lane,
        striker_actor_path=artifacts[0],
        goalkeeper_actor_path=artifacts[1],
        gmt_model_path=artifacts[2],
        gmt_skill_path=artifacts[3],
    )
    assert kwargs["shooter_origin"] == pytest.approx((0.0, -0.45, 0.0))
    assert kwargs["pass_reception_target_m"] == pytest.approx((1.275, -0.47, 0.115))
    assert kwargs["passer_origin"] == pytest.approx((5.10, -0.614060065039216, 0.0))
    assert kwargs["shooter_target"] == pytest.approx((7.50, 0.44, 0.115))
    assert kwargs["goal_spec"].target_y_m == pytest.approx(0.44)
    assert kwargs["goalkeeper_config"].initial_lateral_position_m == pytest.approx(-0.35)
    # The goal and receiving pocket moved together with the front G1; the
    # frozen policy still sees the exact same local target.
    assert kwargs["shooter_policy_target"] == pytest.approx((7.50, 0.70, 0.50))


@pytest.mark.parametrize("drift_kind", ["source", "artifact", "request"])
def test_save_portfolio_rejects_runtime_drift_but_preserves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift_kind: str
) -> None:
    artifacts = tuple(tmp_path / name for name in ("striker", "goalkeeper", "gmt", "skill"))
    for path in artifacts:
        path.write_text("bound", encoding="utf-8")
    qualification = SimpleNamespace(
        body_hash="body",
        kick_prior_hash="kick",
        require_eligible=lambda: None,
    )
    monkeypatch.setattr(portfolio_module, "qualify_g1_assets", lambda _root: qualification)
    monkeypatch.setattr(portfolio_module, "_git_head", lambda _checkout: "commit")
    hashes = iter(("source-before", "source-after" if drift_kind == "source" else "source-before"))
    monkeypatch.setattr(portfolio_module, "_implementation_hash", lambda: next(hashes))
    monkeypatch.setattr(
        portfolio_module,
        "three_role_save_lane_kwargs",
        lambda **values: {
            "lane_id": values["lane"].lane_id,
            "goal_spec": G1TrainingGoalSpec(),
        },
    )
    monkeypatch.setattr(portfolio_module, "trajectory_digest", lambda _trajectory: "same")

    class Result:
        def to_dict(self) -> dict[str, bool]:
            return {"passed": True}

    lateral = {
        "right-channel": (0.5, "right"),
        "center-channel": (0.2, "right"),
        "left-channel": (-0.2, "left"),
        "far-left-channel": (-0.5, "left"),
    }
    monkeypatch.setattr(
        portfolio_module,
        "evaluate_three_role_save_lane",
        lambda **values: {
            "passed": True,
            "glove_contact_position_m": (
                7.0,
                lateral[str(values["trajectory"]["lane_id"].item())][0],
                1.4,
            ),
            "glove_contact_side": lateral[str(values["trajectory"]["lane_id"].item())][1],
        },
    )

    simulate_count = 0

    def fake_simulate(_root: Path, **kwargs: object) -> tuple[Result, dict[str, np.ndarray]]:
        nonlocal simulate_count
        simulate_count += 1
        if drift_kind == "artifact" and simulate_count == 2:
            artifacts[0].write_text("changed", encoding="utf-8")
        if drift_kind == "request" and simulate_count == 2:
            (output / "request.json").write_text("changed", encoding="utf-8")
        return Result(), {"lane_id": np.array(kwargs["lane_id"]), "time": np.array([0, 1])}

    monkeypatch.setattr(portfolio_module, "simulate_shared_world", fake_simulate)
    output = tmp_path / "evidence"
    report = run_three_role_save_portfolio_evidence(
        asset_root=tmp_path,
        striker_actor_path=artifacts[0],
        goalkeeper_actor_path=artifacts[1],
        gmt_model_path=artifacts[2],
        gmt_skill_path=artifacts[3],
        output_dir=output,
        source_checkout=tmp_path / "source",
    )
    assert report["portfolio_gates"]["source_stable_during_run"] is (drift_kind != "source")
    assert report["portfolio_gates"]["artifacts_stable_during_run"] is (drift_kind != "artifact")
    assert report["portfolio_gates"]["request_stable_during_run"] is (drift_kind != "request")
    assert report["passed"] is False
    assert report["promotion_status"] == "REJECTED_DEVELOPMENT"
    assert (output / "evidence.json").is_file()
    assert len(tuple(output.glob("*-trajectory.npz"))) == 4
