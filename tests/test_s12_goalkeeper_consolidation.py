from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw_soccer.training.goalkeeper_consolidation import (
    consolidate_goalkeeper_action_channels,
    consolidate_goalkeeper_checkpoint,
    consolidate_goalkeeper_hierarchical_checkpoint,
)


def test_consolidation_validates_scale_before_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scale"):
        consolidate_goalkeeper_checkpoint(
            source_checkpoint=tmp_path / "missing.pt",
            output_checkpoint=tmp_path / "candidate.pt",
            actor_scale=0.0,
        )


def test_consolidation_requires_new_output(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    with pytest.raises(ValueError, match="new output"):
        consolidate_goalkeeper_checkpoint(
            source_checkpoint=source,
            output_checkpoint=source,
            actor_scale=0.5,
        )


def test_consolidation_scales_only_actor_and_variance(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pt"
    output = tmp_path / "candidate.pt"
    torch.save(
        {
            "state_dict": {
                "actor.weight": torch.full((2, 3), 2.0),
                "actor.bias": torch.full((2,), 4.0),
                "log_std": torch.zeros(2),
                "critic.weight": torch.full((1, 3), 7.0),
            },
            "promotion_status": "SOURCE",
        },
        source,
    )
    report = consolidate_goalkeeper_checkpoint(
        source_checkpoint=source,
        output_checkpoint=output,
        actor_scale=0.5,
    )
    candidate = torch.load(output, map_location="cpu", weights_only=True)
    state = candidate["state_dict"]
    assert torch.equal(state["actor.weight"], torch.ones((2, 3)))
    assert torch.equal(state["actor.bias"], torch.full((2,), 2.0))
    assert torch.allclose(state["log_std"], torch.full((2,), -0.69314718))
    assert torch.equal(state["critic.weight"], torch.full((1, 3), 7.0))
    assert report["actor_scale"] == 0.5
    assert not report["selection_authority"]
    assert candidate["promotion_status"] == "CANDIDATE_PENDING_EVALUATION"


def test_hierarchical_consolidation_keeps_core_stable_and_arms_plastic(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    parent = tmp_path / "parent.pt"
    candidate = tmp_path / "candidate.pt"
    output = tmp_path / "merged.pt"
    parent_state = {
        "trunk.0.weight": torch.zeros((2, 2)),
        "actor.weight": torch.zeros((18, 2)),
        "actor.bias": torch.zeros(18),
        "log_std": torch.zeros(18),
        "critic.weight": torch.zeros((1, 2)),
    }
    candidate_state = {name: torch.ones_like(value) for name, value in parent_state.items()}
    torch.save({"state_dict": parent_state}, parent)
    torch.save({"state_dict": candidate_state}, candidate)

    report = consolidate_goalkeeper_hierarchical_checkpoint(
        parent_checkpoint=parent,
        candidate_checkpoint=candidate,
        output_checkpoint=output,
        trunk_plasticity=0.20,
        core_action_plasticity=0.10,
        arm_action_plasticity=0.75,
    )
    merged = torch.load(output, map_location="cpu", weights_only=True)["state_dict"]
    assert torch.allclose(merged["trunk.0.weight"], torch.full((2, 2), 0.20))
    assert torch.allclose(merged["actor.weight"][:4], torch.full((4, 2), 0.10))
    assert torch.allclose(merged["actor.weight"][4:], torch.full((14, 2), 0.75))
    assert torch.equal(merged["critic.weight"], torch.ones((1, 2)))
    assert report["report_hash"].startswith("sha256:")
    assert not report["selection_authority"]


def test_action_channel_consolidation_scales_core_and_arms_separately(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pt"
    output = tmp_path / "merged.pt"
    torch.save(
        {
            "state_dict": {
                "actor.weight": torch.ones((18, 2)),
                "actor.bias": torch.ones(18),
                "log_std": torch.zeros(18),
                "critic.weight": torch.ones((1, 2)),
            }
        },
        source,
    )
    report = consolidate_goalkeeper_action_channels(
        source_checkpoint=source,
        output_checkpoint=output,
        core_action_scale=0.25,
        arm_action_scale=0.80,
    )
    merged = torch.load(output, map_location="cpu", weights_only=True)["state_dict"]
    assert torch.allclose(merged["actor.weight"][:4], torch.full((4, 2), 0.25))
    assert torch.allclose(merged["actor.weight"][4:], torch.full((14, 2), 0.80))
    assert torch.equal(merged["critic.weight"], torch.ones((1, 2)))
    assert report["report_hash"].startswith("sha256:")
